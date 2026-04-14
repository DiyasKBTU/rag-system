import logging
import traceback

from rest_framework.generics import ListAPIView
from .models import Menu
from .serializers import MenuSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from openai import OpenAI
from .services.rag_client import get_rag_context
from .services.google_docs import get_google_doc_text

logger = logging.getLogger(__name__)


class MenuListAPIView(ListAPIView):
    queryset = Menu.objects.all().order_by("order")
    serializer_class = MenuSerializer

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx["lang"] = self.request.query_params.get("lang", "ru")
        return ctx

    def get_queryset(self):
        parent_id = self.request.query_params.get("parent")
        qs = self.queryset.order_by("order")
        if parent_id:
            return qs.filter(parent_id=parent_id)
        return qs.filter(parent__isnull=True)


def _lang_name(lang: str) -> str:
    return {"kk": "Kazakh", "ru": "Russian", "en": "English"}.get(lang, "Russian")


def get_error_fallback_text(lang: str) -> str:
    """Используется только при технической ошибке (exception), не как ответ GPT."""
    if lang == "kk":
        return (
            "Кешіріңіз, техникалық қате орын алды 😔\n"
            "Сұрағыңызды қайталап көріңіз немесе қабылдау комиссиясына хабарласыңыз:\n"
            "📞 +7 707 510 10 10"
        )
    if lang == "en":
        return (
            "Sorry, a technical error occurred 😔\n"
            "Please try again or contact the admissions office directly:\n"
            "📞 +7 707 510 10 10"
        )
    return (
        "Извините, произошла техническая ошибка 😔\n"
        "Попробуйте повторить вопрос или свяжитесь с приёмной комиссией напрямую:\n"
        "📞 +7 707 510 10 10"
    )


class AskFromDocAPIView(APIView):
    def post(self, request):
        data = request.data or {}
        question = (data.get("question") or "").strip()
        lang = (data.get("lang") or "ru").lower()

        if lang not in ("kk", "ru", "en"):
            lang = "ru"

        if not question:
            return Response(
                {"detail": "question is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # ═══════════════════════════════════════════════════════
            # ПРИОРИТЕТ ИСТОЧНИКОВ ЗНАНИЙ:
            #
            # 1. RAG-сервис (Qdrant + FastAPI) — семантический поиск
            #    по проиндексированным страницам сайта.
            #    Запускается из rag_service/run_api.py.
            #
            # 2. Fallback: Google Docs + парсинг сайта
            #    Используется если RAG недоступен или ничего не нашёл.
            # ═══════════════════════════════════════════════════════

            knowledge_source: str  # "rag" | "fallback"

            # ── Шаг 0: для казахского — переводим запрос на русский ──
            # База знаний индексирована только на русском языке.
            # Эмбеддинги казахского текста плохо совпадают с русскими
            # чанками — перевод запроса резко улучшает качество поиска.
            rag_question = question
            if lang == "kk":
                try:
                    client_translate = OpenAI(api_key=settings.OPENAI_API_KEY)
                    tr = client_translate.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Ты переводчик. Переведи вопрос абитуриента "
                                    "с казахского языка на русский. "
                                    "Верни ТОЛЬКО перевод, без пояснений."
                                ),
                            },
                            {"role": "user", "content": question},
                        ],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    rag_question = tr.choices[0].message.content.strip()
                    logger.info(f"[AskView] Перевод для RAG: '{question}' → '{rag_question}'")
                except Exception as e:
                    logger.warning(f"[AskView] Перевод не удался, ищем оригинал: {e}")
                    rag_question = question  # fallback — ищем как есть

            # ── Шаг 1: пробуем RAG-сервис ─────────────────────────
            rag_context = get_rag_context(rag_question)

            if rag_context:
                knowledge_text = rag_context
                knowledge_source = "rag"
                logger.info(f"[AskView] Источник: RAG ({len(rag_context)} симв)")
            else:
                # ── Шаг 1б: fallback — Google Docs ────────────────
                knowledge_text = get_google_doc_text()
                if knowledge_text:
                    knowledge_source = "fallback"
                    logger.info(f"[AskView] Источник: Google Docs ({len(knowledge_text)} симв)")
                else:
                    # Оба источника недоступны — GPT отвечает сам
                    knowledge_source = "none"
                    logger.info("[AskView] Источники недоступны — GPT без контекста")

            # ── Шаг 2: формируем промпт ────────────────────────────

            # Дополнительная инструкция для казахского языка
            kk_extra = ""
            if lang == "kk":
                kk_extra = """
            КАЗАХСКИЙ ЯЗЫК — ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА:
            — База знаний на русском языке — переводи информацию на казахский в ответе
            — Отвечай ТОЛЬКО на казахском языке (қазақ тілінде жауап бер)
            — Используй корректную казахскую грамматику и университетскую лексику
            — Казахские термины: мамандық (специальность), жатақхана (общежитие),
              оқуға түсу (поступление), грант (грант), оқу ақысы (стоимость обучения),
              қабылдау комиссиясы (приёмная комиссия), құжаттар (документы)
            """

            # Инструкция по поведению при отсутствии информации
            no_info_instructions = {
                "ru": (
                    "Если в базе знаний нет ответа на вопрос — отвечай по-человечески, тепло. "
                    "Например: «К сожалению, у меня пока нет точной информации по этому вопросу 🙏 "
                    "Лучше всего уточнить напрямую в приёмной комиссии — они точно помогут: "
                    "📞 +7 707 510 10 10»"
                ),
                "kk": (
                    "Егер базада жауап болмаса — жылы, адамша жауап бер. "
                    "Мысалы: «Өкінішке орай, бұл сұрақ бойынша нақты ақпарат жоқ 🙏 "
                    "Қабылдау комиссиясына тікелей хабарласқан дұрыс: 📞 +7 707 510 10 10»"
                ),
                "en": (
                    "If the knowledge base has no answer — respond warmly and humanly. "
                    "For example: «Unfortunately, I don't have exact information on this yet 🙏 "
                    "It's best to check directly with the admissions office: 📞 +7 707 510 10 10»"
                ),
            }

            system_prompt = f"""
            Ты — Айдана, дружелюбный консультант приёмной комиссии ЦАИУ
            (Центральноазиатский инновационный университет).

            КТО ТЫ:
            — Живой, тёплый человек, а не робот-автоответчик
            — Помогаешь абитуриентам и родителям разобраться с поступлением
            — Говоришь просто и понятно, как опытный консультант

            ПРАВИЛА:
            — Используй ТОЛЬКО информацию из базы знаний ниже
            — Не придумывай факты, цифры, даты, баллы
            — Не обещай поступление, грант или заселение
            — Не упоминай, что ты ИИ
            — Не указывай источник информации в ответе

            СТИЛЬ:
            — Тёплый, живой, дружелюбный — как консультант вуза, а не как официальный документ
            — Можно использовать 1-2 уместных эмодзи
            — Структурированно: короткие абзацы или списки когда это удобно
            — Отвечай СТРОГО на языке запроса: {_lang_name(lang)}

            ЕСЛИ ИНФОРМАЦИИ НЕТ:
            {no_info_instructions.get(lang, no_info_instructions["ru"])}
            {kk_extra}
            БАЗА ЗНАНИЙ:
            {knowledge_text if knowledge_text else "[база знаний не содержит данных по этому вопросу]"}
            """.strip()

            # ── Шаг 3: запрашиваем GPT ────────────────────────────
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.5,
                max_tokens=600,
            )
            answer = resp.choices[0].message.content.strip()

            # ── Шаг 4: RAG-индикатор ──────────────────────────────
            # used_rag = True  → ответ построен на чанках из Qdrant
            # used_rag = False → fallback (Google Docs) или нет данных
            #
            # Определяем "нет информации" по ключевым фразам в ответе
            no_info_signals = [
                "нет точной информации",
                "нет нақты ақпарат",
                "don't have exact information",
                "+7 707 510 10 10",  # если бот дал номер — значит не нашёл
            ]
            is_no_info = any(signal in answer for signal in no_info_signals)
            used_rag: bool = (knowledge_source == "rag") and (not is_no_info)
            # knowledge_source: "rag" | "fallback" | "none" | "error"

            logger.info(
                f"[AskView] source={knowledge_source} "
                f"is_no_info={is_no_info} used_rag={used_rag}"
            )

            return Response({
                "answer": answer,
                "used_rag": used_rag,
                "knowledge_source": knowledge_source,
            })

        except Exception as e:
            print("OPENAI ERROR repr:", repr(e))
            traceback.print_exc()
            return Response(
                {
                    "answer": get_error_fallback_text(lang),
                    "used_rag": False,
                    "knowledge_source": "error",
                },
                status=200,
            )
