from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


BACK_LABELS = {
    "kk": "⬅ Артқа",
    "ru": "⬅ Назад",
    "en": "⬅ Back",
}

LANG_CHANGE_LABELS = {
    "kk": "🌐 Тілді өзгерту",
    "ru": "🌐 Сменить язык",
    "en": "🌐 Change language",
}


def main_menu_keyboard(
    menus, lang: str = "ru", show_back: bool = False, show_lang_change: bool = False
):
    lang = (lang or "ru").lower()
    keyboard = []

    for menu in menus:
        keyboard.append([KeyboardButton(text=menu["title"])])

    if show_back:
        keyboard.append([KeyboardButton(text=BACK_LABELS.get(lang, BACK_LABELS["ru"]))])

    if show_lang_change:
        keyboard.append(
            [
                KeyboardButton(
                    text=LANG_CHANGE_LABELS.get(lang, LANG_CHANGE_LABELS["ru"])
                )
            ]
        )

    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)
