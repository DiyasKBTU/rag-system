#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reindex.py — Переиндексировать сайт университета.

Запуск из корня проекта:
    python tools/reindex.py

Что делает:
    Скачивает все страницы сайта, разбивает на фрагменты,
    генерирует вопросы через GPT и сохраняет в базу Qdrant.
    Страницы которые не изменились — пропускает автоматически.

Стоимость: ~$0.20–0.30 за полную переиндексацию (204 страницы).
"""

import subprocess
import sys
import os

# Переходим в папку rag_service чтобы правильно импортировались модули
os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.parser.__main__ import run_indexing

if __name__ == "__main__":
    run_indexing()
