#!/bin/bash
# install.sh — Устанавливает systemd-юниты для CAIU бота и RAG-сервиса.
#
# Использование:
#   cd /home/YOUR_USER/caiu-bot/deploy
#   chmod +x install.sh
#   sudo ./install.sh YOUR_USER /home/YOUR_USER/caiu-bot
#
# Аргументы:
#   $1 — имя пользователя (например: ubuntu, caiu, deploy)
#   $2 — абсолютный путь к проекту (например: /home/ubuntu/caiu-bot)
#
# После установки:
#   sudo systemctl status caiu-rag
#   sudo systemctl status caiu-bot
#   journalctl -u caiu-rag -f      # логи RAG в реальном времени
#   journalctl -u caiu-bot -f      # логи бота в реальном времени

set -e

USER_NAME="${1:-YOUR_USER}"
PROJECT_DIR="${2:-/home/${USER_NAME}/caiu-bot}"

if [[ "$USER_NAME" == "YOUR_USER" ]]; then
    echo "ОШИБКА: укажите имя пользователя первым аргументом"
    echo "Пример: sudo ./install.sh ubuntu /home/ubuntu/caiu-bot"
    exit 1
fi

echo "Установка systemd-юнитов..."
echo "  Пользователь: $USER_NAME"
echo "  Проект:       $PROJECT_DIR"
echo ""

# Подставляем реальные значения в шаблоны и копируем в systemd
for SERVICE in caiu-rag caiu-bot; do
    sed \
        -e "s|/home/YOUR_USER/caiu-bot|${PROJECT_DIR}|g" \
        -e "s|YOUR_USER|${USER_NAME}|g" \
        "${SERVICE}.service" > "/etc/systemd/system/${SERVICE}.service"
    echo "  ✓ /etc/systemd/system/${SERVICE}.service"
done

# Перезагружаем конфиги и включаем автозапуск
systemctl daemon-reload

systemctl enable caiu-rag
systemctl enable caiu-bot

echo ""
echo "Готово! Запустить прямо сейчас:"
echo "  sudo systemctl start caiu-rag && sudo systemctl start caiu-bot"
echo ""
echo "Проверить статус:"
echo "  sudo systemctl status caiu-rag caiu-bot"

