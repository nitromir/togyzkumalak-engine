# План интеграции AlphaZero и модернизации Тренировочного Центра

Этот документ описывает шаги по интеграции фреймворка `alpha-zero-general` и улучшению UX/UI процесса обучения.

## Фаза 1: Интеграция AlphaZero Core
1. **Adapter Creation**: Создать `backend/alphazero_adapter.py`.
    * Обернуть `TogyzkumalakBoard` в интерфейс `Game` из alpha-zero-general.
    * Реализовать каноническую форму доски (всегда от лица текущего игрока).
2. **Dual-Head Network**: Создать `backend/alphazero_network.py`.
    * Реализовать архитектуру с двумя головами: Policy (ходы) и Value (оценка позиции).
    * Добавить поддержку загрузки весов из текущих моделей для ускорения старта.
3. **Coach Setup**: Интегрировать логику `Coach.py` для управления циклом Self-Play -> Train -> Evaluate.

## Фаза 2: Инфраструктура Backend & API
1. **Background Tasks**: Реализовать `backend/task_manager.py` для выноса обучения в фоновые процессы.
    * Использовать `apscheduler` или `threading + queue`.
2. **Training API**:
    * `POST /api/training/alphazero/start` - запуск итерации.
    * `GET /api/training/alphazero/status` - прогресс текущей задачи.
    * `GET /api/training/alphazero/metrics` - данные для графиков (Loss, ELO).
3. **Metrics Storage**: Хранение истории обучения в `logs/training_metrics.json`.

## Фаза 3: Frontend & UI/UX (Dashboard & Real-time)
1. **Training Dashboard**: 
    * Визуализация Loss (Policy/Value) и Accuracy.
    * График оценочного ELO в реальном времени.
2. **Probabilistic UX**:
    * Обновить API хода, чтобы возвращать распределение вероятностей.
    * Добавить в UI "индикаторы уверенности" (progress bars) рядом с каждой лункой.
3. **Notifications**:
    * Реализовать Toast-уведомления о завершении итераций обучения.
    * Индикация "Модель обучается..." в шапке сайта.

## Фаза 4: Данные и Оптимизация
1. **Bootstrap**: Использование данных из `gemini_battle` для инициализации AlphaZero.
2. **Parallelism**: Оптимизация MCTS для работы на нескольких ядрах CPU.

---
*Статус: В разработке*
[ ] Адаптер игры
[ ] Нейросеть (Dual-head)
[ ] Таск-менеджер
[ ] Дашборд
[ ] Вероятностный UI*
