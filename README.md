# Langchain RAG Tutorial

Проект для создания и использования векторной базы данных с помощью Langchain и OpenAI.

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository_url>
cd langchain-rag-tutorial
```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv .venv
source .venv/bin/activate  # для Linux/Mac
.venv\Scripts\activate     # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Настройка переменных окружения:
- Скопируйте файл `.env.example` в новый файл `.env`
- Заполните `.env` вашими реальными значениями:
  ```
  OPENAI_API_KEY=your_actual_api_key
  ```

## Использование

1. Создание базы данных:
```bash
python create_database.py
```

2. Запрос к базе данных:
```bash
python query_data.py "ваш вопрос здесь"
```

## Структура проекта

- `create_database.py` - скрипт для создания векторной базы данных
- `query_data.py` - скрипт для выполнения запросов к базе данных
- `data/` - директория с исходными текстами
- `chroma/` - директория с векторной базой данных (создается автоматически)

## Безопасность

- Никогда не коммитьте файл `.env` в репозиторий
- Используйте `.env.example` как шаблон для создания `.env`
- Для разных окружений (development, production) используйте разные `.env` файлы

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `