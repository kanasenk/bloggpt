import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests

app = FastAPI()

# Получаем API ключи из переменных окружения
openai.api_key = os.getenv("OPENAI_API_KEY")      # Ключ OpenAI
currentsapi_key = os.getenv("CURRENTS_API_KEY")   # Ключ Currents API

# Проверяем, что оба API ключа заданы, иначе выбрасываем ошибку
if not openai.api_key or not currentsapi_key:
    raise ValueError("Переменные окружения OPENAI_API_KEY и CURRENTS_API_KEY должны быть установлены")


class Topic(BaseModel):
    topic: str  # Модель данных для получения темы в запросе


# Функция для получения последних новостей на заданную тему
def get_recent_news(topic: str):
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": "en",
        "keywords": topic,
        "apiKey": currentsapi_key,
    }
    # Чтобы не висеть вечно — ограничиваем время ожидания ответа
    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {response.text}")

    news_data = response.json().get("news", [])
    if not news_data:
        return "Свежих новостей не найдено."

    # берем только 5 заголовков
    return "\n".join([article["title"] for article in news_data[:5]])


# Функция для генерации контента на основе темы и новостей
def generate_content(topic: str):
    recent_news = get_recent_news(topic)

    try:
        # ==== 1. Заголовок (сильно урезаем max_tokens) ====
        title = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', "
                    f"с учётом актуальных новостей:\n{recent_news}\n"
                    f"Заголовок должен быть коротким (до 12 слов), интересным и ясно передавать суть темы."
                )
            }],
            max_tokens=24,          # было 60 — уменьшили, чтобы ускорить ответ
            temperature=0.5,
            stop=["\n"],
            request_timeout=20,     # на всякий случай ограничиваем время запроса к OpenAI
        ).choices[0].message.content.strip()

        # ==== 2. Мета-описание (тоже урезаем) ====
        meta_description = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Напишите сжатое, но информативное мета-описание для статьи с заголовком: '{title}'. "
                    f"До 2–3 предложений, с основными ключевыми словами по теме '{topic}'."
                )
            }],
            max_tokens=64,          # было 120 — этого достаточно для 2–3 предложений
            temperature=0.5,
            stop=["\n"],
            request_timeout=20,
        ).choices[0].message.content.strip()

        # ==== 3. Полный текст статьи (главный «тяжёлый» вызов) ====
        post_content = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Напишите подробную статью на тему '{topic}', используя последние новости:\n"
                    f"{recent_news}\n\n"
                    "Требования к статье:\n"
                    "1. Объём — не менее 1500–2500 символов.\n"
                    "2. Чёткая структура с подзаголовками.\n"
                    "3. Вступление, основная часть и заключение.\n"
                    "4. Включите анализ текущих трендов и примеры из актуальных новостей.\n"
                    "5. Каждый абзац — не менее 3–4 предложений.\n"
                    "Пишите на русском языке, понятно и информативно."
                )
            }],
            max_tokens=600,         # было 1500 — сильно уменьшили, но этого достаточно для статьи
            temperature=0.5,
            presence_penalty=0.6,
            frequency_penalty=0.6,
            request_timeout=30,
        ).choices[0].message.content.strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации контента: {str(e)}")


@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    return generate_content(topic.topic)


@app.get("/")
async def root():
    return {"message": "Service is running"}


@app.get("/heartbeat")
async def heartbeat_api():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
