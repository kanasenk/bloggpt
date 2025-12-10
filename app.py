"""
main.py — FastAPI-сервис генерации блог-постов с использованием:
- OpenAI API (GPT-4o / gpt-4o-mini)
- Currents News API (актуальные новости как контекст)

Функциональность:
- POST /generate_post — сгенерировать пост по теме с учётом свежих новостей
- GET  /health         — простой health-check
- GET  /health/details — подробный статус зависимостей и переменных окружения

Перед запуском:
    pip install fastapi uvicorn[standard] openai httpx python-dotenv

И задайте переменные окружения, например через .env:
    OPENAI_API_KEY=ваш_openai_ключ
    CURRENTS_API_KEY=ваш_currents_ключ
    OPENAI_MODEL=gpt-4o-mini   # необязательно, по умолчанию gpt-4o-mini
"""

import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# ==========================
# Загрузка переменных окружения
# ==========================

# Если у вас есть .env файл — он будет автоматически прочитан
load_dotenv()

# Можно переопределить модель через переменную окружения
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_client() -> OpenAI:
    """
    Вспомогательная функция: создаёт клиент OpenAI,
    проверяя наличие API-ключа в окружении.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # В режиме API-сервиса возвращаем понятную ошибку
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY не задан в переменных окружения.",
        )
    # Клиент сам подтянет api_key из окружения, но можно передать явно
    return OpenAI(api_key=api_key)


def get_currents_api_key() -> str:
    """
    Вспомогательная функция: достаёт ключ Currents API.
    """
    api_key = os.getenv("CURRENTS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="CURRENTS_API_KEY не задан в переменных окружения.",
        )
    return api_key


# ==========================
# Pydantic-модели для запросов/ответов
# ==========================


class GenerateRequest(BaseModel):
    """
    Модель входящего запроса для генерации поста.
    """

    topic: str = Field(..., example="Преимущества медитации")
    language: str = Field(
        "ru",
        description=(
            "Язык итогового поста и новостей. "
            "Currents поддерживает несколько языков, чаще всего 'en' или 'ru'."
        ),
        example="ru",
    )
    max_news: int = Field(
        5,
        ge=1,
        le=20,
        description="Максимальное количество новостных материалов, которые использовать как контекст.",
        example=5,
    )


class NewsArticle(BaseModel):
    """
    Упрощённая модель новости из Currents API,
    которую мы вернём клиенту и передадим модели OpenAI.
    """

    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    published: Optional[str] = None
    category: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    """
    Модель ответа сервиса: структура, удобная для дальнейшей интеграции.
    """

    topic: str
    language: str
    title: str
    meta_description: str
    post_content: str
    used_news: List[NewsArticle]


class HealthDetails(BaseModel):
    """
    Детальный health-check, чтобы быстро понять,
    корректно ли настроены ключи и базовые зависимости.
    """

    status: str
    openai_api_key_present: bool
    currents_api_key_present: bool
    model: str


# ==========================
# Инициализация FastAPI-приложения
# ==========================

app = FastAPI(
    title="Blog Post Generator with OpenAI & Currents API",
    description=(
        "Сервис генерирует SEO-оптимизированные блог-посты на основе темы и свежих новостей "
        "из Currents API, используя модель OpenAI."
    ),
    version="1.0.0",
)


# ==========================
# Вспомогательные функции
# ==========================


async def fetch_news_from_currents(
    topic: str, language: str = "ru", max_results: int = 5
) -> List[NewsArticle]:
    """
    Получить список новостей по теме из Currents API.

    Используем эндпоинт поиска:
        GET https://api.currentsapi.services/v1/search

    Параметры:
        keywords — ключевые слова (topic)
        language — код языка
        apiKey   — ключ авторизации

    Возвращаем первые max_results новостей,
    преобразованные в модель NewsArticle.
    """
    api_key = get_currents_api_key()
    url = "https://api.currentsapi.services/v1/search"

    params = {
        "apiKey": api_key,
        "keywords": topic,
        "language": language,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
    except httpx.RequestError as e:
        # Ошибки сети: таймаут, DNS, и т.п.
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при обращении к Currents API: {repr(e)}",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Currents API вернул статус {response.status_code}: {response.text}",
        )

    data = response.json()

    if data.get("status") != "ok":
        raise HTTPException(
            status_code=502,
            detail=f"Currents API вернул ошибку: {data!r}",
        )

    news_raw = data.get("news", [])[:max_results]
    news_list: List[NewsArticle] = []

    for item in news_raw:
        news_list.append(
            NewsArticle(
                title=item.get("title", ""),
                description=item.get("description"),
                url=item.get("url"),
                author=item.get("author"),
                published=item.get("published"),
                category=item.get("category"),
            )
        )

    return news_list


def build_news_context(news_list: List[NewsArticle], language: str) -> str:
    """
    Преобразуем список новостей в компактный текстовый контекст для промпта.

    Чтобы не перегружать модель, берём только ключевые элементы:
    - заголовок
    - дата
    - краткое описание
    """
    if not news_list:
        # Если новостей нет — возвращаем пустую строку, а не None
        return ""

    # Форматируем контекст: список буллетов.
    lines = []
    for idx, n in enumerate(news_list, start=1):
        # Важно: описание может быть None
        description = n.description or ""
        published = n.published or "дата не указана"

        if language == "ru":
            lines.append(
                f"{idx}. {n.title} ({published}) — {description}"
            )
        else:
            # Простейшая англ. формулировка
            lines.append(
                f"{idx}. {n.title} ({published}) — {description}"
            )

    if language == "ru":
        header = "Краткое резюме актуальных новостей по теме:\n"
    else:
        header = "Short summary of recent news on the topic:\n"

    return header + "\n".join(lines)


def generate_post_with_openai(
    topic: str, language: str, news_context: str
) -> GenerateResponse:
    """
    Генерация:
    - заголовка
    - meta description
    - основного текста поста

    с использованием OpenAI Chat Completions API.
    News_context включаем в промпт, чтобы пост опирался на актуальные новости.
    """
    client = get_openai_client()

    # ------------------
    # 1. Генерация заголовка
    # ------------------
    if language == "ru":
        title_prompt = (
            "Ты — редактор профессионального онлайн-медиа.\n"
            "Придумай один короткий, цепляющий заголовок для блог-поста на русском языке.\n"
            f"Тема: «{topic}».\n"
            "Можешь аккуратно опираться на приведённый новостной контекст, "
            "но не копируй заголовки один в один.\n\n"
            f"{news_context}"
        )
    else:
        title_prompt = (
            "You are an editor of a professional online media outlet.\n"
            "Come up with one short and catchy blog post title in English.\n"
            f"Topic: \"{topic}\".\n"
            "You may use the news context below as inspiration, "
            "but do not copy headlines verbatim.\n\n"
            f"{news_context}"
        )

    try:
        title_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional blog editor."},
                {"role": "user", "content": title_prompt},
            ],
            max_tokens=64,
            temperature=0.7,
        )
        title = title_resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при генерации заголовка через OpenAI: {repr(e)}",
        )

    # ------------------
    # 2. Генерация meta description
    # ------------------
    if language == "ru":
        meta_prompt = (
            "Напиши одно meta-описание (до 160 символов) для SEO на русском языке "
            "к следующему заголовку блог-поста.\n"
            "Описание должно быть конкретным, живым и отражать актуальность темы.\n"
            "Не используй кавычки вокруг описания.\n\n"
            f"Заголовок: {title}\n\n"
            f"{news_context}"
        )
    else:
        meta_prompt = (
            "Write a single meta description (up to 160 characters) in English for SEO "
            "for the following blog post title.\n"
            "The description should be specific, engaging, and highlight the topical relevance.\n"
            "Do not wrap the description in quotes.\n\n"
            f"Title: {title}\n\n"
            f"{news_context}"
        )

    try:
        meta_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an SEO specialist."},
                {"role": "user", "content": meta_prompt},
            ],
            max_tokens=80,
            temperature=0.7,
        )
        meta_description = meta_resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при генерации meta-описания через OpenAI: {repr(e)}",
        )

    # ------------------
    # 3. Генерация основного текста поста
    # ------------------
    if language == "ru":
        post_prompt = (
            "Ты — профессиональный автор блогов и журналист.\n"
            "Напиши развёрнутый, увлекательный пост для блога на русском языке "
            "по указанной теме.\n"
            "Структура:\n"
            "1) Вступление (1–2 абзаца)\n"
            "2) 3–5 подзаголовков с логичным развитием мысли\n"
            "3) Практические советы / выводы\n\n"
            "Требования:\n"
            "- Используй короткие абзацы и подзаголовки (Markdown: `## Подзаголовок`)\n"
            "- Вшивай в текст ключевые слова и идеи, связанные с темой\n"
            "- Аккуратно используй факты из новостей, приведённых ниже, "
            "но не копируй текст новостей дословно\n"
            "- Пиши естественно, без штампованных фраз\n\n"
            f"Тема поста: «{topic}»\n\n"
            f"{news_context}\n\n"
            "Теперь напиши полный текст поста."
        )
    else:
        post_prompt = (
            "You are a professional blog writer and journalist.\n"
            "Write a detailed and engaging blog post in English on the given topic.\n"
            "Structure:\n"
            "1) Introduction (1–2 paragraphs)\n"
            "2) 3–5 subheadings that develop the topic logically\n"
            "3) Practical tips / conclusions\n\n"
            "Requirements:\n"
            "- Use short paragraphs and Markdown subheadings (`## Subheading`)\n"
            "- Naturally incorporate keywords and ideas related to the topic\n"
            "- Carefully use facts from the news context below, "
            "but do not copy news text verbatim\n"
            "- Avoid clichés and generic filler phrases\n\n"
            f"Topic: \"{topic}\"\n\n"
            f"{news_context}\n\n"
            "Now write the full blog post."
        )

    try:
        post_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior content writer."},
                {"role": "user", "content": post_prompt},
            ],
            max_tokens=2048,
            temperature=0.8,
        )
        post_content = post_resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при генерации поста через OpenAI: {repr(e)}",
        )

    # Сборка ответа в формате GenerateResponse
    return title, meta_description, post_content


# ==========================
# Эндпоинты FastAPI
# ==========================


@app.get("/", summary="Короткое описание сервиса")
def root():
    """
    Базовый эндпоинт: можно использовать как простой ping.
    """
    return {
        "service": "Blog Post Generator with OpenAI & Currents API",
        "version": "1.0.0",
        "message": "Сервис работает. Используйте POST /generate_post для генерации постов.",
    }


@app.get("/health", summary="Простой health-check")
def health():
    """
    Простейший health-check: возвращает статус OK,
    не проверяя внешние зависимости.
    """
    return {"status": "ok"}


@app.get(
    "/health/details",
    response_model=HealthDetails,
    summary="Расширенный health-check",
)
def health_details():
    """
    Расширенный health-check: проверяет наличие переменных окружения
    и возвращает модель, которая используется.
    """
    openai_present = bool(os.getenv("OPENAI_API_KEY"))
    currents_present = bool(os.getenv("CURRENTS_API_KEY"))

    status = "ok" if openai_present and currents_present else "degraded"

    return HealthDetails(
        status=status,
        openai_api_key_present=openai_present,
        currents_api_key_present=currents_present,
        model=OPENAI_MODEL,
    )


@app.post(
    "/generate_post",
    response_model=GenerateResponse,
    summary="Сгенерировать блог-пост по теме",
)
async def generate_post_endpoint(payload: GenerateRequest):
    """
    Основной эндпоинт:
    1. Получает новости по теме из Currents API
    2. Собирает текстовый контекст из новостей
    3. Генерирует заголовок, meta-описание и полный пост через OpenAI
    """
    # 1. Получаем новости по теме
    news_list = await fetch_news_from_currents(
        topic=payload.topic,
        language=payload.language,
        max_results=payload.max_news,
    )

    # 2. Формируем контекст для промпта
    news_context = build_news_context(news_list, language=payload.language)

    # 3. Вызываем OpenAI для генерации текста
    title, meta_description, post_content = generate_post_with_openai(
        topic=payload.topic,
        language=payload.language,
        news_context=news_context,
    )

    # 4. Возвращаем результат
    return GenerateResponse(
        topic=payload.topic,
        language=payload.language,
        title=title,
        meta_description=meta_description,
        post_content=post_content,
        used_news=news_list,
    )


# ==========================
# Точка входа для запуска через `python main.py`
# ==========================

if __name__ == "__main__":
    import uvicorn

    # Запуск локального сервера:
    #   python main.py
    #
    # Затем можете обратиться:
    #   - http://localhost:8000/docs        (Swagger UI)
    #   - POST /generate_post с JSON телом
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # удобно в разработке, можно выключить в проде
    )
    # Запуск приложения с указанием порта
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
