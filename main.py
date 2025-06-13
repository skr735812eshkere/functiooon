from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from search_module import search_slides

app = FastAPI(
    title="API для поиска по слайдам",
    description="API для семантического поиска информации в PDF-презентациях по численным методам.",
    version="1.0.0"
)

@app.get("/search", summary="Поиск релевантных слайдов по текстовому запросу")
def search_endpoint(query: str = Query(..., description="Текстовый запрос (например, 'Метод Ньютона')")):
    """
    Принимает текстовый запрос, преобразует его в вектор и ищет наиболее
    похожие тексты слайдов в базе данных с помощью FAISS.
    """
    results = search_slides(query)
    # Использование JSONResponse явно задает корректный Content-Type
    # и решает проблемы с отображением кириллицы в некоторых клиентах.
    return JSONResponse(content={"results": results})
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
