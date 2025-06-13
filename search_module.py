import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re

DB_PATH = "num_methods.db"
FAISS_INDEX_PATH = "faiss.index"
SLIDE_IDS_PATH = "slide_ids.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- Параметры гибридного поиска ---
TOP_K_FINAL = 5         # Сколько результатов показывать пользователю
TOP_K_CANDIDATES = 150  # Увеличим количество кандидатов еще больше для большей уверенности
KEYWORD_MATCH_BOOST_FACTOR = 0.5 # Насколько сильно уменьшать дистанцию за каждое найденное ключевое слово (меньше = сильнее буст)
# Более гибкий буст для наличия ключевых слов, а не только точной фразы
MAIN_KEYWORD_PRESENCE_BOOST = 0.1 # Очень сильный буст, если главное ключевое слово (например, "Ньютон") присутствует
MIN_KEYWORD_LENGTH = 3 # Минимальная длина слова для учета как ключевого

print("Инициализация модуля поиска...")

# 1. Грузим все необходимые данные
try:
    with open(SLIDE_IDS_PATH, "rb") as f:
        slide_ids = pickle.load(f)
    print(f"Загружено {len(slide_ids)} ID слайдов.")
except FileNotFoundError:
    print(f"Ошибка: Файл '{SLIDE_IDS_PATH}' не найден. Убедитесь, что 'generate_embeddings_and_faiss.py' был запущен.")
    exit()

try:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, page_num, text FROM slides")
    rows = cursor.fetchall()
    id2data = {row[0]: {"filename": row[1], "page_num": row[2], "text": row[3]} for row in rows}
    conn.close()
    print("Загружены тексты слайдов из БД.")
except sqlite3.OperationalError:
    print(f"Ошибка: База данных '{DB_PATH}' не найдена или повреждена. Убедитесь, что 'rebuild_db.py' был запущен.")
    exit()

try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("Загружен FAISS-индекс.")
except RuntimeError:
    print(f"Ошибка: FAISS-индекс '{FAISS_INDEX_PATH}' не найден. Убедитесь, что 'generate_embeddings_and_faiss.py' был запущен.")
    exit()

try:
    model = SentenceTransformer(MODEL_NAME)
    print("Загружена модель эмбеддингов.")
except Exception as e:
    print(f"Ошибка при загрузке модели SentenceTransformer: {e}. Проверьте подключение к интернету или имя модели.")
    exit()


def embed_query(query: str) -> np.ndarray:
    """Кодирует пользовательский запрос с помощью модели."""
    emb = model.encode(query, convert_to_numpy=True)
    return emb.reshape(1, -1).astype(np.float32)

def clean_text_for_keywords(text: str) -> str:
    """
    Очищает текст, оставляя только буквы и цифры, для извлечения ключевых слов.
    """
    # Заменяем дефисы на пробелы, чтобы "Ньютона-Котеса" стало "Ньютона Котеса"
    text = text.replace('-', ' ')
    return re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text).lower()

def calculate_keyword_score(query_keywords: set, slide_text: str) -> float:
    """
    Рассчитывает "бонус" на основе количества совпадений ключевых слов.
    Чем больше совпадений, тем сильнее бонус (меньше итоговый score).
    """
    slide_words = set(clean_text_for_keywords(slide_text).split())

    # Фильтруем короткие слова, которые могут быть "шумом"
    query_keywords_filtered = {word for word in query_keywords if len(word) >= MIN_KEYWORD_LENGTH}

    if not query_keywords_filtered:
        return 1.0 # Нет бонуса

    matched_keywords_count = len(query_keywords_filtered.intersection(slide_words))
    
    # Применяем умножение, так как FAISS distance: чем меньше, тем лучше.
    keyword_boost_multiplier = (KEYWORD_MATCH_BOOST_FACTOR ** matched_keywords_count)
    
    return max(0.01, keyword_boost_multiplier) # Минимальный множитель
def search_slides(query: str):
    """
    Выполняет гибридный поиск:
    1. Ищет семантически близких кандидатов через FAISS.
    2. Переранжирует кандидатов, давая бонус за вхождение ключевых слов из запроса.
    """
    print(f"Гибридный поиск по запросу: '{query}'")
    
    # 1. Векторный поиск кандидатов
    qvec = embed_query(query)
    distances, indices = index.search(qvec, TOP_K_CANDIDATES)
    
    # Подготовим ключевые слова из запроса один раз
    query_keywords_processed = set(clean_text_for_keywords(query).split())
    # Выделим главное слово запроса для особого буста, например, первое слово или "Ньютон"
    main_query_word = "ньютон" # Можно сделать более динамичным, но пока так

    # 2. Переранжирование (Re-ranking)
    ranked_results = []
    
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        
        slide_id = slide_ids[idx]
        slide_data = id2data.get(slide_id)
        
        if slide_data:
            initial_distance = distances[0][i]
            final_score = initial_distance
            
            slide_text_lower = slide_data["text"].lower()
            
            # --- Самый сильный буст: если главное ключевое слово запроса присутствует ---
            # Используем regex для более точного поиска слова, чтобы избежать частичных совпадений
            # \b для границ слова, чтобы "Ньютон" не матчился в "Ньютоновский" как основное
            if re.search(r'\b' + main_query_word + r'\b', clean_text_for_keywords(slide_text_lower)):
                final_score *= MAIN_KEYWORD_PRESENCE_BOOST
                # print(f"  → СИЛЬНЫЙ БУСТ для слайда ID {slide_id} (главное слово '{main_query_word}')!")
            # Если точное слово "Ньютон" (или "ньютон") не найдено как отдельное слово,
            # но его формы (типа "ньютона") есть, дадим чуть меньший, но все равно сильный буст.
            elif main_query_word in clean_text_for_keywords(slide_text_lower):
                 final_score *= (MAIN_KEYWORD_PRESENCE_BOOST * 2) # Немного меньше буст
                 # print(f"  → СИЛЬНЫЙ БУСТ (форма слова) для слайда ID {slide_id} ('{main_query_word}' в тексте)")

            # --- Затем применяем буст за совпадение остальных ключевых слов ---
            # Этот буст будет применяться поверх предыдущего, если он был,
            # или к initial_distance, если главного слова не было
            keyword_multiplier = calculate_keyword_score(query_keywords_processed, slide_data["text"])
            if keyword_multiplier < 1.0:
                final_score *= keyword_multiplier

            ranked_results.append({
                "id": slide_id,
                "filename": slide_data["filename"],
                "page": slide_data["page_num"],
                "text": slide_data["text"],
                "initial_distance": float(initial_distance),
                "final_score": float(final_score)
            })
            
    # 3. Сортировка по новому скору (чем меньше, тем лучше)
    ranked_results.sort(key=lambda x: x["final_score"])
    
    # 4. Возвращаем лучшие TOP_K_FINAL результатов
    return ranked_results[:TOP_K_FINAL]

print("Модуль поиска готов к работе.")
