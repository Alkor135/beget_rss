"""
Скрипт объединяет рыночные данные и новости по времени с помощью асинхронного слияния (asof join).
Выбираются строки, где ближайший ценовой бар идёт не позже, чем на 6 минут после публикации новости.
При несоответствии условиям ценовые колонки обнуляются, чтобы избежать некорректных ассоциаций.
Данные загружаются из pickle-файлов, приводятся к формату datetime, затем сортируются.
Операция merge_asof обеспечивает корректное сопоставление по временным меткам.
Результат сохраняется в файл combine.pkl с сохранением исходного порядка новостей.
Выводится статистика по количеству совпадений и пропущенных событий.
"""
import pandas as pd

# Пути к файлам (при необходимости укажите полные пути)
prices_path = "minutes_RTS_processed_p8000.pkl"
news_path = "news_embeds.pkl"
out_path = "combine.pkl"

# 1) Загрузка
prices = pd.read_pickle(prices_path)
news = pd.read_pickle(news_path)

# 2) Приведение типов времени
if "TRADEDATE" not in prices.columns:
    raise KeyError("В prices нет колонки 'TRADEDATE'")
prices["TRADEDATE"] = pd.to_datetime(prices["TRADEDATE"], errors="coerce")

if "loaded_at" not in news.columns:
    raise KeyError("В news нет колонки 'loaded_at'")
news["loaded_at"] = pd.to_datetime(news["loaded_at"], errors="coerce")

# 3) Сортировка для merge_asof
prices_sorted = prices.sort_values("TRADEDATE")
news_sorted = news.sort_values("loaded_at")

# 4) forward-asof: первый бар c TRADEDATE >= loaded_at
merged = pd.merge_asof(
    news_sorted,
    prices_sorted,
    left_on="loaded_at",
    right_on="TRADEDATE",
    direction="forward",
    allow_exact_matches=True,
)

# 5) Ограничение 6 минут: оставляем только те матчи, где разница <= 6 минут
max_delta = pd.Timedelta(minutes=6)
delta = merged["TRADEDATE"] - merged["loaded_at"]

# ВАЖНО: используем побитовое '&' и скобки
mask_ok = delta.notna() & (delta <= max_delta)

# Список ценовых/индикаторных колонок, которые нужно обнулить при неуспехе
price_cols = ["TRADEDATE", "OPEN", "CLOSE", "H2", "perc_25", "perc_75"]
for c in price_cols:
    if c in merged.columns:
        merged.loc[~mask_ok, c] = None

# 6) Восстановление исходного индекса (если важен порядок новостей)
merged = merged.sort_index()

# 7) Сохранение результата
merged.to_pickle(out_path)

# Необязательный контроль: можно вывести краткую статистику
print({
    "total_news_rows": len(merged),
    "matched_within_6m": int(mask_ok.sum()),
    "no_match_or_too_late": int((~mask_ok).sum()),
})
