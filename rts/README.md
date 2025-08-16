# Backtesting для фьючерсов РТС.
Запустить [rts_21_00_db_investing_month_to_md.py](rts_21_00_db_investing_month_to_md.py) с отредактированным количеством файлов md, которые нужно создать.  
Для [backtesting.py](backtesting.py) с новыми данными (markdown файлами) нужно удалить [embeddings_cache.pkl](embeddings_cache.pkl) для того, чтобы заново создать векторные представления.
Это нужно сделать, потому что векторные представления для новых данных не будут совпадать с векторными представлениями, которые были созданы ранее, и это может привести к ошибкам при выполнении кода.

# Предсказание на следующую торговую сессию.
Для предсказания на следующую торговую сессию нужно запустить:
1. [rts_download_minutes_to_db.py](rts_download_minutes_to_db.py)
2. [rts_21_00_convert_minutes_to_days.py](rts_21_00_convert_minutes_to_days.py)
3. [rts_21_00_db_investing_month_to_md.py](rts_21_00_db_investing_month_to_md.py) - выставить 30 файлов `markdown`.
4. [rts_predict_next_session.py](rts_predict_next_session.py) - предсказанием будет считаться направление `Next_bar` ближайшего похожего по векторам `markdown` файла, который был создан в предыдущем шаге.