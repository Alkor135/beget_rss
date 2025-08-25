import logging  # Выводим лог на консоль и в файл
import datetime
import os
import time
from pathlib import Path

from QuikPy import QuikPy  #

# Параметры
class_code = 'SPBFUT'  # Класс для фьючерсов на СПб бирже
sec_code = 'RIU5'  # Тикер
quantity = 1  # Объем (int, так как в QUIK объем целочисленный)
client_code = '126LXS/1UGN1'  # Код клиента
firm_id = 'MC0003300000'  # Фирма
account = 'SPBFUT192yc'  # Счет

logger = logging.getLogger('trade.transactions')  # Будем вести лог
# Инициализация QuikPy (предполагается, что QuikSharp.lua запущен в QUIK)
qp = QuikPy()  # По умолчанию подключается к localhost:34130, если нужно - укажите host и port: QuikPy(host='127.0.0.1', port=34130)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    # Формат сообщения
                    datefmt='%d.%m.%Y %H:%M:%S',  # Формат даты
                    level=logging.DEBUG,
                    # Уровень логируемых событий NOTSET/DEBUG/INFO/WARNING/ERROR/CRITICAL
                    handlers=[logging.FileHandler('transactions.log', encoding='utf-8'),
                              logging.StreamHandler()])  # Лог записываем в файл и выводим на консоль
# В логе время указываем по МСК
logging.Formatter.converter = lambda *args: datetime.datetime.now(tz=qp.tz_msk).timetuple()

logger.info("Скрипт запущен. Ожидание файлов с предсказаниями...")

while True:
    # Получение текущей даты для имени файла
    today = datetime.date.today().isoformat()  # Формат YYYY-MM-DD
    file_path = Path(fr'C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\{today}.txt')
    # Имя для переименованного файла
    processed_path = Path(fr'C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\{today}_processed.txt')

    if os.path.exists(file_path) and not os.path.exists(processed_path):  # Проверяем, существует ли файл и не обработан ли уже
        logger.info(f"Обнаружен файл: {file_path}. Обработка...")

        # Шаг 1: Проверка подключения к QUIK
        if qp.is_connected()['data'] != 1:
            print("Нет подключения к QUIK. Пропуск обработки.")
            time.sleep(60)  # Подождать минуту перед следующей проверкой
            continue

        # Шаг 2: Проверка, торгуется ли фьючерс
        param_ex = qp.get_param_ex(class_code, sec_code, 'STATUS')['data']
        trade_status = param_ex['param_value'] if param_ex else '0'
        if trade_status != '3':  # '3' обычно означает активные торги (проверьте в документации QUIK для вашего инструмента)
            print(f"Фьючерс {sec_code} не торгуется (статус: {trade_status}). Пропуск обработки.")
            time.sleep(60)
            continue

        # Шаг 3: Чтение файла и определение направления
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Ошибка чтения файла: {e}. Пропуск.")
            time.sleep(60)
            continue

        direction = None
        if "Предсказанное направление: down" in content:
            direction = 'down'
        elif "Предсказанное направление: up" in content:
            direction = 'up'
        else:
            print("В файле нет предсказанного направления. Переименование в _invalid.")
            os.rename(file_path, file_path + '_invalid')
            time.sleep(60)
            continue

        # Шаг 4: Получение текущей позиции по фьючерсу
        holding = qp.get_futures_holding(firm_id, account, sec_code, 0)['data']  # 0 - открытая позиция
        current_pos = int(holding['totalnet']) if holding and 'totalnet' in holding else 0  # totalnet >0 - long, <0 - short

        # Шаг 5: Получение лучших BID и ASK из стакана
        quotes = qp.GetQuoteLevel2(class_code, sec_code)['data']
        if not quotes or not quotes['bid'] or not quotes['offer']:
            print("Не удалось получить котировки. Пропуск обработки.")
            time.sleep(60)
            continue

        best_bid = float(quotes['bid'][-1]['price'])  # Лучшая цена на покупку (для SELL)
        best_ask = float(quotes['offer'][0]['price'])  # Лучшая цена на продажу (для BUY)

        # Шаг 6: Логика торговли
        trans_reply = None
        if direction == 'down':
            if current_pos == 0:
                # Выставить лимитный ордер на SELL по BID
                trans_reply = qp.SendLimitOrder(class_code, sec_code, 'S', account, quantity, best_bid, client_code)
                print(f"Отправлен SELL ордер: {trans_reply}")
            elif current_pos > 0:
                # Перевернуть позицию: SELL (current_pos + quantity) по BID
                reverse_qty = current_pos + quantity
                trans_reply = qp.SendLimitOrder(class_code, sec_code, 'S', account, reverse_qty, best_bid, client_code)
                print(f"Отправлен ордер на переворот (SELL {reverse_qty}): {trans_reply}")
        elif direction == 'up':
            if current_pos == 0:
                # Выставить лимитный ордер на BUY по ASK
                trans_reply = qp.SendLimitOrder(class_code, sec_code, 'B', account, quantity, best_ask, client_code)
                print(f"Отправлен BUY ордер: {trans_reply}")
            elif current_pos < 0:
                # Перевернуть позицию: BUY (|current_pos| + quantity) по ASK
                reverse_qty = abs(current_pos) + quantity
                trans_reply = qp.SendLimitOrder(class_code, sec_code, 'B', account, reverse_qty, best_ask, client_code)
                print(f"Отправлен ордер на переворот (BUY {reverse_qty}): {trans_reply}")

        # Если транзакция отправлена успешно (проверьте по trans_reply), переименовать файл
        if trans_reply and 'trans_id' in trans_reply:  # Пример проверки успеха (адаптируйте по документации QuikPy)
            os.rename(file_path, processed_path)
            print(f"Файл обработан и переименован в {processed_path}")
        else:
            print("Ошибка отправки ордера. Файл не переименован.")

    # Пауза перед следующей проверкой (например, 60 секунд)
    time.sleep(60)