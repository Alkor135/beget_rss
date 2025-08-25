"""
Торговый скрипт...
"""
import logging  # Выводим лог на консоль и в файл
import datetime
import itertools  # Итератор для уникальных номеров транзакций
from pathlib import Path

from QuikPy import QuikPy

# Параметры
class_code = 'SPBFUT'  # Класс для фьючерсов на СПб бирже
sec_code = 'RIU5'  # Тикер
quantity = 1  # Объем (int, так как в QUIK объем целочисленный)
# Ссылка на файл предсказаний
today = datetime.date.today().isoformat()  # Формат YYYY-MM-DD
file_path = Path(fr'C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\{today}.txt')

# Обработчики подписок
def on_trans_reply(data):
    """Обработчик события ответа на транзакцию пользователя"""
    logger.info(f'OnTransReply: {data}')
    global order_num
    order_num = int(data['data']['order_num'])  # Номер заявки на бирже
    logger.info(f'Номер транзакции: {data["data"]["trans_id"]}, Номер заявки: {order_num}')

# Генерация имени лог-файла с временной меткой запуска
log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'transactions_{log_timestamp}.log'

# Настройка логирования
logger = logging.getLogger('QuikPy.Transactions')
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # Новый файл с временной меткой
        logging.StreamHandler()  # Вывод на консоль
    ]
)

qp = QuikPy()  # Инициализация QuikPy (соединения с Quik)

# Установка времени логов по МСК
logging.Formatter.converter = lambda *args: datetime.datetime.now(tz=qp.tz_msk).timetuple()

logger.info("Скрипт запущен.")

# Подписки на события
qp.on_trans_reply = on_trans_reply  # Ответ на транзакцию пользователя. Если транзакция выполняется из QUIK, то не вызывается
qp.on_order = lambda data: logger.info(f'OnOrder: {data}')  # Получение новой / изменение существующей заявки
qp.on_stop_order = lambda data: logger.info(f'OnStopOrder: {data}')  # Получение новой / изменение существующей стоп заявки
qp.on_trade = lambda data: logger.info(f'OnTrade: {data}')  # Получение новой / изменение существующей сделки
qp.on_futures_client_holding = lambda data: logger.info(f'OnFuturesClientHolding: {data}')  # Изменение позиции по срочному рынку
qp.on_depo_limit = lambda data: logger.info(f'OnDepoLimit: {data}')  # Изменение позиции по инструментам
qp.on_depo_limit_delete = lambda data: logger.info(f'OnDepoLimitDelete: {data}')  # Удаление позиции по инструментам

# Ищем первый счет с режимом торгов тикера
account = next((account for account in qp.accounts if class_code in account['class_codes']), None)
if not account:  # Если счет не найден
    logger.error(f'Торговый счет для режима торгов {class_code} не найден')
    exit()

client_code = account['client_code']  # Код клиента
logger.info(f"Код клиента: {client_code=}")

trade_account_id = account['trade_account_id']  # Счет
logger.info(f"Счет: {trade_account_id=}")

firm_id = account['firm_id']  # ID фирмы
logger.info(f"ID фирмы: {firm_id=}")

last_price = float(qp.get_param_ex(class_code, sec_code, 'LAST')['data']['param_value'])
logger.info(f"Последняя цена {sec_code}: {last_price}")

money_limits = qp.get_money_limits()['data'][0]  # Все денежные лимиты (остатки на счетах)
logger.info(f"О счете {money_limits=}")
logger.info(f"Остатки на счете {money_limits['client_code']=} {money_limits['firmid']=}: {money_limits['currentbal']=}")
# logger.info(f"Остатки на счете {money_limits['client_code']=} {money_limits['firmid']=}: {money_limits['currentbal']=}")

si = qp.get_symbol_info(class_code, sec_code)  # Спецификация тикера
logger.info(f"Спецификация тикера {sec_code}: {si}")

# 19-и значный номер заявки на бирже / номер стоп заявки на сервере
order_num = 0
# Номер транзакции задается пользователем
trans_id = itertools.count(1)

positions = qp.get_futures_holdings(0)['data']  # Все фьючерсные позиции (список словарей)
# Выбор позиции по тикеру фьючерса
selected_position = next((pos for pos in positions if pos['sec_code'] == sec_code), None)
# Объем и направление текущей позиции (totalnet >0 - long, <0 - short, 0 - нет позиций)
current_pos = int(selected_position['totalnet']) if selected_position and 'totalnet' in selected_position else 0
if selected_position:
    logger.info(f"Найдена позиция для {sec_code}: {current_pos=}")
else:
    logger.warning(f"Позиция для {sec_code} не найдена")

# Чтение файла и определение направления
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
except Exception as e:
    logger.warning(f"Ошибка чтения файла: {e}.")


direction = None
if "Предсказанное направление: down" in content:
    direction = 'down'
    logger.info(f"Предсказанное направление: down: {direction=}")
elif "Предсказанное направление: up" in content:
    direction = 'up'
    logger.info(f"Предсказанное направление: up: {direction=}")
else:
    logger.warning(f"В файле нет предсказанного направления.")

# Логика торговли ---------------------------------------------------------------------------------
trans_reply = None
if direction == 'down':
    if current_pos == 0:
        # Выставить рыночный ордер на SELL
        # Цена исполнения по рынку. Для фьючерсных заявок цена больше последней при покупке и меньше последней при продаже. Для остальных заявок цена = 0
        # market_price = last_price * 0.99
        market_price = round(last_price * 0.99 / 10) * 10
        market_price = round(market_price, 1)
        logger.info(
            f'Заявка {class_code}.{sec_code} на продажу лотом {quantity} '
            f'по рыночной цене {market_price} при {last_price=}'
        )
        transaction = {  # Все значения должны передаваться в виде строк
            'TRANS_ID': str(next(trans_id)),  # Следующий номер транзакции
            'CLIENT_CODE': client_code,  # Код клиента
            'ACCOUNT': trade_account_id,  # Счет
            'ACTION': 'NEW_ORDER',  # Тип заявки: Новая лимитная/рыночная заявка
            'CLASSCODE': class_code,  # Код режима торгов
            'SECCODE': sec_code,  # Код тикера
            'OPERATION': 'S',  # B = покупка, S = продажа
            'PRICE': str(market_price),  # Цена исполнения по рынку
            'QUANTITY': str(quantity),  # Кол-во в лотах
            'TYPE': 'M'}  # L = лимитная заявка (по умолчанию), M = рыночная заявка
        logger.info(f'Заявка отправлена на рынок: {qp.send_transaction(transaction)["data"]}')
    elif current_pos > 0:
        # Перевернуть позицию: SELL объемом (current_pos + quantity) по рыночной цене
        reverse_qty = current_pos + quantity
        # Цена исполнения по рынку. Для фьючерсных заявок цена больше последней при покупке и меньше последней при продаже. Для остальных заявок цена = 0
        # market_price = qp.price_to_quik_price(
        #     class_code, sec_code, qp.quik_price_to_price(class_code, sec_code, last_price * 0.99)
        # ) if account['futures'] else 0
        market_price = round(last_price * 0.99 / 10) * 10
        market_price = round(market_price, 1)
        logger.info(
            f'Заявка {class_code}.{sec_code} на продажу лотом {reverse_qty} '
            f'по рыночной цене {market_price} при {last_price=}'
        )
        transaction = {  # Все значения должны передаваться в виде строк
            'TRANS_ID': str(next(trans_id)),  # Следующий номер транзакции
            'CLIENT_CODE': client_code,  # Код клиента
            'ACCOUNT': trade_account_id,  # Счет
            'ACTION': 'NEW_ORDER',  # Тип заявки: Новая лимитная/рыночная заявка
            'CLASSCODE': class_code,  # Код режима торгов
            'SECCODE': sec_code,  # Код тикера
            'OPERATION': 'S',  # B = покупка, S = продажа
            'PRICE': str(market_price),  # Цена исполнения по рынку
            'QUANTITY': str(reverse_qty),  # Кол-во в лотах
            'TYPE': 'M'}  # L = лимитная заявка (по умолчанию), M = рыночная заявка
        logger.info(f'Заявка отправлена на рынок: {qp.send_transaction(transaction)["data"]}')
    else:
        logger.info(
            f'Направление открытой позиции совпадает. {current_pos=}, {direction=}. '
            f'Никаких действий не предпринимаем.'
        )
elif direction == 'up':
    if current_pos == 0:
        # Выставить рыночный ордер на BUY
        # Цена исполнения по рынку. Для фьючерсных заявок цена больше последней при покупке и меньше последней при продаже. Для остальных заявок цена = 0
        # market_price = qp.price_to_quik_price(
        #     class_code, sec_code, qp.quik_price_to_price(class_code, sec_code, last_price * 1.01)
        # ) if account['futures'] else 0
        market_price = round(last_price * 1.01 / 10) * 10
        market_price = round(market_price, 1)
        logger.info(
            f'Заявка {class_code}.{sec_code} на покупку лотом {quantity} '
            f'по рыночной цене {market_price} при {last_price=}'
        )
        transaction = {  # Все значения должны передаваться в виде строк
            'TRANS_ID': str(next(trans_id)),  # Следующий номер транзакции
            'CLIENT_CODE': client_code,  # Код клиента
            'ACCOUNT': trade_account_id,  # Счет
            'ACTION': 'NEW_ORDER',  # Тип заявки: Новая лимитная/рыночная заявка
            'CLASSCODE': class_code,  # Код режима торгов
            'SECCODE': sec_code,  # Код тикера
            'OPERATION': 'B',  # B = покупка, S = продажа
            'PRICE': str(market_price),  # Цена исполнения по рынку
            'QUANTITY': str(quantity),  # Кол-во в лотах
            'TYPE': 'M'}  # L = лимитная заявка (по умолчанию), M = рыночная заявка
        logger.info(f'Заявка отправлена на рынок: {qp.send_transaction(transaction)["data"]}')

    elif current_pos < 0:
        # Перевернуть позицию: BUY объемом (|current_pos| + quantity) по рыночной цене
        reverse_qty = abs(current_pos) + quantity
        # Цена исполнения по рынку. Для фьючерсных заявок цена больше последней при покупке и меньше последней при продаже. Для остальных заявок цена = 0
        # market_price = qp.price_to_quik_price(
        #     class_code, sec_code, qp.quik_price_to_price(class_code, sec_code, last_price * 1.01)
        # ) if account['futures'] else 0
        market_price = round(last_price * 1.01 / 10) * 10
        market_price = round(market_price, 1)
        logger.info(
            f'Заявка {class_code}.{sec_code} на покупку лотом {reverse_qty} '
            f'по рыночной цене {market_price} при {last_price=}'
        )
        transaction = {  # Все значения должны передаваться в виде строк
            'TRANS_ID': str(next(trans_id)),  # Следующий номер транзакции
            'CLIENT_CODE': client_code,  # Код клиента
            'ACCOUNT': trade_account_id,  # Счет
            'ACTION': 'NEW_ORDER',  # Тип заявки: Новая лимитная/рыночная заявка
            'CLASSCODE': class_code,  # Код режима торгов
            'SECCODE': sec_code,  # Код тикера
            'OPERATION': 'B',  # B = покупка, S = продажа
            'PRICE': str(market_price),  # Цена исполнения по рынку
            'QUANTITY': str(reverse_qty),  # Кол-во в лотах
            'TYPE': 'M'}  # L = лимитная заявка (по умолчанию), M = рыночная заявка
        logger.info(f'Заявка отправлена на рынок: {qp.send_transaction(transaction)["data"]}')
    else:
        logger.info(
            f'Направление открытой позиции совпадает. {current_pos=}, {direction=}. '
            f'Никаких действий не предпринимаем.'
        )

# Закрываем соединение для запросов и поток обработки функций обратного вызова
qp.close_connection_and_thread()