import datetime
import os
import logging
import backtrader as bt  # pip install backtrader
from BackTraderQuik import QKStore  # Импорт из BackTraderQuik (скопируйте файлы из https://github.com/cia76/BackTraderQuik)

# Настройка логирования для обработки ошибок
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('trading_log.txt')]
)
logger = logging.getLogger(__name__)

class PredictionStrategy(bt.Strategy):
    params = (
        ('class_code', 'SPBFUT'),  # Класс для фьючерсов
        ('sec_code', 'RIU5'),      # Тикер
        ('quantity', 1),           # Объем
        ('client_code', '126LXS/1UGN1'),  # Код клиента
        ('account', 'SPBFUT192yc'),       # Счет
        ('file_dir', r'C:\Users\Alkor\gd\predict_ai\mix_investing_ollama\\'),  # Директория файлов
    )

    def __init__(self):
        self.dataclose = self.datas[0].close  # Для доступа к ценам (если нужны исторические данные)
        self.position_size = 0  # Текущий размер позиции
        self.last_processed_date = None  # Чтобы отслеживать обработанные файлы

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            logger.info(f"Ордер принят: {order.info}")
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY исполнен: Цена: {order.executed.price}, Объем: {order.executed.size}")
            elif order.issell():
                logger.info(f"SELL исполнен: Цена: {order.executed.price}, Объем: {order.executed.size}")
            self.position_size = self.position.size  # Обновляем позицию

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.error(f"Ордер отклонен/отменен: {order.status}")

    def notify_trade(self, trade):
        if trade.isclosed:
            logger.info(f"Торговля закрыта: PnL: {trade.pnl}")

    def next(self):
        try:
            # Получение текущей даты для имени файла
            today = datetime.date.today().isoformat()  # YYYY-MM-DD
            if today == self.last_processed_date:
                return  # Уже обработано сегодня, пропуск

            file_path = self.p.file_dir + today + '.txt'
            processed_path = file_path + '_processed'

            if os.path.exists(file_path) and not os.path.exists(processed_path):
                logger.info(f"Обнаружен файл: {file_path}. Обработка...")

                # Чтение файла
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                direction = None
                if "Предсказанное направление: down" in content:
                    direction = 'down'
                elif "Предсказанное направление: up" in content:
                    direction = 'up'
                else:
                    logger.warning("В файле нет предсказанного направления. Переименование в _invalid.")
                    os.rename(file_path, file_path + '_invalid')
                    return

                # Получение текущей позиции
                self.position_size = self.position.size  # В Backtrader позиция доступна напрямую

                # Получение цен (best bid/ask из datas или broker)
                best_bid = self.datas[0].bid[0] if hasattr(self.datas[0], 'bid') else self.dataclose[0]  # Адаптируйте, если есть стакан
                best_ask = self.datas[0].ask[0] if hasattr(self.datas[0], 'ask') else self.dataclose[0]

                # Логика торговли
                if direction == 'down':
                    if self.position_size == 0:
                        # Лимитный SELL по BID
                        self.sell(exectype=bt.Order.Limit, price=best_bid, size=self.p.quantity)
                        logger.info(f"Отправлен SELL ордер по {best_bid}")
                    elif self.position_size > 0:
                        # Переворот: SELL (position + quantity)
                        reverse_qty = self.position_size + self.p.quantity
                        self.sell(exectype=bt.Order.Limit, price=best_bid, size=reverse_qty)
                        logger.info(f"Отправлен ордер на переворот (SELL {reverse_qty}) по {best_bid}")
                elif direction == 'up':
                    if self.position_size == 0:
                        # Лимитный BUY по ASK
                        self.buy(exectype=bt.Order.Limit, price=best_ask, size=self.p.quantity)
                        logger.info(f"Отправлен BUY ордер по {best_ask}")
                    elif self.position_size < 0:
                        # Переворот: BUY (|position| + quantity)
                        reverse_qty = abs(self.position_size) + self.p.quantity
                        self.buy(exectype=bt.Order.Limit, price=best_ask, size=reverse_qty)
                        logger.info(f"Отправлен ордер на переворот (BUY {reverse_qty}) по {best_ask}")

                # Переименование файла после отправки ордера
                os.rename(file_path, processed_path)
                logger.info(f"Файл обработан и переименован в {processed_path}")
                self.last_processed_date = today

        except Exception as e:
            logger.error(f"Ошибка в стратегии: {str(e)}")

# Основной запуск Backtrader с QUIK
if __name__ == '__main__':
    try:
        # Создание store из BackTraderQuik
        store = QKStore()  # По умолчанию localhost:34130, настройте если нужно: QKStore(Host='127.0.0.1', ReqPort=34130, etc.)

        # Создание cerebro
        cerebro = bt.Cerebro(stdstats=False)  # Без стандартных статистик для живой торговли

        # Добавление стратегии
        cerebro.addstrategy(PredictionStrategy)

        # Добавление данных (для фьючерса, live данные)
        data = store.getdata(dataname=f'{PredictionStrategy.params.sec_code}',  # Тикер
                             timeframe=bt.TimeFrame.Minutes,     # Таймфрейм, адаптируйте
                             compression=1,                      # Компрессия
                             fromdate=datetime.datetime.now() - datetime.timedelta(days=1),  # Загрузка недавних данных
                             live=True)                          # Живые данные
        cerebro.adddata(data)

        # Настройка брокера
        broker = store.getbroker(client_code=PredictionStrategy.params.client_code,
                                 account=PredictionStrategy.params.account)
        cerebro.setbroker(broker)

        # Запуск
        logger.info("Запуск стратегии в Backtrader...")
        cerebro.run()
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске: {str(e)}")