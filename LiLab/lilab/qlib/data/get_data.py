import time
import pandas as pd
from yahooquery import Ticker

class StockDataFetcher:
    def __init__(self, symbols, interval=300, callback=None, proxy_port=None):
        self.symbols = symbols
        self.interval = interval
        self.callback = callback
        if proxy_port is None:
            self.proxies = None
        else:
            self.proxies = {
                'http': f'http://127.0.0.1:{proxy_port}',
                'https': f'http://127.0.0.1:{proxy_port}',
                }
        self.ticker = Ticker(symbols, proxies=self.proxies)
        self.data_frame = pd.DataFrame()

        self.start_fetching(period='1d')

    def get_price_history(self, period='10min'):
        try:
            history_data = self.ticker.history(period=period, interval='1m', start='2024-10-11')
            return history_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def _fetch(self, period='10min'):
        new_data = self.get_price_history(period=period)
        if not new_data.empty:
            self.save_to_dataframe(new_data)
            print(f"Updated data at {pd.Timestamp.now()}")

    def start_fetching(self, period='10min'):
        while True:
            self._fetch(period=period)
            time.sleep(self.interval)

    def save_to_dataframe(self, new_data):
        self.data_frame = pd.concat([self.data_frame, new_data]).sort_index()
        # self.data_frame = pd.merge(self.data_frame, new_data, left_index=True, right_index=True, how='outer')
        self.data_frame = self.data_frame.drop_duplicates()  # 去除重复的数据行
        if self.callback is not None:
            self.callback(self.data_frame)

    def get_data(self):
        """
        获取当前保存的所有股票数据
        :return: Pandas DataFrame 格式的股票数据
        """
        return self.data_frame

# 使用方法
if __name__ == "__main__":
    # 创建 StockDataFetcher 对象，监控 'AAPL', 'MSFT' 的股票，每隔 5 分钟更新一次
    fetcher = StockDataFetcher('000333.SZ 000001.SZ', interval=60,\
                               proxy_port=8890,
                               callback=lambda df:print(df))
    fetcher.start_fetching()
