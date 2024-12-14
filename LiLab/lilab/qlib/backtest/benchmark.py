from qlib.data import D
import pandas as pd

class BENCHBase:
    def __init__(self, 
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 market='csi300', \
                 benchmark="SH000300", \
                 feat="Alpha158", \
                 label='r1',
                 account=1000000,
                 filter_pipe=None,
                 ):
        self.time_span = time_span
        self.fit_time_split = fit_time_split
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.backtest_split = backtest_split
        self.market = market
        self.benchmark = benchmark
        self.feat = feat
        self.label = label
        self.account = account
        self.filter_pipe = filter_pipe
        
        self.instruments = D.instruments(market=market, filter_pipe=filter_pipe)

        if label == 'r0':
            self.label_code = (["Ref($close, -1)/$close - 1"], ["LABEL0"])
        elif label == 'r1':
            self.label_code = (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
        else:
            raise NotImplementedError()
        
        self.data_handler_config = {
            "start_time": time_span[0],
            "end_time": time_span[1],
            "fit_start_time": fit_time_split[0],
            "fit_end_time": fit_time_split[1],
            "instruments": self.instruments,
            "label": self.label_code,            
            # 'infer_processors': [
            #     {
            #     'class': 'CSZFillna',
            #     }
            # ],
            # 'learn_processors':  [
            #     {
            #     'class': 'CSZFillna',
            #     }
            # ],
        }

        if feat == "Alpha158":
            self.feat_config = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": self.data_handler_config,
            }
        elif feat == "Alpha360":
            self.feat_config = {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": self.data_handler_config,
            }
        else:
            raise NotImplementedError()
        
        self.dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": self.feat_config,
                "segments": {
                    "train": train_split,
                    "valid": valid_split,
                    "test": test_split
                },
            },
        }
        self.executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        }
        self.backtest_config = {
            "start_time": backtest_split[0],
            "end_time": backtest_split[1],
            "account": self.account,
            "benchmark": self.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }

class BENCH_A(BENCHBase):
    def __init__(self, 
                 time_span=("2010-01-01","2024-09-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2010-01-01", "2018-12-31"),
                 valid_split=("2019-01-01", "2020-12-31"),
                 test_split=("2021-01-01", "2024-09-01"),
                 backtest_split=("2021-01-01", "2024-09-01"),
                 **kwargs
                 ):
        super(BENCH_A, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )
        
class BENCH_B(BENCHBase):
    def __init__(self, 
                 time_span=("2005-01-01","2021-01-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2005-01-01", "2013-12-31"),
                 valid_split=("2014-01-01", "2015-12-31"),
                 test_split=("2016-01-01", "2021-01-01"),
                 backtest_split=("2016-01-01", "2021-01-01"),
                 **kwargs
                 ):
        super(BENCH_B, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )
        
class BENCH_C(BENCHBase):
    def __init__(self, 
                 time_span=("2005-01-01","2016-01-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2005-01-01", "2008-12-31"),
                 valid_split=("2009-01-01", "2010-12-31"),
                 test_split=("2011-01-01", "2016-01-01"),
                 backtest_split=("2011-01-01", "2016-01-01"),
                 **kwargs
                 ):
        super(BENCH_C, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )
        
class BENCH_LPY(BENCHBase):
    def __init__(self, 
                 time_span=("2017-01-01","2024-01-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2017-01-01", "2021-12-31"),
                 valid_split=("2022-01-01", "2022-12-31"),
                 test_split=("2023-01-01", "2024-01-01"),
                 backtest_split=("2023-01-01", "2024-01-01"),
                 **kwargs
                 ):
        super(BENCH_LPY, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )
        
class BENCH_NOW(BENCHBase):
    def __init__(self, 
                 time_span=("2005-01-01","2024-12-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2005-01-01","2020-01-01"),
                 valid_split=("2020-01-01","2024-11-01"),
                 test_split=("2024-11-01", "2024-12-01"),
                 backtest_split=("2024-11-01", "2024-12-01"),
                 **kwargs
                 ):
        super(BENCH_NOW, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )

class BENCH_Set1(BENCHBase):
    def __init__(self, 
                 time_span=("2010-01-01","2024-12-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2010-01-01","2020-12-31"),
                 valid_split=("2021-01-01","2023-12-31"),
                 test_split=("2024-01-01","2024-12-01"),
                 backtest_split=("2024-01-01","2024-12-01"),
                 **kwargs
                 ):
        super(BENCH_Set1, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )

class BENCH_Set2(BENCHBase):
    def __init__(self, 
                 time_span=("2015-01-01","2020-01-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2015-01-01","2020-01-01"),
                 valid_split=("2015-01-01","2020-01-01"),
                 test_split=("2015-01-01","2020-01-01"),
                 backtest_split=("2020-01-01","2021-01-01"),
                 **kwargs
                 ):
        super(BENCH_Set2, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )

class BENCH_Set3(BENCHBase):
    def __init__(self, 
                 time_span=("2010-01-01","2015-01-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2010-01-01","2015-01-01"),
                 valid_split=("2010-01-01","2015-01-01"),
                 test_split=("2010-01-01","2015-01-01"),
                 backtest_split=("2015-01-01","2016-01-01"),
                 **kwargs
                 ):
        super(BENCH_Set3, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )

class BENCH_Set4(BENCHBase):
    def __init__(self, 
                 time_span=("2005-01-01","2010-01-01"),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2005-01-01","2010-01-01"),
                 valid_split=("2005-01-01","2010-01-01"),
                 test_split=("2005-01-01","2010-01-01"),
                 backtest_split=("2010-01-01","2011-01-01"),
                 **kwargs
                 ):
        super(BENCH_Set3, self).__init__(
                 time_span,
                 fit_time_split,
                 train_split,
                 valid_split,
                 test_split,
                 backtest_split,
                 **kwargs
        )

class BENCH_Step(BENCHBase):
    def __init__(self, 
                 test_split,
                 **kwargs
                 ):
        super(BENCH_Step, self).__init__(
                 time_span=test_split,
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=test_split,
                 valid_split=test_split,
                 test_split=test_split,
                 backtest_split=test_split,
                 **kwargs
        )


class BENCH_Train_Step(BENCHBase):
    def __init__(self, 
                 test_split,
                 today,
                 **kwargs
                 ):
        super(BENCH_Train_Step, self).__init__(
                 time_span=("2022-01-01",today),
                 fit_time_split=("2022-01-01","2024-01-01"),
                 train_split=("2018-01-01", "2022-12-31"),
                 valid_split=("2023-01-01", "2023-12-31"),
                 test_split=test_split,
                 backtest_split=test_split,
                 **kwargs
        )