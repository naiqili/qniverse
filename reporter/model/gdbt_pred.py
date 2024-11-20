import qlib
import pandas as pd
from pandas import Timestamp
import json
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.filter import NameDFilter
from qlib.backtest.position import Position
from qlib.data import D

import copy
import pprint
import warnings
warnings.filterwarnings('ignore')

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

from lilab.qlib.utils.tools import normalize_position_history, load_position_text, load_position_history, save_position_history, fill_price_position_history
from lilab.qlib.backtest.benchmark import BENCH_Step
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str)
args = parser.parse_args()

# TODAY = '2024-10-18'
TODAY = args.today
test_split = (TODAY, TODAY)

EXP_NAME, rid = 'EXP_BENCH', '9f4458c45b0d4bf8b2357cc5a271cd45'

TOPK = 10
NDROP = 2
HT = 10

info = {
    'ALGO': ['GBDT'],
    'market': ['scsi300'], 
    'benchmark': ["SH000300"], 
    'feat': ["Alpha360"], 
    'label': ['r1'],
    'params': [f'topk {TOPK} ndrop {NDROP} HT {HT}']
}

nameDFilter = NameDFilter(name_rule_re='(SH60[0-9]{4})|(SZ00[0-9]{4})')
filter_pipe=[nameDFilter]
benchmark = BENCH_Step(test_split,
                        market=info['market'][0], \
                        benchmark=info['benchmark'][0], \
                        feat=info['feat'][0], \
                        label=info['label'][0], \
                        account=10000000, \
                        filter_pipe=filter_pipe)

dataset = init_instance_by_config(benchmark.dataset_config)
# dataset.prepare('test')

with R.start(experiment_name=EXP_NAME, uri='../gbdt/mlrun'):
    recorder = R.get_recorder(recorder_id=rid)
    model = recorder.load_object("trained_model")
    ba_rid = R.get_recorder().id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()
    
    pred_df = recorder.load_object("pred.pkl")

rank_df = pred_df.droplevel(0).sort_values(by='score', ascending=False)
top10: pd.DataFrame = rank_df.head(10)
bottom10: pd.DataFrame = rank_df.tail(10)

# top10.to_markdown('./tmp/gdbt_top10.md')
# bottom10.to_markdown('./tmp/gdbt_bottom10.md')

import dataframe_image as dfi

dfi.export(top10, './tmp/gdbt_top10.png',table_conversion='matplotlib')
dfi.export(bottom10, './tmp/gdbt_bottom10.png',table_conversion='matplotlib')

infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date'],
                       'date': ['2024-11-16', TODAY]})

dfi.export(infodf, './tmp/gdbt_info.png',table_conversion='matplotlib')