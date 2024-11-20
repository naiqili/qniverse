# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.filter import NameDFilter

import warnings
warnings.filterwarnings('ignore')

# %%
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# %%
from lilab.qlib.backtest.benchmark import *


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str)
args = parser.parse_args()

# TODAY = '2024-10-18'
TODAY = args.today
test_split = ('2024-01-05', TODAY)

EXP_NAME, rid = 'EXP_BENCH', '9f4458c45b0d4bf8b2357cc5a271cd45'

SAVE_CSV = True
TOPK = 10
NDROP = 2
HT = 10

info = {
    'ALGO': ['GBDT'],
    'BENCH_DATASET': ['BENCH_Step'],
    'market': ['csi300'], 
    'benchmark': ["SH000300"], 
    'feat': ["Alpha360"], 
    'label': ['r1'],
    'params': [f'topk {TOPK} ndrop {NDROP} HT {HT}']
}

nameDFilter = NameDFilter(name_rule_re='(SH60[0-9]{4})|(SZ00[0-9]{4})')
filter_pipe=[nameDFilter]
benchmark = eval(info['BENCH_DATASET'][0])(\
                 market=info['market'][0], \
                 benchmark=info['benchmark'][0], \
                 feat=info['feat'][0], \
                 label=info['label'][0], \
                 account=10000000, \
                 test_split=test_split,
                 filter_pipe=filter_pipe)

# %%
with R.start(experiment_name=EXP_NAME, uri='../gbdt/mlrun'):
    recorder = R.get_recorder(recorder_id=rid)
    model = recorder.load_object("trained_model")
dataset = init_instance_by_config(benchmark.dataset_config)

strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "model": model,
        "dataset": dataset,
        "topk": TOPK,
        "n_drop": NDROP,
        "hold_thresh": HT,
    },
}

EXP_NAME = 'realworld_test'

port_analysis_config = {
    "executor": benchmark.executor_config,
    "strategy": strategy_config,
    "backtest": benchmark.backtest_config
}

# backtest and analysis
with R.start(experiment_name=EXP_NAME, uri='../gbdt/mlrun'):
    recorder = R.get_recorder(recorder_id=rid)

    # prediction
    recorder = R.get_recorder(recorder_id=rid)
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()


from qlib.contrib.report import analysis_model, analysis_position

with R.start(experiment_name=EXP_NAME, uri='../gbdt/mlrun'):
    recorder = R.get_recorder(recorder_id=rid)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    recorder.save_objects(artifact_path='portfolio_analysis', **{'pred_label.pkl':pred_label})
    print(recorder)

from lilab.qlib.utils.metrics import get_detailed_report, compute_signal_metrics

label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

rp = get_detailed_report(info, pred_label, report_normal_df, analysis_df)
rp['EXP_NAME'] = f"'{EXP_NAME}'"
rp['rid'] = f"'{rid}'"

import plotly
fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdt_rg_{i}.jpg')

    
fig_list = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdt_rag_{i}.jpg')
    


label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

fig_list = analysis_position.score_ic_graph(pred_label, show_notebook=False)

for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdt_sig_{i}.jpg')


fig_list = analysis_model.model_performance_graph(pred_label, show_notebook=False)
for i, fig in enumerate(fig_list):
    fig: plotly.graph_objs.Figure = fig
    fig.write_image(f'./tmp/gbdt_mpg_{i}.jpg')