import argparse

import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from src.dataset import MTSDatasetH
from lilab.qlib.backtest.benchmark import *
import plotly
import torch
import warnings
warnings.filterwarnings('ignore')
import os,sys
os.chdir(sys.path[0])

# python train_today.py --config_file configs/config_test.yaml

pretrained_dict = {
'PatchTST': '5bc77afc2c434c32a2a07b5554f8a38b', #'b9abb5c01a4c46ecb090afdf3571eed6',
'PDF': 'e9bfe3cdf0b64966b3368feb2bd6305f',
'SegRNN': 'baea86c973bf4c06b65d038bba18a576',
'TimeBridge': 'b2c946befdd24ba980c6c01da8899a82',
'TimeMixer': 'a97f63c66ceb4596bde4b02d416df8d1',
'TimesNet': 'b850b745abfa4ebdba0b92a411786520',
'WFTNet': '47101b2cac2b4136a50562406b421986'
}


def main(seed, config_file="configs/config_wftnet.yaml"):

    # set random seed
    with open(config_file) as f:
        config = yaml.safe_load(f)

    model_name = config["task"]["model"]["kwargs"]["model_type"]
    EXP_NAME, RID = model_name, pretrained_dict[model_name]

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],

    )

    TODAY = '2024-12-10'
    test_split = ('2024-01-01', TODAY)
    filter_pipe = []
    # config  BENCH_Train_Step  test_split=test_split, today=TODAY, filter_pipe=filter_pipe
    benchmark = eval(config["task"]["qlib_dataset"]["class"])(**config["task"]["qlib_dataset"]["kwargs"], 
                                                              test_split=test_split, filter_pipe=filter_pipe)
    dataset = init_instance_by_config(benchmark.dataset_config)
    config["task"]["dataset"]["kwargs"]["handler"] = dataset.handler
    config["task"]["dataset"]["kwargs"]["segments"] = dataset.segments
    print(dataset.segments)
    # origin data -> MTS DataLoader
    dataset = eval(config["task"]["dataset"]["class"])(**config["task"]["dataset"]["kwargs"])
    # dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    TOPK = 10
    NDROP = 2
    HT = 10

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

    # model.fit(dataset)
    EXP_NAME = config["task"]["model"]['kwargs']['model_type']
    URI = '/home/linq/finance/qniverse/mlrun'
    
    with R.start(experiment_name=EXP_NAME, uri=URI):
        rid = RID
        # model.fit(dataset)
        # R.save_objects(trained_model=model)
        # rid = R.get_recorder().id
        print("*******")
        print(rid)
        print("*******")
    
    port_analysis_config = {
        "executor": benchmark.executor_config,
        "strategy": strategy_config,
        "backtest": benchmark.backtest_config
    }
    

    # backtest and analysis
    with R.start(experiment_name=EXP_NAME, uri=URI):
        recorder = R.get_recorder(recorder_id=rid)
        model = recorder.load_object("trained_model")
        # model = torch.load(f'{URI}/3/{rid}/artifacts/trained_model')
        # prediction
        recorder = R.get_recorder(recorder_id=rid)
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    from qlib.contrib.report import analysis_model, analysis_position

    # load record
    with R.start(experiment_name=EXP_NAME, uri=URI):
        recorder = R.get_recorder(recorder_id=rid)
        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        recorder.save_objects(artifact_path='portfolio_analysis', **{'pred_label.pkl':pred_df})
        print(recorder)

    fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_rg_{i}.jpg',engine='kaleido')

    
    fig_list = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_rag_{i}.jpg',engine='kaleido')
    


    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    # pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    fig_list = analysis_position.score_ic_graph(pred_df, show_notebook=False)

    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_sig_{i}.jpg',engine='kaleido')


    fig_list = analysis_model.model_performance_graph(pred_df, show_notebook=False)
    for i, fig in enumerate(fig_list):
        fig: plotly.graph_objs.Figure = fig
        fig.write_image(f'../tmp/{EXP_NAME}_mpg_{i}.jpg',engine='kaleido')

    rank_df = pred_df.droplevel(0).sort_values(by='score', ascending=False)
    top10: pd.DataFrame = rank_df.head(10)
    bottom10: pd.DataFrame = rank_df.tail(10)

    top10.to_markdown(f'../tmp/{model_name}_top10.md')
    bottom10.to_markdown(f'../tmp/{model_name}_bottom10.md')

    import dataframe_image as dfi

    dfi.export(top10, f'../tmp/{model_name}_top10.png',table_conversion='matplotlib')
    dfi.export(bottom10, f'../tmp/{model_name}_bottom10.png',table_conversion='matplotlib')

    infodf = pd.DataFrame({'label': ['Model update date', 'Prediction generation date'],
                       'date': ['2024-11-16', TODAY]})

    dfi.export(infodf, f'../tmp/{model_name}_info.png',table_conversion='matplotlib')


if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_wftnet.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))
