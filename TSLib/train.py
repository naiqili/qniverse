import argparse

import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from src.dataset import MTSDatasetH
from lilab.qlib.backtest.benchmark import *
import warnings
warnings.filterwarnings('ignore')
import os,sys
os.chdir(sys.path[0])

# python train.py --config_file configs/config_wftnet.yaml

def main(seed, config_file="configs/config_wftnet.yaml"):

    # set random seed
    with open(config_file) as f:
        config = yaml.safe_load(f)

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

    benchmark = eval(config["task"]["qlib_dataset"]["class"])(**config["task"]["qlib_dataset"]["kwargs"])
    dataset = init_instance_by_config(benchmark.dataset_config)
    config["task"]["dataset"]["kwargs"]["handler"] = dataset.handler
    config["task"]["dataset"]["kwargs"]["segments"] = dataset.segments
    print(dataset.segments)
    # origin data -> MTS DataLoader
    dataset = eval(config["task"]["dataset"]["class"])(**config["task"]["dataset"]["kwargs"])
    # dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    # model.fit(dataset)

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
    EXP_NAME = 'PatchTST'
    URI = '/home/linq/finance/qniverse/mlrun1'

    with R.start(experiment_name=EXP_NAME, uri=URI):
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        print(rid)

    port_analysis_config = {
        "executor": benchmark.executor_config,
        "strategy": strategy_config,
        "backtest": benchmark.backtest_config
    }

    # backtest and analysis
    with R.start(experiment_name=EXP_NAME, uri=URI):
        recorder = R.get_recorder(recorder_id=rid)
        model = recorder.load_object("trained_model")

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

    # analysis
    fig = analysis_position.report_graph(report_normal_df, show_notebook=False)[0]
    fig.write_image("plot_image.jpg", format='jpeg')


if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_wftnet.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))
