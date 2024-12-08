import argparse

import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config
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

    # train model
    model.fit(dataset)



if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_wftnet.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))
