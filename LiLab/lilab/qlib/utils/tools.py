import qlib
import pandas as pd
from pandas import Timestamp
import json
import os
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.filter import NameDFilter
from qlib.backtest.position import Position
from qlib.data import D


def load_position_file(path='realworld_result/update_position.txt'):
    if not os.path.exists(path):
        return {}
    dt = cash = None
    pos_dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line)==0:
                continue
            k, v = line.split()
            if k=='cash':
                cash = float(v)
            elif k=='date':
                dt = v + ' 09:00:00'
            else:
                pos_dict[k] = int(v)
    if dt is None or cash is None:
        raise RuntimeError('Datetime or cash not found.')
    dt = pd.to_datetime(dt)
    return {dt: Position(cash=cash, position_dict=pos_dict)}

def load_position_text(text: str):
    dt = cash = None
    pos_dict = {}
    for line in text.split('\n'):
        line = line.strip()
        if len(line)==0:
            continue
        k, v = line.split()
        if k=='cash':
            cash = float(v)
        elif k=='date':
            dt = v + ' 09:00:00'
        else:
            pos_dict[k] = int(v)
    if dt is None or cash is None:
        raise RuntimeError('Datetime or cash not found.')
    dt = pd.to_datetime(dt)
    return {dt: Position(cash=cash, position_dict=pos_dict)}

def save_position_history(pos_history, path='./realworld_result/position_history.json'):
    save_dict = {}
    for dt, account in pos_history.items():
        try:
            save_dict[str(dt)] = eval(str(account))
        except:
            print('save_position_history error')
    with open(path, 'w') as json_file:
        json.dump(save_dict, json_file, indent=4) 

def load_position_history(path='./realworld_result/position_history.json'):
    res_dict = {}
    with open(path) as json_file:
        pos_history = json.load(json_file) 
    for dt, account in pos_history.items():
        pos_dict = {}
        cash = 0
        for k, v in account['position'].items():
            if k=='cash':
                cash = v
            else:
                pos_dict[k] = v
        res_dict[pd.to_datetime(dt)] = Position(cash=cash, position_dict=pos_dict)
    return res_dict

def fill_price_position_history(pos_history):
    for dt, account in pos_history.items():
        stock_list = account.get_stock_list()
        price_df = D.features(
            stock_list,
            ["$close"],
            dt.date(),
            dt.date()
        ).dropna()
        price_dict = price_df.groupby(["instrument"]).tail(1).reset_index(level=1, drop=True)["$close"].to_dict()
        for stock in stock_list:
            account.position[stock]["price"] = price_dict[stock]
        account.position["now_account_value"] = account.calculate_value()
    return pos_history

def normalize_position_history(pos_history, denormalize=False):
    dt_lst = sorted(list(pos_history.keys()))
    new_history = {}
    for dt, cur_pos in pos_history.items():
        factor_df = D.features(cur_pos.get_stock_list(), ['$factor'], \
                               start_time=dt.date(), end_time=dt.date(), freq='day')
        factor_df = factor_df.fillna(1)
        # print(factor_df)
        cash = cur_pos.get_cash()
        new_amount_dict = {}
        for s, a in cur_pos.get_stock_amount_dict().items():
            if denormalize:
                new_amount_dict[s] = int(a * float(factor_df.loc[s, '$factor']))
            else:
                new_amount_dict[s] = int(a / float(factor_df.loc[s, '$factor']))
        new_history[dt] = Position(cash=cash, position_dict=new_amount_dict)
    return new_history