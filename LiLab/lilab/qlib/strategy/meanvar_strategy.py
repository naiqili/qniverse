# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union
from abc import ABC

from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
import cvxpy

class MeanVarStrategy(BaseSignalStrategy):
    def __init__(
        self,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        mode='opt_turnover_with_mean_var',
        risk_gamma=0.001,
        expect_r_daily=None,
        expect_r_anual=0.03,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.mode = mode
        self.risk_gamma = risk_gamma
        if expect_r_daily is None:
            self.expect_r_daily = (1 + expect_r_anual)**(1/252) - 1
        else:
            self.expect_r_daily = expect_r_daily

    def mean_var_opt(self, stock_list, pred, cur_weight):
        if self.mode == 'opt_turnover_with_mean_var':
            res = self.opt_turnover_with_mean_var(stock_list, pred, cur_weight)
        return res
    
    def opt_turnover_with_mean_var(self, stock_list, pred, cur_weight):
        x_len = len(stock_list)
        r_hat = pred.loc[stock_list, 'mean'].values
        var = pred.loc[stock_list, 'var'].values
        cov = np.diag(var)
        x = cvxpy.Variable(x_len)  # optimization vector variable
        x0 = cvxpy.Parameter(x_len)
        r = cvxpy.Parameter(x_len)  # placeholder for vector c
        gamma = cvxpy.Parameter(nonneg=True)
        risk = cvxpy.quad_form(x, cov)
#         obj = cvxpy.Minimize(cvxpy.norm(x - x0, 2) + gamma*risk)  #define objective function
        obj = cvxpy.Minimize(cvxpy.sum_squares(x - x0) + gamma*risk)  #define objective function
        x0.value = cur_weight
        prob = cvxpy.Problem(obj, [cvxpy.sum(x) <= 1, cvxpy.sum(x) >= 0.8, x.T @ r >= self.expect_r_daily, x >= 0])
#         prob = cvxpy.Problem(obj, [cvxpy.sum(x) == 1, x.T @ r >= eps])
        r.value = r_hat
        gamma.value = self.risk_gamma
        try:
            prob.solve(solver='ECOS')  # solve the problem
            x_opt = x.value  # the optimal variable
            turnover = np.abs((x - x0).value).sum()
            exp_r = (x.T @ r).value
            exp_var = risk.value
            get_module_logger('MeanVar').info(f'Turnover: {turnover:.4f}, Expect_r: {exp_r:.8f}, Expect_var: {exp_var:.8f}')
        except:
            x_opt = cur_weight
            get_module_logger('MeanVar').info('No solution')
        res = pd.DataFrame({'weight': x_opt}, index=stock_list)
        return res

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        if pred_score is None:
            return TradeDecisionWO([], self)
    
        if self.only_tradable:
            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        total_wealth = current_temp.calculate_value()
        current_stock_list = list(map(str, pred_score.index))
        current_stock_list = filter_stock(current_stock_list)
        if len(current_stock_list) == 0:
            return TradeDecisionWO([], self)

        _cur_weight = np.asarray([current_temp.get_stock_weight(code) 
                                 if code in current_temp.get_stock_list()
                                 else 0
                                 for code in current_stock_list])
        cur_weight = pd.DataFrame({'weight': _cur_weight}, index=current_stock_list)
        # cur_weight /= cur_weight.sum()
        target_weights = self.mean_var_opt(current_stock_list, pred_score, _cur_weight)

        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            cur_amount = current_temp.get_stock_amount(code=code)

            if target_weights.loc[code, 'weight'] is None:
                continue
            target_value = total_wealth * target_weights.loc[code, 'weight']            

            if target_weights.loc[code, 'weight'] < cur_weight.loc[code, 'weight']: # sell 
                deal_price = self.trade_exchange.get_deal_price(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, 
                    direction=OrderDir.SELL
                )
                target_amount = target_value / deal_price
                sell_amount = cur_amount - target_amount
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                if sell_amount <= 0: continue
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # update cash
                    cash += trade_val - trade_cost
            # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
            # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
            # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
            else: # buy
                # check is stock suspended
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
                ):
                    continue
                # buy order
                deal_price = self.trade_exchange.get_deal_price(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, 
                    direction=OrderDir.BUY
                )
                target_amount = target_value / deal_price
                buy_amount = target_amount - cur_amount
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
                if buy_amount <= 0: continue
                buy_order = Order(
                    stock_id=code,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.BUY,  # 1 for buy
                )
                buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

