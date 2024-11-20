import numpy as np
import pandas as pd
from qlib.contrib.report.analysis_position.score_ic import _get_score_ic


def compute_signal_metrics(list1, list2, seasonality=1):
    # Convert lists to numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    arr1 = np.nan_to_num(arr1, nan=0)
    arr2 = np.nan_to_num(arr2, nan=0)

    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(np.mean((arr1 - arr2) ** 2))

    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(arr1 - arr2))

    # MAPE: Mean Absolute Percentage Error
    mape = np.mean(np.abs((arr1 - arr2) / arr2)) * 100  # MAPE in percentage

    # MASE: Mean Absolute Scaled Error
    # Calculate naive forecast (seasonality can be adjusted)
    naive_forecast = arr1[seasonality:]  # shift by seasonality
    naive_error = np.abs(arr1[seasonality:] - arr1[:-seasonality])
    mae_naive = np.mean(naive_error)
    
    mase = mae / mae_naive

    return rmse, mae, mape, mase

def calculate_annualized_return_volatility_strict(report_normal_df, periods_per_year=238, with_cost=False, excess=False):
    returns = report_normal_df['return']
    returns = np.array(returns)
    if with_cost:
        returns -= report_normal_df['cost']
    if excess:
        returns -= report_normal_df['bench']
    
    # Calculate total return (geometric mean)
    total_return = np.prod(1 + returns) - 1
    
    # Number of periods
    n_periods = len(returns)
    
    # Calculate annualized return
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # Calculate annualized volatility (standard deviation)
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    return total_return, annualized_return, volatility

def calculate_annualized_return_volatility(report_normal_df, periods_per_year=238, with_cost=False, excess=False):
    returns = report_normal_df['return']
    returns = np.array(returns)
    if with_cost:
        returns -= report_normal_df['cost']
    if excess:
        returns -= report_normal_df['bench']
    
    # Calculate total return (geometric mean)
    total_return = np.prod(1 + returns) - 1
    
    # Calculate annualized return
    annualized_return = np.mean(returns) * periods_per_year
    
    # Calculate annualized volatility (standard deviation)
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    return total_return, annualized_return, volatility

def get_detailed_report(info, pred_label, report_normal_df, analysis_df):
    rmse, mae, mape, mase = compute_signal_metrics(pred_label.iloc[:, 0], pred_label.iloc[:, 1])
    _ic_df = _get_score_ic(pred_label)
    _ic_df = _ic_df.mean()

    rp = pd.DataFrame(info, index=[pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')])
    rp.index.name = 'date'
    rp['days'] = (report_normal_df.index[-1]-report_normal_df.index[0])
    rp['init wealth'] = report_normal_df.iloc[0]['account']
    rp['final wealth'] = report_normal_df.iloc[-1]['account']
    rp['total wealth'] = rp['final wealth'] / rp['init wealth'] 
    rp['turnover rate'] = report_normal_df['turnover'].mean()
    rp['total turnover'] = report_normal_df.iloc[-1]['total_turnover']
    rp['cost rate'] = report_normal_df['cost'].mean()
    rp['total cost'] = report_normal_df.iloc[-1]['total_cost']
    rp['bench return rate'] = report_normal_df['bench'].mean()

    total_return, annualized_return, annualized_volatility = \
        calculate_annualized_return_volatility(report_normal_df, with_cost=True)
    rp['total return w cost'] = total_return
    rp['annualized return w cost'] = annualized_return
    rp['annualized volatility w cost'] = annualized_volatility
    rp['annualized sharpe ratio w cost'] = annualized_return / annualized_volatility

    total_return, annualized_return, annualized_volatility = \
        calculate_annualized_return_volatility(report_normal_df, with_cost=True, excess=True)
    rp['excess total return w cost'] = total_return
    rp['excess annualized return w cost'] = annualized_return
    rp['excess annualized volatility w cost'] = annualized_volatility
    rp['excess annualized sharpe ratio w cost'] = annualized_return / annualized_volatility

    rp['information ratio w cost'] = analysis_df.loc[('excess_return_with_cost', 'information_ratio'), 'risk']
    rp['max drawdown w cost'] = analysis_df.loc[('excess_return_with_cost', 'max_drawdown'), 'risk']


    total_return, annualized_return, annualized_volatility = \
        calculate_annualized_return_volatility(report_normal_df, with_cost=False)
    rp['total return wo cost'] = total_return
    rp['annualized return wo cost'] = annualized_return
    rp['annualized volatility wo cost'] = annualized_volatility
    rp['annualized sharpe ratio wo cost'] = annualized_return / annualized_volatility

    total_return, annualized_return, annualized_volatility = \
        calculate_annualized_return_volatility(report_normal_df, with_cost=False, excess=True)
    rp['excess total return wo cost'] = total_return
    rp['excess annualized return wo cost'] = annualized_return
    rp['excess annualized volatility wo cost'] = annualized_volatility
    rp['excess annualized sharpe ratio wo cost'] = annualized_return / annualized_volatility

    rp['information ratio wo cost'] = analysis_df.loc[('excess_return_without_cost', 'information_ratio'), 'risk']
    rp['max drawdown wo cost'] = analysis_df.loc[('excess_return_without_cost', 'max_drawdown'), 'risk']

    rp[['IC', 'rank IC']] = _ic_df[['ic', 'rank_ic']]
    rp['RMSE'] = rmse
    rp['MAE'] = mae
    # rp['MAPE'] = mape
    # rp['MASE'] = mase

    return rp
