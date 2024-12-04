import wk_db
import pandas as pd

wk_date = wk_db.read_sql('SELECT MAX(trade_dt) FROM a_share_market;', db_loc='low_freq_db')
wk_date = str(wk_date.iloc[0,0])
# print('WK Dateset Last Date: ', wk_date)

cal = pd.read_csv('/data/linq/.qlib/qlib_data/cn_data/calendars/day.txt')
qlib_date = str(cal.iloc[-1,0])
# print('QLIB Dateset Last Date: ', qlib_date)

if qlib_date < wk_date:
    # print('New data detected')
    print(wk_date)
else:
    print('0')
