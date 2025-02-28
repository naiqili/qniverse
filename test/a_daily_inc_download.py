# %%
import pymysql
import csv
import pandas as pd

output_file = 'D:/Lab/finance/wk_data/a_share_market.csv'
df = pd.read_csv(output_file)
local_date = df['trade_dt'].iloc[-1]
print(f'Local latest date: {local_date}')

# %% 
connection = pymysql.connect(host='119.147.213.121', port=9030, user='quant_data_reader', password='0C3D52FA2E', database='low_freq_db')

cursor = connection.cursor()
query = f"SELECT max(trade_dt) as trade_dt FROM a_share_market"
cursor.execute(query)
row = cursor.fetchall()
remote_date = row[0][0]
print(f'Remote latest date: {remote_date}')

# %%
# Number of rows to fetch per query
batch_size = 1000

# Open the CSV file for writing
with open(output_file, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write headers
    # cursor.execute("SELECT * FROM a_share_market order by trade_dt LIMIT 1")
    # writer.writerow([i[0] for i in cursor.description])

    # Fetch and write data in batches
    offset = 0
    while True:
        query = f"SELECT * FROM a_share_market WHERE trade_dt>'{local_date}' order by trade_dt LIMIT {batch_size} OFFSET {offset}"
        cursor.execute(query)
        
        rows = cursor.fetchall()
        if not rows:
            break

        print(rows[0])
        writer.writerows(rows)
        offset += batch_size

# Clean up
cursor.close()
connection.close()

print(f"Data successfully exported to {output_file}")
