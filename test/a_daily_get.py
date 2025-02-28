import pymysql
import csv

connection = pymysql.connect(host='119.147.213.121', port=9030, user='quant_data_reader', password='0C3D52FA2E', database='low_freq_db')

cursor = connection.cursor()

query = f"SELECT * FROM a_share_market WHERE trade_dt>'2024-12-01'"
cursor.execute(query)

rows = cursor.fetchall()
print(cursor.description)
print(rows)

# Clean up
cursor.close()
connection.close()
