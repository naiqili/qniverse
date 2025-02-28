import pymysql
import csv

connection = pymysql.connect(host='119.147.213.121', port=9030, user='quant_data_reader', password='0C3D52FA2E', database='low_freq_db')

cursor = connection.cursor()

# Number of rows to fetch per query
batch_size = 1000

# Open the CSV file for writing
output_file = './a_share_market.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write headers
    cursor.execute("SELECT * FROM a_share_market order by trade_dt LIMIT 1")
    writer.writerow([i[0] for i in cursor.description])

    # Fetch and write data in batches
    offset = 0
    while True:
        query = f"SELECT * FROM a_share_market order by trade_dt LIMIT {batch_size} OFFSET {offset}"
        cursor.execute(query)
        
        rows = cursor.fetchall()
        if not rows:
            break

        writer.writerows(rows)
        offset += batch_size

# Clean up
cursor.close()
connection.close()

print(f"Data successfully exported to {output_file}")
