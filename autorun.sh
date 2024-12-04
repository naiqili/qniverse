echo '=================START======================'
echo $(date)

ED=$(python ./reporter/get_date.py)

if [ $ED == "0" ]; then
    echo "No new data. Have a nice day!"
else
    echo "New data detected: $ED"
    echo 'Update QLIB Database'
    python ./reporter/data_update.py
    python ~/finance/qniverse/LiLab/lilab/qlib/scripts/dump_bin.py dump_all --csv_path /data/linq/.qlib/qlib_data/cn_data/cn_1d_norm --qlib_dir /data/linq/.qlib/qlib_data/cn_data --freq day  --exclude_fields date,symbol
    echo 'QLIB Database updated'

    echo 'Running baselines...'
fi