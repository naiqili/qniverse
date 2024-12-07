#!/bin/bash

source ~/.bashrc 
conda activate qlib39
cd /home/linq/finance/qniverse

ST='2024-01-01'

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
    python reporter/model/gdbt_pred.py --today $ED
    python reporter/model/gdbt_fig.py --btstart $ST --btend $ED
    python reporter/model/olmar.py --btstart $ST --btend $ED
    python reporter/model/kelly.py --btstart $ST --btend $ED
    python reporter/model/up.py --btstart $ST --btend $ED
    python reporter/model/ons.py --btstart $ST --btend $ED
    # python reporter/model/anticor.py --btstart $ST --btend $ED
    # python reporter/model/corn.py --btstart $ST --btend $ED
    # python reporter/model/bnn.py --btstart $ST --btend $ED
    # python reporter/model/pamr.py --btstart $ST --btend $ED
    # python reporter/model/wmamr.py --btstart $ST --btend $ED
    python reporter/model/mpt.py --btstart $ST --btend $ED
    python reporter/model/eg.py --btstart $ST --btend $ED
    # python reporter/model/tco.py --btstart $ST --btend $ED

    echo 'push to github...'
    git add .
    git commit -m $ED
    git push
fi