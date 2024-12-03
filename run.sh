ST='2024-01-01'
ED='2024-12-01'
# python reporter/model/gdbt_pred.py --today '2024-12-02'
# python reporter/model/gdbt_fig.py --btstart '2024-03-01' --btend '2024-12-01'

# python reporter/model/olmar.py --btstart '2024-01-01' --btend '2024-12-01'

# python reporter/model/kelly.py --btstart '2024-01-01' --btend '2024-12-01'

# python reporter/model/up.py --btstart $ST --btend $ED

python reporter/model/ons.py --btstart $ST --btend $ED

git add .
git commit -m $ED
git push