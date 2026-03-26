import json
with open('01_EDA_Completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
total = len(nb['cells'])
print("Total cells:", total)
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    src = ''.join(c['source'])
    line1 = src.split('\n', maxsplit=1)[0][:100]
    print(f"  Cell {i} [{ct}]: {line1}")
