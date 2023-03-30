import pandas as pd

with open('jerex_results.txt', 'r') as f:
    data = f.read()

fold_results = data.split('--- Entity Mentions ---')

entity_results = {}
relation_results = {}

for fold_result in fold_results:
    if fold_result == '':
        continue
    _, fold_result = fold_result.split('--- Entities ---')

    entity_result, relation_result = fold_result.split('--- Relations ---')
    _, relation_result = relation_result.split('With entity type')

    line: str
    for line in entity_result.split('\n'):
        line = line.strip()
        if line == '':
            continue
        if line.startswith('type') or line.startswith('micro') or line.startswith('macro'):
            continue
        tag, _, _, f, _ = line.split('\t')
        if tag not in ['Activity Data', 'Actor']:
            continue
        f = float(f) / 100
        if tag not in entity_results:
            entity_results[tag] = []
        entity_results[tag].append(f)

    for line in relation_result.split('\n'):
        line = line.strip()
        if line == '':
            continue
        if line.startswith('type') or line.startswith('micro') or line.startswith('macro'):
            continue
        tag, _, _, f, _ = line.split('\t')
        f = float(f) / 100
        if tag not in relation_results:
            relation_results[tag] = []
        relation_results[tag].append(f)

relation_results = {
    t: sum(v) / len(v) for t, v in relation_results.items()
}

entity_results = {
    t: sum(v) / len(v) for t, v in entity_results.items()
}

print(relation_results)
print(entity_results)

df: pd.DataFrame = pd.read_pickle('experiments.pkl')
print(df)
for tag, value in relation_results.items():
    df = df.append({
        'experiment_name': 'jerex-relations',
        'tag': tag.lower(),
        'f1': value
    }, ignore_index=True)
for tag, value in entity_results.items():
    df = df.append({
        'experiment_name': 'jerex-entities',
        'tag': tag.lower(),
        'f1': value
    }, ignore_index=True)
print(df)
pd.to_pickle(df, 'experiments.jerex.pkl')
