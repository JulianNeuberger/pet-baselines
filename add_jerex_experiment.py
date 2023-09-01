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
        if line.startswith('type') or line.startswith('macro'):
            continue

        if line.startswith('micro'):
            tag = 'overall'
            _, p, r, f1, _ = line.split('\t')
        else:
            tag, p, r, f1, _ = line.split('\t')

        f1 = float(f1) / 100
        p = float(p) / 100
        r = float(r) / 100
        if tag not in relation_results:
            relation_results[tag] = []
        relation_results[tag].append((f1, p, r))

print(relation_results)
relation_results = {
    t: tuple(sum(metrics) / len(metrics) for metrics in list(zip(*v))) for t, v in relation_results.items()
}

entity_results = {
    t: sum(v) / len(v) for t, v in entity_results.items()
}

print(relation_results)
print(entity_results)

df: pd.DataFrame = pd.read_pickle('experiments.pkl')
print(df)
rows = []
for tag, value in relation_results.items():
    rows.append({
        'experiment_name': 'jerex-relations',
        'tag': tag.lower(),
        'f1': value[0],
        'p': value[1],
        'r': value[2]
    })
for tag, value in entity_results.items():
    rows.append({
        'experiment_name': 'jerex-entities',
        'tag': tag.lower(),
        'f1': value
    })

jerex_df = pd.DataFrame.from_records(rows).set_index(['experiment_name', 'tag'])
df = jerex_df.combine_first(df)
print(df)
pd.to_pickle(df, 'experiments.jerex.pkl')
