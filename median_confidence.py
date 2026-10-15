import sys
import pandas as pd

df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'test.xlsx')
cols = ['PD vs. Atypical Confidence', 'MSAp vs. PSP Confidence']


def summarize(series):
    series = series.dropna()
    return pd.Series({
        'Mean (SD)': f'{series.mean():.1f} ({series.std():.1f})',
        'Median (IQR)': f'{series.median():.1f} ({series.quantile(0.25):.1f}-{series.quantile(0.75):.1f})',
    })


def by_accuracy(d):
    return {'All': d, 'Accuracy = 1': d[d['Accuracy'] == 1], 'Accuracy = 0': d[d['Accuracy'] == 0]}


rows = []
for group, dg in df.groupby('Group'):
    subsets = list(by_accuracy(dg).items())
    for dx in ['PD', 'MSAp', 'PSP']:
        subsets += [(f'{dx} | {label}', d) for label, d in by_accuracy(dg[dg['AIDP DX'] == dx]).items()]
    for label, d in subsets:
        for col in cols:
            row = summarize(d[col])
            row['Group'] = group
            row['Subset'] = label
            row['Measure'] = col
            rows.append(row)

out = pd.DataFrame(rows)[['Group', 'Subset', 'Measure', 'Mean (SD)', 'Median (IQR)']]
print(out)
out.to_excel('confidence_summary.xlsx', index=False)
