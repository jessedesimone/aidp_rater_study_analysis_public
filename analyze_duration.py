import pandas as pd
from scipy import stats

v = pd.read_excel('final_variance_case.xlsx', 'Case Difficulty Rankings')
v = v.loc[:, ~v.columns.str.startswith('Unnamed')].copy()
v.columns = ['Rank', 'Case', 'Dx', 'BLUP', 'Acc', 'Corr', 'N', 'ClinAcc', 'ClinCorr', 'ClinN', 'AidpAcc', 'AidpCorr', 'AidpN']
dd = pd.read_excel('disease_duration.xlsx')
dd.columns = ['Subject', 'dur']
dd['Case'] = dd.Subject.str.replace('-', '_', regex=False)
d = v.merge(dd[['Case', 'dur']], on='Case', how='left')


def mean_sd(s):
    return f'{s.mean():.1f} ({s.std():.1f})'


def compare(sub, label):
    clin = 100 * sub.ClinCorr.sum() / sub.ClinN.sum()
    aidp = 100 * sub.AidpCorr.sum() / sub.AidpN.sum()
    _, p = stats.fisher_exact([
        [sub.AidpCorr.sum(), sub.AidpN.sum() - sub.AidpCorr.sum()],
        [sub.ClinCorr.sum(), sub.ClinN.sum() - sub.ClinCorr.sum()],
    ])
    print(f'{label}: cases={len(sub)}, Clinical {clin:.1f}%, AIDP+ {aidp:.1f}%, gain {aidp - clin:+.1f}, p={p:.2e}')


print(f'All cases: n={len(d)}, duration mean (SD) {mean_sd(d.dur)}, missing {d.dur.isna().sum()}')
for dx, g in d.groupby('Dx'):
    print(f'{dx}: n={len(g)}, duration mean (SD) {mean_sd(g.dur)}')

atypical = d[d.Dx != 'PD']
print(f'Atypical: n={len(atypical)}, <2 y {(atypical.dur < 2).sum()}, <3 y {(atypical.dur < 3).sum()}')
compare(atypical, 'Atypical, all')
compare(atypical[atypical.dur < 2], 'Atypical, <2 y')
compare(atypical[atypical.dur >= 2], 'Atypical, >=2 y')
