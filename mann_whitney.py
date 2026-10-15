import sys
import pandas as pd
from scipy import stats

df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'test_confidence.xlsx')
clinical = df['clinical'].dropna()
aidp = df['AIDP'].dropna()

summary = pd.DataFrame(
    {name: [s.median(), s.quantile(0.25), s.quantile(0.75)] for name, s in [('clinical', clinical), ('AIDP', aidp)]},
    index=['Median', 'Q1', 'Q3'],
)
u, p = stats.mannwhitneyu(clinical, aidp, alternative='two-sided')
rank_biserial = 1 - 2 * u / (len(clinical) * len(aidp))

print(summary)
print(f'U = {u:.2f}')
print(f'p = {p:.4e}')
print(f'rank-biserial = {rank_biserial:.4f}')
print(f'n = {len(clinical)}, {len(aidp)}')
