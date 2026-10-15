import pandas as pd

# total_clinical, misdiagnosed_clinical, total_aidp, misdiagnosed_aidp
data = {
    'MSAp+PSP': [447, 94, 543, 22],
    'MSAp': [222, 70, 271, 20],
    'PSP': [225, 24, 272, 2],
}

rows = []
for category, (n_clin, mis_clin, n_aidp, mis_aidp) in data.items():
    risk_clin = mis_clin / n_clin
    risk_aidp = mis_aidp / n_aidp
    rr = risk_clin / risk_aidp if risk_aidp > 0 else float('inf')
    rows.append({
        'Category': category,
        'Risk_Clinical': risk_clin,
        'Risk_AIDP': risk_aidp,
        'Relative_Risk': rr,
        'Pct_Reduction': 100 if rr == float('inf') else (1 - 1 / rr) * 100,
    })

print(pd.DataFrame(rows))
