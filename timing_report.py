import sys
import pandas as pd

exclude_path = sys.argv[1] if len(sys.argv) > 1 else None

complete = pd.read_excel('complete_cases.xlsx')
timing = pd.read_csv('timing.csv')
df = complete.merge(timing, on=['Neurologist ID', 'Case ID'], how='left')
df = df.drop(columns=['Name_y', 'Group_y'])
df = df[df['Case ID'] != '15_002']
df = df.rename(columns={
    'Name_x': 'User Name',
    'Group_x': 'Group',
    'Diagnosis Time (sec)': 'diagnosis_time',
    'Completion Time (sec)': 'completion_time',
    'Biomarker Report Time (sec)': 'biomarker_time',
    'Clinical Imaging Time (sec)': 'imaging_time',
    'Clinical Scales Time (sec)': 'scale_time',
    'Clinical Videos Time (sec)': 'video_time',
})
if exclude_path:
    df = df[~df['Case ID'].isin(pd.read_excel(exclude_path).CASE_ID)]

time_cols = ['diagnosis_time', 'completion_time', 'imaging_time', 'biomarker_time', 'scale_time', 'video_time']
cols = ['Group', 'AIDP DX'] + time_cols
combined = pd.concat([
    df[cols].assign(Condition='All'),
    df[df['AIDP DX'] != 'PD'][cols].assign(Condition='Atypical'),
    df[df['AIDP DX'].isin(['PD', 'MSAp', 'PSP'])][cols].assign(Condition=lambda x: x['AIDP DX']),
])

summary = (
    combined.groupby(['Group', 'Condition'])[time_cols]
    .agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('Median', 'median'),
        ('IQR_25', lambda x: x.quantile(0.25)),
        ('IQR_75', lambda x: x.quantile(0.75)),
    ])
    .round(2)
    .stack(level=0)
)
print(summary)
summary.to_excel('time_summary.xlsx')
