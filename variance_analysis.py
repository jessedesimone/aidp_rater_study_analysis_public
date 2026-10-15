import argparse, re, os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
import scipy.sparse as sp

def fit_glmm(df, fixed=('group',), y='correct', g1='case', g2='rater'):
    d = df.reset_index(drop=True)
    X = np.ones((len(d), 1))
    for f in fixed:
        X = np.column_stack([X, (d[f] == sorted(d[f].unique())[-1]).astype(float)])
    c_codes, c_lv = pd.factorize(d[g1]); r_codes, r_lv = pd.factorize(d[g2])
    nc, nr = len(c_lv), len(r_lv)
    n=len(d); Z = sp.csr_matrix((np.ones(2*n),(np.r_[np.arange(n),np.arange(n)],np.r_[c_codes,nc+r_codes])),shape=(n,nc+nr))
    yv = d[y].values.astype(float); p = X.shape[1]
    def inner(theta, beta, u0):
        sc, sr = np.exp(theta); prec = np.r_[np.full(nc, 1/sc**2), np.full(nr, 1/sr**2)]
        u = np.zeros_like(u0)
        def pen(u):
            eta = X@beta + Z@u
            return np.sum(yv*eta - np.logaddexp(0, eta)) - 0.5*np.sum(prec*u**2)
        cur = pen(u)
        for _ in range(200):
            mu = expit(X@beta + Z@u); W = mu*(1-mu)
            grad = Z.T@(yv-mu) - prec*u
            H = (Z.T@sp.diags(W)@Z).toarray() + np.diag(prec)
            step = np.linalg.solve(H, grad); t = 1.0
            while pen(u + t*step) < cur - 1e-12 and t > 1e-8: t /= 2
            u = u + t*step; new = pen(u)
            if abs(new-cur) < 1e-11 and np.abs(t*step).max() < 1e-8: cur = new; break
            cur = new
        mu = expit(X@beta + Z@u); W = mu*(1-mu)
        H = (Z.T@sp.diags(W)@Z).toarray() + np.diag(prec)
        eta = X@beta + Z@u
        ll = np.sum(yv*eta - np.logaddexp(0, eta)) - 0.5*np.sum(prec*u**2)
        ll += 0.5*(np.sum(np.log(prec)) - np.linalg.slogdet(H)[1])
        return ll, u
    state = {'u': np.zeros(nc+nr)}
    def nll(par):
        ll, u = inner(par[:2], par[2:], state['u']); state['u'] = u; return -ll if np.isfinite(ll) else 1e10
    x0 = np.r_[np.log([1.7, 0.9]), 2.0, np.zeros(p-1)]
    res = minimize(nll, x0, method='Nelder-Mead', options={'xatol':1e-6,'fatol':1e-10,'maxiter':2000})
    res = minimize(nll, res.x, method='Nelder-Mead', options={'xatol':1e-7,'fatol':1e-11,'maxiter':2000})
    ll, u = inner(res.x[:2], res.x[2:], state['u'])
    sc, sr = np.exp(res.x[:2])
    return dict(var_case=sc**2, var_rater=sr**2, beta=res.x[2:], loglik=ll,
                blup_case=pd.Series(u[:nc], index=c_lv), blup_rater=pd.Series(u[nc:], index=r_lv))

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[idx] * n / rank); q[idx] = prev
    return q

def sig_symbol(v):
    return '***' if v < .001 else '**' if v < .01 else '*' if v < .05 else '\u2020' if v < .10 else ''

def case_table(d, blup):
    rows = []
    for case, g in d.groupby('case'):
        row = {'Case': str(case), 'Diagnosis': g.case_type.iloc[0], 'BLUP': round(blup[case], 4),
               'Overall Accuracy': g.correct.mean(), 'Overall Correct': int(g.correct.sum()), 'Overall N': len(g)}
        for lab, grp in [('Clinical', 'Clinical'), ('Clinical+Neuropacs', 'Clinical+neuropacs')]:
            s = g[g.group == grp]
            row.update({f'{lab} Accuracy': s.correct.mean(), f'{lab} Correct': int(s.correct.sum()), f'{lab} N': len(s)})
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(['BLUP', 'Case']).reset_index(drop=True)
    t.insert(0, 'Rank', t['BLUP'].rank(method='min').astype(int))
    t = t.rename(columns={'BLUP': 'Random Intercept\n(BLUP)'})
    return t

def neurologist_table(d, blup_rater, raw, spec):
    names = raw.drop_duplicates('Neurologist ID').set_index('Neurologist ID')['Name']
    acc = d.groupby('rater').correct.mean() * 100
    grp = d.groupby('rater').group.first()
    t = pd.DataFrame({'Neurologist ID': acc.index, 'Name': names.reindex(acc.index).values,
                      'Accuracy (%)': acc.round(3).values, 'Random Intercept (BLUP)': blup_rater.reindex(acc.index).values,
                      'group': grp.values})
    t['Specialty'] = t['Name'].map(spec) if spec is not None else ''
    out = []
    for g in ['Clinical', 'Clinical+neuropacs']:
        s = t[t.group == g].sort_values(['Accuracy (%)', 'Random Intercept (BLUP)', 'Name'])
        out.append(pd.DataFrame([{'Neurologist ID': g}]))
        out.append(s[['Neurologist ID', 'Name', 'Specialty', 'Accuracy (%)', 'Random Intercept (BLUP)']])
        out.append(pd.DataFrame([{}]))
    return pd.concat(out[:-1], ignore_index=True)

def num(series):
    s = series.astype(str).str.strip()
    out = pd.to_numeric(s.str.extract(r'^(-?\d+(?:\.\d+)?)')[0], errors='coerce')
    out = out.where(~s.str.lower().isin(['successful', 'unsuccessful']), (s.str.lower() == 'successful').astype(float))
    return out

def find(df, pattern):
    hits = [c for c in df.columns if re.search(pattern, str(c), re.I)]
    if len(hits) != 1: raise KeyError(f'{pattern!r} matched {hits}')
    return hits[0]

SUMMARY_ITEMS = {
    'MOCA':   [('Visuospatial/Executive', r'Visuospacial/Executive Total'), ('Naming', r'Naming Total'), ('Attention', r'Attention Total'),
               ('Language', r'Language Total'), ('Abstraction', r'Abstraction Total'), ('Delayed Recall', r'Delayed Recall Score'),
               ('Orientation', r'Orientation Total'), ('Total Score', r'^\d+\. Total Score$')],
    'PSPRS':  [('History', r'History \(0-24\)'), ('Mentation', r'Mentation \(0-16\)'), ('Bulbar', r'Bulbar \(0-8\)'),
               ('Ocular', r'Ocular \(0-16\)'), ('Limb', r'Limb \(0-16\)'), ('Gait', r'Gait \(0-20\)'), ('Total Score', r'TOTAL \(0-100\)')],
    'UMSARS': [('Part I', r'Total score Part I$'), ('Part II', r'Total score Part II$')],
    'UPDRS':  [('Mentation/Behavior/Mood', r'Mentation Behavior and Mood Score'), ('Activity of Daily Living', r'Activity of Daily Living Score'),
               ('Motor Examination', r'Motor Examination Score'), ('Total Score', r'^\d+\. Total Score$')],
}

def load_scale(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df['Subject ID'] = df['Subject ID'].astype(str).str.strip().str.replace('-', '_')
    return df.drop_duplicates('Subject ID').set_index('Subject ID')

def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 4 or x[m].nunique() < 2: return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(x[m], y[m]); return r, p, int(m.sum())

def spearman_block(blup, scale_df, items):
    rows, praw = [], []
    for lab, col in items:
        r, p, n = corr(blup, num(scale_df[col]).reindex(blup.index))
        rows.append({'Item': lab, 'n': n, 'Spearman \u03c1': round(r, 4), 'p': round(p, 4)}); praw.append(p)
    t = pd.DataFrame(rows)
    t['q'] = np.round(bh_fdr(np.nan_to_num(praw, nan=1.0)), 4)
    t['sig'] = t['q'].map(sig_symbol)
    return t

def run_spearman(blup_by_dx, data_dir, out_path):
    note1 = 'Spearman Correlation: GLMM BLUP vs Clinical Scales \u2014 {}'
    note2 = ('rho: positive = higher score \u2192 easier diagnosis (more typical); negative = higher score \u2192 harder diagnosis '
             '(atypical). BLUP from GLMM random intercept: more positive = easier case.')
    summ, broad = [], {}
    msa_items, psp_ocular, psp_gait = None, None, None
    for dx in ['MSAp', 'PSP', 'PD']:
        path = os.path.join(data_dir, f'clinical_scales_{dx}.xlsx'); blup = blup_by_dx[dx]
        for scale, items in SUMMARY_ITEMS.items():
            sh = load_scale(path, scale)
            t = spearman_block(blup, sh, [(lab, find(sh, pat)) for lab, pat in items])
            for _, r in t.iterrows():
                summ.append([dx, scale, r['Item'], r['n'], r['Spearman \u03c1'], r['p'], r['q'], r['sig']])
        if dx == 'MSAp':
            sh = load_scale(path, 'UMSARS'); cols = list(sh.columns)
            c0 = cols.index(find(sh, r'^\d+\. 1\. Speech$')); c1 = cols.index(find(sh, r'Total score Part I$'))
            msa_items = spearman_block(blup, sh, [(c, c) for c in cols[c0:c1]])
        if dx == 'PSP':
            sh = load_scale(path, 'PSPRS')
            ocular = ['Voluntary upward saccades', 'Voluntary downward saccades', 'Voluntary left and right saccades', 'Eyelid dysfunction']
            gait = ['Neck rigidity or dystonia', 'Arising from chair', 'Gait', 'Postural stability (on backward pull)',
                    'Sitting down (may touch seat or back but not arms of chair)']
            psp_ocular = spearman_block(blup, sh, [(k, find(sh, r'^\d+\. \d+\. ' + re.escape(k) + r'$|^\d+\. \d+\. ' + re.escape(k))) for k in ocular])
            psp_gait = spearman_block(blup, sh, [(k, find(sh, r'^\d+\. \d+\. ' + re.escape(k) + r'(?!.*\bor\b.*chair)')) if k != 'Gait'
                                                 else (k, find(sh, r'^\d+\. 26\. Gait$')) for k in gait])
        rows = []
        x = pd.ExcelFile(path)
        for sheet in x.sheet_names:
            sh = load_scale(path, sheet)
            lab = {'demographic': 'Demographics', 'epworth_sleep': 'Epworth Sleepiness Scale', 'hamilton_anxiety': 'Hamilton Anxiety',
                   'hamilton_depression': 'Hamilton Depression', 'MOCA': 'MOCA', 'schwab_england': 'Schwab & England', 'PDQ39': 'PDQ-39',
                   'PSPRS': 'PSPRS', 'UMSARS': 'UMSARS', 'UPDRS': 'UPDRS', 'RBD': 'RBD', 'neuro_screen': 'Neuro screen'}.get(sheet, sheet)
            for col in sh.columns:
                if re.search(r'Date|Name|Specify|Initials|Administered|IMPRESSION|PLAN|EXAMINATION|Subject ID', str(col), re.I): continue
                v = num(sh[col]).reindex(blup.index)
                if v.notna().sum() < 10 or v.nunique() < 2: continue
                r, p, n = corr(blup, v)
                rows.append([lab, re.sub(r'^\d+\. ', '', str(col)), col, round(r, 3), round(p, 4), sig_symbol(p), n])
        broad[dx] = pd.DataFrame(rows, columns=['Scale', 'Item', 'Column Name', 'Spearman \u03c1', 'p-value', 'Sig.', 'n'])
    with pd.ExcelWriter(out_path, engine='openpyxl') as xw:
        s = pd.DataFrame(summ, columns=['Diagnosis', 'Clinical Scale', 'Item', 'n', 'Spearman \u03c1', 'p-value', 'FDR q-value', 'Significance'])
        pd.DataFrame([[note1.format('MSAp')], [note2]]).to_excel(xw, sheet_name='Summary', index=False, header=False)
        s.to_excel(xw, sheet_name='Summary', index=False, startrow=2)
        for name, t, lab in [('MSAp Item-Level', msa_items, 'UMSARS Part I Item')]:
            o = t.rename(columns={'Item': lab, 'p': 'p (uncorrected)', 'q': 'FDR q-value', 'sig': 'Significance'})
            o.to_excel(xw, sheet_name=name, index=False)
            pd.DataFrame([[f'FDR correction (Benjamini-Hochberg) applied across all {len(t)} UMSARS Part I items.']]).to_excel(xw, sheet_name=name, index=False, header=False, startrow=len(o) + 2)
        o1 = psp_ocular.rename(columns={'Item': 'PSPRS Ocular Item', 'p': 'p (uncorrected)', 'q': 'FDR q-value', 'sig': 'Significance'})
        o2 = psp_gait.rename(columns={'Item': 'PSPRS Gait Item', 'p': 'p (uncorrected)', 'q': 'FDR q-value', 'sig': 'Significance'})
        o1.to_excel(xw, sheet_name='PSP Item-Level', index=False)
        pd.DataFrame([[f'FDR correction (Benjamini-Hochberg) applied across all {len(o1)} PSPRS Ocular items.']]).to_excel(xw, sheet_name='PSP Item-Level', index=False, header=False, startrow=len(o1) + 1)
        r0 = len(o1) + 2
        o2.to_excel(xw, sheet_name='PSP Item-Level', index=False, startrow=r0)
        pd.DataFrame([[f'FDR correction (Benjamini-Hochberg) applied across all {len(o2)} PSPRS Gait items.']]).to_excel(xw, sheet_name='PSP Item-Level', index=False, header=False, startrow=r0 + len(o2) + 1)
        r = 0
        for dx in ['MSAp', 'PSP', 'PD']:
            pd.DataFrame([[note1.format(dx)], [note2]]).to_excel(xw, sheet_name='Broad Item Analysis', index=False, header=False, startrow=r)
            broad[dx].to_excel(xw, sheet_name='Broad Item Analysis', index=False, startrow=r + 2); r += len(broad[dx]) + 5
    return s, msa_items, psp_ocular, psp_gait

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.'); ap.add_argument('--out-dir', default='.')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    d = pd.read_excel(os.path.join(a.data_dir, 'final_dx_glmm_input.xlsx')); d['case'] = d['case'].astype(str)
    raw = pd.read_excel(os.path.join(a.data_dir, 'final_dx_complete.xlsx'), sheet_name='Raw')
    sp_path = os.path.join(a.data_dir, 'rater_specialty.csv')
    spec = pd.read_csv(sp_path).set_index('Name')['Specialty'] if os.path.exists(sp_path) else None

    fit = fit_glmm(d, ('group',))
    vc, vr = fit['var_case'], fit['var_rater']; tot = vc + vr + np.pi ** 2 / 3
    with open(os.path.join(a.out_dir, 'variance_components.txt'), 'w') as f:
        f.write(f'sigma2 case  = {vc:.3f}\nsigma2 rater = {vr:.3f}\nICC case  = {vc/tot:.3f}\nICC rater = {vr/tot:.3f}\n'
                f'group coef (Clinical+neuropacs vs Clinical) = {fit["beta"][1]:.4f}, OR = {np.exp(fit["beta"][1]):.2f}\nlogLik = {fit["loglik"]:.3f}\n')
    print(open(os.path.join(a.out_dir, 'variance_components.txt')).read())

    ct = case_table(d, fit['blup_case'])
    with pd.ExcelWriter(os.path.join(a.out_dir, 'final_variance_case.xlsx'), engine='openpyxl') as xw:
        ct.to_excel(xw, sheet_name='Case Difficulty Rankings', index=False)
    nt = neurologist_table(d, fit['blup_rater'], raw, spec)
    with pd.ExcelWriter(os.path.join(a.out_dir, 'final_variance_neurologist.xlsx'), engine='openpyxl') as xw:
        nt.to_excel(xw, sheet_name='Explained Variance', index=False)

    ctype = d.groupby('case').case_type.first()
    blup_by_dx = {dx: fit['blup_case'][ctype[ctype == dx].index].round(4) for dx in ['MSAp', 'PSP', 'PD']}
    for dx in blup_by_dx: blup_by_dx[dx].index = blup_by_dx[dx].index.astype(str)
    run_spearman(blup_by_dx, a.data_dir, os.path.join(a.out_dir, 'Spearman_BLUP_ClinicalScales_v6.xlsx'))
    print('done ->', a.out_dir)

if __name__ == '__main__':
    main()
