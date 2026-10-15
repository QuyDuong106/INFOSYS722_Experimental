"""Render report evidence from existing CSV outputs; does not refit models."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gallstone-report-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
OUT = HERE / 'figures' / 'modeling'
GEN = HERE / 'generated'
OUT.mkdir(parents=True, exist_ok=True)
GEN.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})
models = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'stacking']
names = ['Logistic regression', 'Random forest', 'XGBoost', 'LightGBM', 'Stacking']
colors = ['#16697a', '#c65d21']
metrics = pd.read_csv(RESULTS / 'holdout/metrics.csv')
valid = metrics.query("evaluation == 'validation'")

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.png'), dpi=160)
    plt.close(fig)

def table(caption, columns, rows, widths=None):
    spec = widths or ('l' * len(columns))
    return '\\begin{table}[H]\n\\centering\\small\n\\caption{' + caption + '}\n\\begin{tabular}{' + spec + '}\n\\toprule\n' + ' & '.join(columns) + r' \\' + '\n\\midrule\n' + '\n'.join(' & '.join(map(str,row)) + r' \\' for row in rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, metric, title in zip(axes, ['roc_auc', 'recall'], ['Validation ROC–AUC', 'Validation sensitivity']):
    for j, selection in enumerate(['cart', 'rfe']):
        for i, model in enumerate(models):
            values = valid.query('selection == @selection and model == @model')[metric].to_numpy()
            assert len(values) == 5
            x = i + (j - .5) * .27
            ax.scatter(x + np.linspace(-.045,.045,5), values, color=colors[j], alpha=.4, s=17)
            ax.scatter(x, values.mean(), color=colors[j], marker='D', s=48,
                       label=['CART', 'RFE–CART'][j] if i == 0 else None)
    ax.axhline(.8, color='#777777', ls='--', lw=1)
    ax.set(title=title, ylim=(.55,.95), xticks=range(5), xticklabels=['Logistic','Forest','XGBoost','LightGBM','Stacking'])
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=.15)
axes[0].legend(loc='lower left', frameon=False)
fig.tight_layout(); save(fig, 'model_comparison')

fig, axes = plt.subplots(1,5,figsize=(11,2.7))
for ax, model, name in zip(axes,models,names):
    r = valid.query("seed == 42 and selection == 'rfe' and model == @model").iloc[0]
    cm=np.array([[r.TN,r.FP],[r.FN,r.TP]],dtype=int)
    assert cm.sum()==48
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=24)
    for (y,x),v in np.ndenumerate(cm): ax.text(x,y,str(v),ha='center',va='center',fontsize=13,color='white' if v>14 else '#123')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Absent','Present'],yticklabels=['Absent','Present'],title=name+'\nAUC '+f'{r.roc_auc:.3f}',xlabel='Predicted')
    ax.tick_params(labelsize=8)
axes[0].set_ylabel('Observed'); fig.tight_layout(); save(fig,'execution_evidence')

cv=pd.read_csv(RESULTS/'tuning/seed_42_logistic_cv.csv')
fig,ax=plt.subplots(figsize=(7,3.1))
ax.bar(range(len(cv)),cv.mean_test_score,color=[colors[0] if r==1 else '#a8b9c0' for r in cv.rank_test_score])
for i,r in cv.iterrows(): ax.text(i,r.mean_test_score+.0015,f'{r.mean_test_score:.4f}',ha='center')
ax.set(xticks=range(len(cv)),xticklabels=[f'C = {v:g}' for v in cv.param_model__C],ylim=(.8,.9),ylabel='Mean training-CV ROC–AUC',title='Tested alternatives · logistic regression · seed 42')
fig.tight_layout(); save(fig,'parameter_search')
rows=[]
for model,name in zip(models[:-1],names[:-1]):
    c=pd.read_csv(RESULTS/f'tuning/seed_42_{model}_cv.csv')
    best=c.sort_values('rank_test_score').iloc[0]
    import ast
    pars=ast.literal_eval(best['params'])
    desc=', '.join(f'{k.replace("model__", "").replace("_", " ")}={v}' for k,v in pars.items())
    rows.append([name,len(c),desc,f'{best.mean_test_score:.4f}'])
text=table('Executed training-CV alternatives: seed 42, RFE--CART. Winning configurations are selected within each family; scores are not test results.', ['Model','Trials','Best tested settings','CV AUC'],rows,'lp{0.8cm}p{6.5cm}r')
text+='\nAcross seeds 42, 7, 21, 84 and 123, the selected logistic $C$ values are 0.1, 0.1, 0.1, 1 and 10. The figure below exposes the actual seed-42 alternatives, rather than only listing a proposed search grid. Source: \\path{results/tuning/seed_42_*_cv.csv}.\n'
(GEN/'parameter_evidence.tex').write_text(text)

perm=pd.read_csv(RESULTS/'iteration_patterns/permutation.csv')
means=perm.groupby('feature').auc_drop.mean().sort_values().tail(8)
fig,ax=plt.subplots(figsize=(8,4))
ax.barh(means.index,means.values,color=colors[0],alpha=.75)
for i,f in enumerate(means.index): ax.scatter(perm.query('feature == @f').auc_drop,np.full(5,i),color='#27343a',s=15,zorder=3)
ax.axvline(0,color='black',lw=.7)
ax.set(xlabel='Validation ROC–AUC decrease after shuffling',title='Model reliance on predictors · RFE logistic reference')
fig.tight_layout(); save(fig,'permutation_patterns')

cycle=pd.read_csv(RESULTS/'iteration_patterns/cycle_metrics.csv')
fig,axes=plt.subplots(1,2,figsize=(9,4))
for ax,metric,title in zip(axes,['auc','sensitivity'],['Validation ROC–AUC','Validation sensitivity']):
    for seed,g in cycle.groupby('seed',sort=False):
        vals=g.set_index('top_k').loc[[20,10],metric]
        ax.plot([0,1],vals,marker='o',lw=1.4,label=f'Seed {seed}')
    ax.axhline(.8,color='#777777',ls='--',lw=1)
    ax.set(xticks=[0,1],xticklabels=['Before: 20 predictors','After: 10 predictors'],title=title,xlim=(-.15,1.15),ylim=(.64,.91))
    ax.grid(axis='y',alpha=.15)
axes[1].legend(fontsize=8,loc='upper right',frameon=False)
fig.tight_layout(); save(fig,'iteration_comparison')

rows=[]
for folder,label in [('holdout','Fixed holdout'),('tuning','Tuned holdout'),('lopo','Leave-one-out')]:
    pred=pd.read_csv(RESULTS/folder/'predictions.csv')
    met=pd.read_csv(RESULTS/folder/'metrics.csv')
    rows.append([label,len(met),f'{len(pred):,}',met['model'].nunique()])
text=table('Execution inventory counted from saved metrics and prediction files. Counts include the dummy where present and repeated evaluations of the same source records.', ['Workflow','Metric rows','Prediction rows','Model families'],rows)
text+='\nSources: \\path{results/holdout/}, \\path{results/tuning/} and \\path{results/lopo/}, each containing \\path{metrics.csv} and \\path{predictions.csv}. Prediction rows are evaluation appearances, not unique patients. Saved fitted pipelines are in \\path{models/holdout/}, \\path{models/tuned/} and \\path{models/lopo/}.\n'
(GEN/'execution_evidence.tex').write_text(text)
selected=pd.read_csv(RESULTS/'holdout/selected_test_metrics.csv')
rows=[]
for col,label in [('roc_auc','ROC--AUC'),('recall','Sensitivity'),('specificity','Specificity'),('f1','F1')]:
    rows.append([label,f'{selected[col].mean():.3f}',f'{selected[col].min():.3f}--{selected[col].max():.3f}'])
text=table('Internal test performance of the validation-selected fixed procedure across five seeds. Ranges describe split variation, not confidence intervals.', ['Metric','Mean','Minimum--maximum'],rows)
text+='\nThese results come from \\path{results/holdout/selected_test_metrics.csv}; selectors vary by seed. They describe the selected procedure, whereas the all-candidate tables in Section~\\ref{subsec:statistical-patterns} describe individual models. Reused, overlapping test partitions limit independence.\n'
(GEN/'evaluation_summary.tex').write_text(text)
print('Generated five figures and three evidence tables from saved results.')

# Like-for-like descriptive comparison: fixed RFE candidates only.
lopo = pd.read_csv(RESULTS / 'lopo/metrics.csv').set_index('model')
rows = []
for model, name in zip(models, names):
    h = metrics.query("model == @model and selection == 'rfe'")
    v = h.query("evaluation == 'validation'")
    t = h.query("evaluation == 'test'")
    assert len(v) == len(t) == 5
    rows.append([name, *[f'{x:.3f}' for x in [v.roc_auc.mean(), t.roc_auc.mean(), lopo.loc[model, 'roc_auc'], v.recall.mean(), t.recall.mean(), lopo.loc[model, 'recall']]]])
comparison = r'''\begin{table}[H]
\centering\small
\caption{Holdout and LOPO comparison for fixed RFE--CART candidates at threshold 0.5. Holdout entries average five seeds; LOPO entries pool 319 held-out predictions from one complete run.}
\label{tab:holdout-lopo-comparison}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{ROC--AUC} & \multicolumn{3}{c}{Sensitivity} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Model & Holdout val. & Holdout test & LOPO & Holdout val. & Holdout test & LOPO \\
\midrule
'''
comparison += '\n'.join(' & '.join(row) + r' \\' for row in rows)
comparison += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
(GEN / 'holdout_lopo_comparison.tex').write_text(comparison)
