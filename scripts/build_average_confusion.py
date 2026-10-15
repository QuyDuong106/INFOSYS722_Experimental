"""Plot the descriptive mean CM across five fixed Experimental LOPO families."""
from pathlib import Path
import csv, json, os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/gallstone-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/lopo/metrics.csv'
FAMILIES = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'stacking']
with SOURCE.open() as f:
    rows = [r for r in csv.DictReader(f) if r['model'] in FAMILIES]
assert len(rows) == 5 and {r['model'] for r in rows} == set(FAMILIES)
assert all(int(r['rows']) == 319 and r['seed'] == '42' for r in rows)
mean = {k: sum(int(r[k]) for r in rows) / 5 for k in ['TN','FP','FN','TP']}
assert sum(mean.values()) == 319
cm = [[mean['TN'],mean['FP']], [mean['FN'],mean['TP']]]
fig, ax = plt.subplots(figsize=(3.8,3.2))
ax.imshow(cm,cmap='Blues',vmin=0,vmax=161)
for y in range(2):
    for x in range(2):
        kind = [['TN','FP'],['FN','TP']][y][x]
        ax.text(x,y,f'{kind}\n{cm[y][x]:.1f}',ha='center',va='center',fontsize=13,color='white' if cm[y][x]>80 else '#18324a')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Absent','Present'],yticklabels=['Absent','Present'],xlabel='Predicted class',ylabel='Observed class',title='Mean across five fixed LOPO models')
fig.tight_layout()
out=ROOT/'figures/evaluation'
out.mkdir(exist_ok=True)
fig.savefig(out/'average_fixed_lopo_confusion.pdf',bbox_inches='tight')
fig.savefig(out/'average_fixed_lopo_confusion.png',dpi=300,bbox_inches='tight')
(out/'average_fixed_lopo_confusion.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT.parent)),families=FAMILIES,seed=42,mean_counts=mean,error_rate=(mean['FP']+mean['FN'])/319,interpretation='Mean across model families, not seeds or independent patients'),indent=2))
print(mean, 'mean error rate', (mean['FP']+mean['FN'])/319)
