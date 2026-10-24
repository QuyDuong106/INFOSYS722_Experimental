"""Reconstruct training-only CART selectors from all saved Stacking folds."""
from pathlib import Path
import json, os, sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/codex-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from threadpoolctl import threadpool_limits
from utils import workflow_data, frame_signature, make_evaluation_model

LABELS={'bmi':'BMI','hyperlipidemia':'Hyperlipidaemia','ldl':'LDL cholesterol','vma_kg':'VMA (source field)','crp':'CRP','vitamin_d':'Vitamin D','icw':'Intracellular water','ecw':'Extracellular water','tbw':'Total body water','bm':'Estimated bone mass','mm':'Muscle mass','weight':'Weight','height':'Height','age':'Age','gender':'Sex','tbfr_pct':'Body-fat percentage','hgb':'Haemoglobin','hfa':'Hepatic fat accumulation','tfc':'Total fat content','vfa':'Visceral fat area','ecf_tbw':'ECF/TBW ratio','ast':'AST','alt':'ALT','lm_pct':'Lean-mass percentage','obesity_pct':'Obesity percentage','vfr':'Visceral fat rating','hdl':'HDL cholesterol','dm':'Diabetes','protein_pct':'Protein percentage'}
def main():
 X,y=workflow_data();source=ROOT/'results/stacking_v4';out=ROOT/'figures/selection';out.mkdir(exist_ok=True,parents=True)
 config=json.loads((source/'run_config.json').read_text());assert config['signature']==frame_signature(X,y)
 assert json.loads((source/'progress.json').read_text())['complete']
 rows=[]
 for i,row in enumerate(X.index):
  checkpoint=json.loads((source/'folds'/f'fold_{i:04d}_stacking.json').read_text());assert checkpoint['source_row']==row
  counts=[checkpoint['best_params'][f'{base}__select__n_features_to_select'] for base in ('xgboost','lightgbm','ridge')];assert len(set(counts))==1
  train=X.drop(index=row)
  preparation=make_evaluation_model('logistic',train,config['seed'],'rfe',counts[0])[:2]
  with threadpool_limits(limits=1):preparation.fit(train,y.drop(index=row))
  names=preparation.named_steps['impute'].get_feature_names_out();selector=preparation.named_steps['select'];mask=selector.get_support()
  importance=np.zeros(len(names));importance[mask]=selector.estimator_.feature_importances_
  np.testing.assert_allclose(importance.sum(),1,atol=1e-12)
  for name,selected,score in zip(names,mask,importance):rows.append(dict(fold=i,source_row=row,feature=name,selected=int(selected),importance=score,k=counts[0]))
  if i==0:
   fig,ax=plt.subplots(figsize=(8,4.2))
   plot_tree(selector.estimator_,feature_names=[LABELS.get(n,n) for n in names[mask]],max_depth=1,filled=True,rounded=True,fontsize=12,ax=ax,class_names=['Absent','Present'])
   fig.tight_layout()
   for suffix in ('pdf','png'):fig.savefig(out/f'cart_selection_tree.{suffix}',dpi=200,bbox_inches='tight')
   plt.close(fig)
  if (i+1)%50==0:print('Selectors reconstructed:',i+1,flush=True)
 evidence=pd.DataFrame(rows);evidence.to_csv(out/'cart_fold_importance.csv',index=False)
 summary=evidence.groupby('feature').agg(mean_cart_importance=('importance','mean'),sd_cart_importance=('importance','std'),selected_folds=('selected','sum')).sort_values('mean_cart_importance',ascending=False)
 summary['selected_pct']=100*summary.selected_folds/len(X);summary.to_csv(out/'cart_importance.csv')
 (out/'protocol.json').write_text(json.dumps(dict(source='results/stacking_v4',folds=len(X),signature=frame_signature(X,y),seed=config['seed'],cart_max_depth=5,cart_min_samples_leaf=4,selection='RFE, step 1',aggregation='Final selector CART importance, zero for excluded features; equal weight per outer fold',shared_selector='Identical preparation, selector and k across all three base learners on the full outer training fold; counted once per fold',illustration='First outer fold, source row 0 excluded; top two levels of the depth-5 selector, not the Stacking classifier'),indent=2)+'\n')
 print(summary.head(12).to_string())
if __name__=='__main__':main()
