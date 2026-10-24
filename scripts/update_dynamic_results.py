"""Validate completed dynamic runs and export paper results and CART selection plots."""
from pathlib import Path
import json,os,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/codex-mpl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
from utils import workflow_data,frame_signature

NAMES={'logistic':'Logistic regression','random_forest':'Random forest','xgboost':'XGBoost','lightgbm':'LightGBM','stacking':'Stacking'}
LABELS={'crp':'CRP','vitamin_d':'Vitamin D','bm':'Estimated bone mass','hyperlipidemia':'Hyperlipidaemia','alp':'ALP','vma_kg':'VMA (source field)','ecf_tbw':'ECF/TBW ratio','ecw':'Extracellular water','ast':'AST','weight':'Weight'}

def main():
 X,y=workflow_data();signature=frame_signature(X,y);dynamic=ROOT/'results/dynamic_individual_v4';stacking=ROOT/'results/stacking_v4';out=ROOT/'figures/selection';out.mkdir(exist_ok=True)
 configs=[]
 for source in (dynamic,stacking):
  config=json.loads((source/'run_config.json').read_text());progress=json.loads((source/'progress.json').read_text())
  assert progress['complete'] and progress['evaluated_folds']==len(X)==319
  assert config['signature']==signature and config['k_values']==[10,20,30,38]
  assert config['seed']==42 and config['trials']==20 and config['inner_folds']==3 and config['threshold']==.5
  configs.append(config)
 assert configs[0]['packages']==configs[1]['packages']
 a=pd.read_csv(dynamic/'feature_selection_all_folds.csv');a=a[['model','fold','source_row','feature','selected','cart_importance']]
 b=pd.read_csv(out/'cart_fold_importance.csv').rename(columns={'importance':'cart_importance'});b['model']='stacking'
 assert json.loads((out/'protocol.json').read_text())['signature']==signature
 evidence=pd.concat([a,b[a.columns]],ignore_index=True)
 assert set(evidence.model)==set(NAMES)
 assert len(evidence)==len(X)*len(X.columns)*len(NAMES)
 assert not evidence.duplicated(['model','fold','feature']).any()
 assert set(evidence.feature)==set(X.columns)
 np.testing.assert_allclose(evidence.groupby(['model','fold']).cart_importance.sum(),1,atol=1e-10)
 metrics=[];counts=[]
 for name in NAMES:
  source=stacking if name=='stacking' else dynamic
  pred=pd.read_csv(source/'predictions.csv');pred=pred.loc[pred.model.eq(name)].sort_values('source_row')
  assert len(pred)==len(X) and pred.source_row.is_unique
  np.testing.assert_array_equal(pred.actual_present,y.loc[pred.source_row])
  np.testing.assert_array_equal(pred.predicted_present,pred.probability_present.ge(.5).astype(int))
  for fold in range(len(X)):
   cp=json.loads((source/'folds'/f'fold_{fold:04d}_{name}.json').read_text())
   subset=evidence.loc[evidence.model.eq(name)&evidence.fold.eq(fold)]
   assert cp['source_row']==int(pred.iloc[fold].source_row)
   np.testing.assert_allclose(cp['probability'],pred.iloc[fold].probability_present,atol=1e-12)
   k=cp['best_params']['xgboost__select__n_features_to_select'] if name=='stacking' else cp['validation']['selected_k']
   assert int(subset.selected.sum())==k and subset.source_row.eq(cp['source_row']).all()
   if name=='stacking':assert all(cp['best_params'][f'{base}__select__n_features_to_select']==k for base in ('xgboost','lightgbm','ridge'))
   else:
    saved=pd.DataFrame(cp['feature_selection']).set_index('feature').sort_index();actual=subset.set_index('feature').sort_index()
    np.testing.assert_array_equal(saved.selected,actual.selected)
    np.testing.assert_allclose(saved.cart_importance,actual.cart_importance,atol=1e-12)
   counts.append(dict(model=name,fold=fold,selected_k=k))
  actual=pred.actual_present;prob=pred.probability_present;decision=prob.ge(.5);tn,fp,fn,tp=confusion_matrix(actual,decision,labels=[0,1]).ravel()
  metrics.append(dict(model=name,accuracy=accuracy_score(actual,decision),roc_auc=roc_auc_score(actual,prob),sensitivity=recall_score(actual,decision),specificity=tn/(tn+fp),precision=precision_score(actual,decision),f1=f1_score(actual,decision),fnr=fn/(fn+tp),TN=tn,FP=fp,FN=fn,TP=tp))
 metrics=pd.DataFrame(metrics)
 for name in NAMES:
  source=stacking if name=='stacking' else dynamic
  stored=pd.read_csv(source/'metrics.csv').set_index('model').loc[name]
  computed=metrics.set_index('model').loc[name]
  for key in ('accuracy','roc_auc','specificity','precision','f1','fnr'):
   np.testing.assert_allclose(computed[key],stored[key],atol=1e-10,rtol=0)
  np.testing.assert_allclose(computed.sensitivity,stored.recall,atol=1e-10,rtol=0)
 metrics.to_csv(out/'dynamic_model_metrics.csv',index=False)
 counts=pd.DataFrame(counts).groupby(['model','selected_k']).size().unstack(fill_value=0).reindex(NAMES)
 counts.to_csv(out/'dynamic_feature_count_summary.csv')
 per=evidence.groupby(['model','feature'],as_index=False).agg(appearances=('selected','sum'),mean_cart_importance=('cart_importance','mean'))
 per['evaluated_folds']=len(X);per['selection_pct']=100*per.appearances/len(X);per.to_csv(out/'dynamic_per_model_features.csv',index=False)
 total=evidence.groupby('feature').agg(appearances=('selected','sum'),mean_cart_importance=('cart_importance','mean'),sd_cart_importance=('cart_importance','std'))
 total['opportunities']=len(X)*len(NAMES);total['selection_pct']=100*total.appearances/total.opportunities
 total=total.reset_index().sort_values(['appearances','mean_cart_importance','feature'],ascending=[False,False,True]).reset_index(drop=True)
 total.insert(0,'rank',np.arange(1,len(total)+1));total.to_csv(out/'dynamic_all_features.csv',index=False)
 top=total.head(10).copy();top['label']=top.feature.map(LABELS).fillna(top.feature);top.to_csv(out/'dynamic_top10_features.csv',index=False)
 # Four-new-model-only aggregation is supplied separately to make the denominator explicit.
 individual=a.groupby('feature').agg(appearances=('selected','sum'),mean_cart_importance=('cart_importance','mean'));individual['opportunities']=319*4;individual['selection_pct']=100*individual.appearances/individual.opportunities
 individual.sort_values(['appearances','mean_cart_importance'],ascending=False).to_csv(out/'dynamic_individual_features.csv')
 lines=[r'\begin{table}[htbp]',r'\caption{Ten most frequently retained features across five models and 319 exclusions (1,595 selection opportunities). Ties are ordered by mean CART score. Stacking is counted once per exclusion.}',r'\label{tab:cart-importance}',r'\centering\small',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Feature & Appearances & Retained (\%) & CART score \\',r'\midrule']
 for r in top.itertuples():lines.append(f'{r.label} & {r.appearances} & {r.selection_pct:.2f} & {r.mean_cart_importance:.4f} '+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(ROOT/'cart_importance.tex').write_text('\n'.join(lines)+'\n')
 columns=['accuracy','roc_auc','sensitivity','specificity','f1','fnr'];lines=[r'\begin{table}[htbp]',r'\caption{Dynamic feature selection: pooled held-out performance at threshold 0.5. All models search $k\in\{10,20,30,38\}$; bold marks observed metric extrema (lower for FNR), without significance testing.}',r'\label{tab:tuned-lopo-results}',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',r'Model & Acc. & AUC & Sens. & Spec. & F1 & FNR \\',r'\midrule']
 for _,r in metrics.iterrows():
  values=[]
  for col in columns:
   best=metrics[col].min() if col=='fnr' else metrics[col].max();v=f'{r[col]:.4f}'
   values.append(r'\textbf{'+v+'}' if np.isclose(r[col],best,rtol=0,atol=1e-12) else v)
  lines.append(NAMES[r.model]+' & '+' & '.join(values)+' '+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(ROOT/'dynamic_performance.tex').write_text('\n'.join(lines)+'\n')
 # Side-by-side frequency and CART scores, with labels readable at manuscript width.
 plot=top.iloc[::-1];fig,axes=plt.subplots(1,2,figsize=(8.2,4.8),sharey=True,gridspec_kw={'width_ratios':[1,1.05]})
 positions=np.arange(len(plot));axes[0].barh(positions,plot.selection_pct,color='#286f8e');axes[0].set_yticks(positions,plot.label);axes[0].set_xlim(0,110);axes[0].set_xlabel('Retained (%)');axes[0].set_title('Selection frequency')
 for i,r in enumerate(plot.itertuples()):axes[0].text(r.selection_pct+1,i,f'{r.selection_pct:.1f}',va='center',fontsize=11)
 axes[1].barh(positions,plot.mean_cart_importance,color='#aa6139');axes[1].set_xlim(0,.52);axes[1].set_xlabel('Mean CART score');axes[1].set_title('Final selector importance')
 for i,r in enumerate(plot.itertuples()):axes[1].text(r.mean_cart_importance+.007,i,f'{r.mean_cart_importance:.3f}',va='center',fontsize=11)
 for ax in axes:
  ax.tick_params(labelsize=12);ax.xaxis.label.set_size(12);ax.title.set_size(13);ax.spines[['top','right']].set_visible(False)
 fig.tight_layout()
 for ext in ('pdf','png'):fig.savefig(out/f'dynamic_top10_features.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
 matrix=per.pivot(index='feature',columns='model',values='selection_pct').loc[top.feature,list(NAMES)]
 fig,ax=plt.subplots(figsize=(8.2,4.8));im=ax.imshow(matrix.to_numpy(),cmap='Blues',vmin=0,vmax=100,aspect='auto')
 ax.set_yticks(range(10),top.label,fontsize=12);ax.set_xticks(range(5),[NAMES[n] for n in NAMES],rotation=25,ha='right',fontsize=11)
 for i in range(10):
  for j in range(5):ax.text(j,i,f'{matrix.iloc[i,j]:.1f}',ha='center',va='center',fontsize=11,color='white' if matrix.iloc[i,j]>65 else 'black')
 fig.colorbar(im,ax=ax,label='Retained (%)');fig.tight_layout()
 for ext in ('pdf','png'):fig.savefig(out/f'dynamic_feature_frequency_by_model.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
 (out/'dynamic_summary_protocol.json').write_text(json.dumps(dict(date='2026-10-24',signature=signature,sources=['results/dynamic_individual_v4','results/stacking_v4'],models=list(NAMES),folds_per_model=319,opportunities=1595,ranking='Selection count descending; ties: mean normalised final CART score descending, then feature name',stacking_weight='One selector per exclusion; identical three-base full-training selectors not tripled',cart='Final RFE tree; removed fields assigned zero; equal weight per model/exclusion',independence='Repeated models and overlapping training folds are not independent patient replications'),indent=2)+'\n')
 print(metrics.to_string(index=False));print(top[['label','appearances','selection_pct','mean_cart_importance']].to_string(index=False))
if __name__=='__main__':main()
