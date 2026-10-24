"""Paired held-out LR/Stacking explanations and shared-error audit.

No model tuning or evaluation labels enter fitting. LR pipelines are loaded from
completed fold artifacts; Stacking locals are reconstructed from its checkpoints.
"""
from pathlib import Path
import os,json,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/codex-mpl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from feature_selection_evidence import load_fold_pipeline
from optuna_lopo import make_tuning_template
from utils import workflow_data,frame_signature,categorical_columns,disease_probability

LABELS={'crp':'CRP','icw':'Intracellular water','bm':'Estimated bone mass','weight':'Weight','tbfr_pct':'Body-fat percentage','vitamin_d':'Vitamin D','ecw':'Extracellular water','tbw':'Total body water','ecf_tbw':'ECF/TBW ratio','hgb':'Haemoglobin','mm':'Muscle mass','ast':'AST','vfa':'Visceral fat area','lm_pct':'Lean-mass percentage'}
NAMES={'logistic':'Logistic regression','stacking':'Stacking'}

def main():
 X,y=workflow_data();sig=frame_signature(X,y);out=ROOT/'figures/xai/comparison';out.mkdir(parents=True,exist_ok=True)
 sources={'logistic':ROOT/'results/dynamic_individual_v4','stacking':ROOT/'results/stacking_v3'}
 predictions={}
 for name,source in sources.items():
  cfg=json.loads((source/'run_config.json').read_text());assert cfg['signature']==sig
  assert json.loads((source/'progress.json').read_text())['complete']
  pred=pd.read_csv(source/'predictions.csv');pred=pred[pred.model.eq(name)].set_index('source_row').sort_index()
  assert pred.index.equals(X.index) and len(pred)==319
  np.testing.assert_array_equal(pred.actual_present,y)
  np.testing.assert_array_equal(pred.predicted_present,pred.probability_present.ge(.5))
  predictions[name]=pred
 paired=pd.DataFrame({'actual':y})
 for name,pred in predictions.items():
  paired[f'{name}_probability']=pred.probability_present
  paired[f'{name}_prediction']=pred.predicted_present
  paired[f'{name}_wrong']=pred.predicted_present.ne(y)
 shared=paired[paired.logistic_wrong & paired.stacking_wrong].copy();shared['error_type']=np.where(shared.actual.eq(1),'FN','FP')
 shared['joint_wrong_confidence']=np.where(shared.actual.eq(1),1-shared[['logistic_probability','stacking_probability']].max(axis=1),shared[['logistic_probability','stacking_probability']].min(axis=1))
 selected={kind:int(shared[shared.error_type.eq(kind)].sort_values('joint_wrong_confidence',ascending=False,kind='stable').index[0]) for kind in ('FN','FP')}
 paired.to_csv(out/'paired_predictions.csv',index_label='source_row');shared.join(X).to_csv(out/'shared_errors.csv',index_label='source_row')
 comparison=dict(rows=319,both_correct=int((~paired.logistic_wrong & ~paired.stacking_wrong).sum()),shared_wrong=len(shared),logistic_only_wrong=int((paired.logistic_wrong & ~paired.stacking_wrong).sum()),stacking_only_wrong=int((~paired.logistic_wrong & paired.stacking_wrong).sum()),agreement=int(paired.logistic_prediction.eq(paired.stacking_prediction).sum()),shared_FN=shared[shared.error_type.eq('FN')].index.tolist(),shared_FP=shared[shared.error_type.eq('FP')].index.tolist(),selected_cases=selected,auc={n:float(roc_auc_score(y,p.probability_present)) for n,p in predictions.items()})
 (out/'error_overlap.json').write_text(json.dumps(comparison,indent=2)+'\n')
 previous=json.loads((ROOT/'figures/xai/protocol.json').read_text());assert previous['dataset_signature']==sig and previous['source_results']=='results/stacking_v3'
 sampled=previous['sampled_rows'];assert len(sampled)==20
 stack_values=pd.read_csv(ROOT/'figures/xai/sampled_shap.csv',index_col='source_row').loc[sampled,X.columns]
 logistic_values={};local_shap={};local_lime={};metadata=[];lime_rows=[];coefficient_rows=[]
 jobs=[('logistic',r) for r in dict.fromkeys([*sampled,*selected.values()])]+[('stacking',r) for r in selected.values()]
 for name,row in jobs:
  source=sources[name];cfg=json.loads((source/'run_config.json').read_text())
  if name=='logistic':model,cp=load_fold_pipeline(source,name,row)
  else:
   cp=json.loads((source/'folds'/f'fold_{X.index.get_loc(row):04d}_stacking.json').read_text())
   model=make_tuning_template(name,X.drop(index=row),cfg['seed'],cfg['top_k']);model.set_params(**cp['best_params'])
   with threadpool_limits(limits=1):model.fit(X.drop(index=row),y.drop(index=row))
  assert cp['source_row']==row
  train=X.drop(index=row).copy();test=X.loc[[row]].copy()
  for c in X:
   fill=train[c].mode().iloc[0] if c in categorical_columns(train) else train[c].median()
   train[c]=train[c].fillna(fill);test[c]=test[c].fillna(fill)
  def predict(v):
   with threadpool_limits(limits=1):return model.predict_proba(pd.DataFrame(np.asarray(v),columns=X.columns))
  probability=float(predict(test)[0,1]);np.testing.assert_allclose(probability,cp['probability'],atol=1e-8,rtol=0)
  np.testing.assert_allclose(probability,predictions[name].loc[row,'probability_present'],atol=1e-8,rtol=0)
  kernel=shap.KernelExplainer(lambda v:predict(v)[:,1],train.sample(25,random_state=42).to_numpy(),link='identity')
  np.random.seed(42);values=np.asarray(kernel.shap_values(test.to_numpy(),nsamples=256,l1_reg=0.0,silent=True))[0];base=float(kernel.expected_value)
  np.testing.assert_allclose(base+values.sum(),probability,atol=1e-8,rtol=0)
  if name=='logistic':logistic_values[row]=values
  if row in selected.values():
   kind=next(k for k,v in selected.items() if v==row);local_shap[(name,kind)]=values
   pd.DataFrame({'feature':X.columns,'value':test.iloc[0].to_numpy(),'shap':values}).to_csv(out/f'{name}_{kind.lower()}_shap.csv',index=False)
   explainer=LimeTabularExplainer(train.to_numpy(),feature_names=list(X.columns),categorical_features=[X.columns.get_loc(c) for c in categorical_columns(train)],class_names=['Absent','Present'],random_state=42,mode='classification',discretize_continuous=True)
   # Explain all source fields so paired panels can compare the same conditions.
   exp=explainer.explain_instance(test.iloc[0].to_numpy(),predict,labels=(1,),num_features=X.shape[1],num_samples=3000)
   entries=[]
   for (i,weight),(condition,_) in zip(exp.as_map()[1],exp.as_list(label=1)):
    entries.append(dict(feature=X.columns[i],condition=condition,weight=float(weight)))
    lime_rows.append(dict(model=name,source_row=row,error_type=kind,feature=X.columns[i],condition=condition,weight=float(weight)))
   local_lime[(name,kind)]=pd.DataFrame(entries).set_index('feature')
   exp.save_to_file(str(out/f'{name}_{kind.lower()}_lime.html'))
   meta=dict(model=name,source_row=row,error_type=kind,actual=int(y.loc[row]),probability=probability,shap_base=base,lime_r2=float(exp.score),lime_prediction=float(np.asarray(exp.local_pred).ravel()[0]))
   if name=='logistic':
    classifier=model.named_steps['model'];z=model[:-1].transform(test);coef=classifier.coef_[0];logit=float(classifier.intercept_[0]+np.asarray(z).ravel()@coef)
    np.testing.assert_allclose(1/(1+np.exp(-logit)),probability,atol=1e-12)
    meta.update(logistic_C=float(classifier.C),selected_k=int(cp['validation']['selected_k']),logit=logit)
    for feature,beta in zip(model[:-1].get_feature_names_out(),coef):coefficient_rows.append(dict(source_row=row,transformed_feature=feature,coefficient=float(beta),intercept=float(classifier.intercept_[0])))
   metadata.append(meta)
  print('Explained',name,row,f'{probability:.6f}',flush=True)
 lr_values=np.stack([logistic_values[row] for row in sampled]);pd.DataFrame(lr_values,index=sampled,columns=X.columns).to_csv(out/'logistic_sampled_shap.csv',index_label='source_row')
 imp=pd.DataFrame({'logistic':np.abs(lr_values).mean(axis=0),'stacking':stack_values.abs().mean(axis=0)},index=X.columns);imp.index.name='feature';imp['combined_mean']=imp[['logistic','stacking']].mean(axis=1);imp.sort_values('combined_mean',ascending=False).to_csv(out/'global_shap_comparison.csv')
 shap.summary_plot(lr_values,X.loc[sampled],max_display=8,show=False,plot_size=(6.2,3.7));plt.title('Logistic regression: held-out SHAP explanations',fontsize=11)
 for ext in ('pdf','png'):plt.savefig(out/f'logistic_global_shap.{ext}',dpi=220,bbox_inches='tight')
 plt.close('all')
 pd.DataFrame(metadata).to_csv(out/'local_explanation_metadata.csv',index=False);pd.DataFrame(lime_rows).to_csv(out/'local_lime_weights.csv',index=False);pd.DataFrame(coefficient_rows).to_csv(out/'logistic_coefficients.csv',index=False)
 # Compact paired SHAP/LIME panels: two shared cases, not two full-width case plots.
 fig,axes=plt.subplots(2,2,figsize=(9,5.1));colours={'logistic':'#286f8e','stacking':'#aa6139'}
 for r,kind in enumerate(('FN','FP')):
  row=selected[kind];left,right=axes[r]
  scores=np.stack([local_shap[(n,kind)] for n in NAMES]);order=np.argsort(np.abs(scores).mean(axis=0))[-4:];pos=np.arange(4)
  for j,n in enumerate(NAMES):left.barh(pos+(j-.5)*.32,scores[j,order],height=.3,color=colours[n],label=NAMES[n])
  left.set_yticks(pos,[f'{X.columns[i]} = {X.loc[row].iloc[i]:.3g}' for i in order]);left.set_xlabel('SHAP probability contribution',fontsize=12);left.set_title(f'{kind}, row {row}: SHAP',fontsize=13)
  first=local_lime[('logistic',kind)].reindex(X.columns);second=local_lime[('stacking',kind)].reindex(X.columns);assert first.condition.equals(second.condition)
  order=np.argsort(np.abs(np.stack([first.weight,second.weight])).mean(axis=0))[-4:]
  for j,n in enumerate(NAMES):right.barh(pos+(j-.5)*.32,local_lime[(n,kind)].reindex(X.columns).weight.iloc[order],height=.3,color=colours[n])
  right.set_yticks(pos,first.condition.iloc[order]);right.set_xlabel('LIME surrogate weight',fontsize=12);right.set_title(f'{kind}, row {row}: LIME',fontsize=13)
 for ax in axes.ravel():ax.axvline(0,color='grey',linewidth=.7);ax.tick_params(labelsize=12);ax.spines[['top','right']].set_visible(False)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=12);fig.tight_layout(rect=(0,0,1,.92))
 for ext in ('pdf','png'):fig.savefig(out/f'shared_errors_shap_lime.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
 lines=[r'\begin{table}[htbp]',r'\caption{Mean absolute probability-scale SHAP contributions for twenty common sampled exclusions. Fields are ordered by the mean of the two model-specific means.}',r'\label{tab:global-shap-comparison}',r'\centering\small',r'\begin{tabular}{@{}lrr@{}}',r'\toprule',r'Feature & Logistic regression & Stacking \\',r'\midrule']
 for feature,v in imp.sort_values('combined_mean',ascending=False).head(6).iterrows():lines.append(f'{LABELS.get(feature,feature.replace("_",r"\_"))} & {v.logistic:.4f} & {v.stacking:.4f} '+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);(ROOT/'global_shap_comparison.tex').write_text('\n'.join(lines)+'\n')
 protocol=dict(signature=sig,sources={k:str(v.relative_to(ROOT)) for k,v in sources.items()},sampled_rows=sampled,selected_cases=selected,selection='Maximise minimum wrong-class confidence across both models, separately for shared FN/FP; retrospective',shap_background=25,shap_samples=256,lime_samples=3000,lime_features=38,local_display_features=4,seed=42,global_stacking='Reused validated prior held-out SHAP values for the same twenty records and settings',global_logistic='Saved excluded-record pipelines, training-only backgrounds',local_stacking='Reconstructed excluded-record pipelines; matched saved probabilities',interpretation='Model behaviour, not causal effects or independently validated clinical explanations')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 print('ERROR OVERLAP',json.dumps(comparison));print('GLOBAL',imp.sort_values('combined_mean',ascending=False).head(8).to_string());print('LOCAL',pd.DataFrame(metadata).to_string(index=False))
if __name__=='__main__':main()
