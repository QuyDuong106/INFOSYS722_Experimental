"""Export tuned logistic Leave One Patient Out SHAP/LIME report figures."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from threadpoolctl import threadpool_limits
from utils import workflow_data, make_evaluation_model, categorical_columns, frame_signature


def main():
    X,y = workflow_data()
    source = ROOT/'results/optuna_logistic'
    config = json.loads((source/'run_config.json').read_text())
    assert json.loads((source/'progress.json').read_text())['complete']
    assert config['signature'] == frame_signature(X,y)
    review = pd.read_csv(source/'predictions.csv').set_index('source_row')
    out = ROOT/'figures/xai'
    out.mkdir(parents=True,exist_ok=True)
    errors = {}
    for kind in ['FN','FP']:
        group=review[review.error_type.eq(kind)]
        errors[kind]=int(group.probability_present.idxmin() if kind=='FN' else group.probability_present.idxmax())
    sampled=pd.Series(X.index).sample(20,random_state=42).tolist()
    values_by_row,local_rows={},[]
    for row in dict.fromkeys([*sampled,*errors.values()]):
        fold=X.index.get_loc(row)
        checkpoint=json.loads((source/'folds'/f'fold_{fold:04d}_logistic.json').read_text())
        raw_train=X.drop(index=row)
        train,test=raw_train.copy(),X.loc[[row]].copy()
        for c in train:
            fill=raw_train[c].mode().iloc[0] if c in categorical_columns(raw_train) else raw_train[c].median()
            train[c],test[c]=train[c].fillna(fill),test[c].fillna(fill)
        model=make_evaluation_model('logistic',raw_train,config['seed'],'rfe',config['top_k'])
        model.set_params(**checkpoint['best_params'])
        with threadpool_limits(limits=1):model.fit(raw_train,y.drop(index=row))
        def predict(v):
            with threadpool_limits(limits=1):
                return model.predict_proba(pd.DataFrame(np.asarray(v),columns=X.columns))
        probability=float(predict(test)[0,1])
        np.testing.assert_allclose(probability,checkpoint['probability'],atol=1e-8)
        kernel=shap.KernelExplainer(lambda v:predict(v)[:,1],train.sample(25,random_state=42).to_numpy(),link='identity')
        np.random.seed(42)
        values=np.asarray(kernel.shap_values(test.to_numpy(),nsamples=256,l1_reg=0.0,silent=True))[0]
        base=float(kernel.expected_value)
        np.testing.assert_allclose(base+values.sum(),probability,atol=1e-8)
        values_by_row[row]=values
        print('Explained',row,probability,flush=True)
        if row in errors.values():
            kind=next(k for k,v in errors.items() if v==row)
            lime=LimeTabularExplainer(train.to_numpy(),feature_names=list(X.columns),
                categorical_features=[X.columns.get_loc(c) for c in categorical_columns(train)],
                class_names=['Absent','Present'],random_state=42,mode='classification',discretize_continuous=True)
            exp=lime.explain_instance(test.iloc[0].to_numpy(),predict,labels=(1,),num_features=8,num_samples=3000)
            local_rows.append(dict(source_row=row,error_type=kind,actual=int(y.loc[row]),probability=probability,
                shap_base=base,lime_r2=float(exp.score),lime_prediction=float(np.asarray(exp.local_pred).ravel()[0])))
            fig,axes=plt.subplots(1,2,figsize=(12,4.7))
            order=np.argsort(np.abs(values))[-8:]
            axes[0].barh([f'{X.columns[i]} = {test.iloc[0,i]:.3g}' for i in order],values[order],
                color=['#bd3545' if v>0 else '#2875ad' for v in values[order]])
            axes[0].set_xlabel('SHAP contribution to P(gallstones present)')
            axes[0].set_title(f'(a) SHAP | baseline {base:.3f}',fontsize=11)
            conditions,weights=zip(*exp.as_list(label=1))
            order=np.argsort(np.abs(weights));weights=np.asarray(weights)
            axes[1].barh(np.asarray(conditions)[order],weights[order],color=['#bd3545' if v>0 else '#2875ad' for v in weights[order]])
            axes[1].set_xlabel('LIME local surrogate weight')
            axes[1].set_title(f'(b) LIME | local R² = {exp.score:.3f}',fontsize=11)
            for ax in axes:
                ax.axvline(0,color='gray',linewidth=.8)
                ax.tick_params(axis='both',labelsize=9)
            fig.suptitle(f'Source row {row}: {"false negative" if kind=="FN" else "false positive"} | P(present) = {probability:.3f}',fontsize=12)
            fig.tight_layout()
            for suffix in ['pdf','png']:fig.savefig(out/f'local_{kind.lower()}.{suffix}',dpi=200,bbox_inches='tight')
            plt.close(fig)
            exp.save_to_file(str(out/f'local_{kind.lower()}.html'))
    values=np.stack([values_by_row[row] for row in sampled])
    shap.summary_plot(values,X.loc[sampled],max_display=12,show=False,plot_size=(8,5))
    plt.title('Tuned logistic regression: held-out SHAP explanations',fontsize=12)
    for suffix in ['pdf','png']:plt.savefig(out/f'global_shap.{suffix}',dpi=200,bbox_inches='tight')
    plt.close('all')
    importance=pd.Series(np.abs(values).mean(axis=0),index=X.columns).sort_values(ascending=False)
    importance.rename('mean_absolute_shap').to_csv(out/'global_importance.csv')
    pd.DataFrame(values,index=sampled,columns=X.columns).to_csv(out/'sampled_shap.csv',index_label='source_row')
    pd.DataFrame(local_rows).to_csv(out/'local_explanation_metadata.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(model='logistic',evaluation='Tuned Leave One Patient Out',
        source_results='results/optuna_logistic',sampled_rows=sampled,error_rows=errors,background_rows=25,
        shap_samples=256,lime_samples=3000,seed=42,selection='Most confident errors, selected retrospectively',
        dataset_signature=frame_signature(X,y),interpretation='Separate held-out fold models and training-only backgrounds'),indent=2))
    print(importance.head(6).to_string())
    print(pd.DataFrame(local_rows).to_string(index=False))

if __name__=='__main__':main()
