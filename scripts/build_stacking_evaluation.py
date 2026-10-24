"""Compare held-out logistic regression and Stacking discrimination and errors.

The FNR/specificity view rotates the usual ROC coordinates by 180 degrees:
(FPR, TPR) -> (1 - TPR, 1 - FPR). Its area remains ROC AUC.
Confusion-matrix annotations use Logistic regression / Stacking order.
"""
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/codex-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'Logistic regression': ('dynamic_individual_v4', 'logistic'),
    'Stacking': ('stacking_v4', 'stacking'),
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    matrices = []
    labels = None
    for name, (directory, model) in SOURCES.items():
        frame = pd.read_csv(ROOT / 'results' / directory / 'predictions.csv')
        frame = frame.loc[frame.model.eq(model)].set_index('source_row').sort_index()
        assert len(frame) == 319 and frame.index.is_unique
        if labels is None:
            labels = frame.actual_present
        else:
            pd.testing.assert_series_equal(labels, frame.actual_present)
        fpr, tpr, _ = roc_curve(frame.actual_present, frame.probability_present)
        fnr, specificity = 1 - tpr, 1 - fpr
        score = roc_auc_score(frame.actual_present, frame.probability_present)
        np.testing.assert_allclose(auc(fnr, specificity), score, atol=1e-12)
        axes[0].plot(fnr[::-1], specificity[::-1], lw=1.8,
                     label=f'{name} (AUC {score:.4f})')
        matrices.append(confusion_matrix(frame.actual_present,
                                        frame.probability_present.ge(.5), labels=[0, 1]))
    axes[0].plot([0, 1], [0, 1], '--', color='grey', lw=.8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='False-negative rate (FN / positives)',
                ylabel='Specificity (TN / negatives)', title='Held-out discrimination')
    axes[0].legend(loc='lower right', fontsize=8, frameon=False)
    axes[0].set_aspect('equal', adjustable='box')

    # One heatmap, with colour representing the mean count of the two models.
    axes[1].imshow(np.mean(matrices, axis=0), cmap='Blues', vmin=0,
                   vmax=max(m.max() for m in matrices))
    for i in range(2):
        for j in range(2):
            values = [int(matrix[i, j]) for matrix in matrices]
            axes[1].text(j, i, f'{values[0]} / {values[1]}', ha='center', va='center',
                         fontsize=13, color='white' if np.mean(values) > 75 else 'black')
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Absent', 'Present'],
                yticklabels=['Absent', 'Present'], xlabel='Predicted label', ylabel='True label',
                title='Counts: Logistic regression / Stacking')
    axes[1].title.set_fontsize(9)
    fig.tight_layout(w_pad=2)
    for suffix in ('pdf', 'png'):
        fig.savefig(ROOT / f'figures/evaluation/stacking_evaluation.{suffix}',
                    dpi=240, bbox_inches='tight')
    plt.close(fig)
    print('Verified matching 319 exclusions and equivalent ROC areas.')
    print('Confusion counts (Logistic regression / Stacking):')
    print(np.stack(matrices, axis=-1))


if __name__ == '__main__':
    main()
