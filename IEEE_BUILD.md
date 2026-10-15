Build the IEEE conference manuscript from this directory:

```sh
latexmk -pdf -outdir=ieee_build -interaction=nonstopmode -halt-on-error main.tex
cp ieee_build/main.pdf main.pdf
```

IEEEtran.cls and IEEEtranN.bst are included from the official CTAN IEEEtran distribution. The latter supports the manuscript's narrative author citations through natbib while formatting a numbered IEEE bibliography. The previous manuscript is retained in archive/ieee_revision_2026-10-09/.

The average confusion figure uses results/lopo/metrics.csv (seed 42, five fixed predictive families; dummy excluded). Rebuild it with an environment containing Matplotlib:

```sh
python scripts/build_average_confusion.py
```

The average is across model families on the same 319 records, not repeated runs. The original Experimental LOPO ROC/confusion image is reproduced without modification. XAI figures explain the separate Optuna-tuned logistic regression outer-fold pipelines, as documented in figures/xai/protocol.json.
