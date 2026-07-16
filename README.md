Authors: Dr. John Edison, Kit Ao, Panwa Promtep, Kellen Roddy
Date: April 2024
Johns Hopkins University

This codebase is designed to supplement the instruction of ASPEN Plus Software.
ASPEN Plus, while a very powerful tool in chemical process simulations, has a reputation for being outdated, prone to crashing, and unhelpful when user-entered inputs do not produce a valid design.
These Examples show why different parameters are not feasible, and can provide students learning ASPEN with a direction for which values to change after a simulation failed to converge.

## Interactive demos

The six Example notebooks run in the browser, with live sliders and no install:

https://dayakaran.github.io/Ext_Distill_Sim/lab/index.html?path=Example1.ipynb

Swap the `path=` for Example2 through Example6. The first load takes 20 to 30
seconds while Pyodide fetches numpy, scipy and matplotlib. It is cached after that.

## Running locally

```
pip install -e .
jupyter lab Example1.ipynb
```

## Rebuilding the hosted site

```
pip install -r requirements-jupyterlite.txt
python tools/build_jupyterlite.py --output _site
python -m http.server -d _site
```

The notebooks at the repo root are the source of truth. `tools/build_jupyterlite.py`
patches copies for the browser (disables `text.usetex`, which needs a LaTeX binary
Pyodide does not have, and installs the library as a wheel), so never edit the
notebooks inside the built site or from within JupyterLite: those changes live in
browser storage and are overwritten on the next build.

Pushing to `main` rebuilds and republishes the site via GitHub Actions.

## Model equations

The double-feed extractive column follows Knapp and Doherty (1994), "Minimum
Entrainer Flows for Extractive Distillation: a Bifurcation Theoretic Approach",
AIChE J 40(2):243, doi:10.1002/aic.690400206. Equation numbers cited in
`DistillationDoubleFeed.py` refer to that paper. The single-feed fixed-point and
residue-curve machinery follows Fidkowski, Malone and Doherty (1991),
doi:10.1002/aic.690371202.
