# When Rotation Curves Can't Decide

Code for a Bayesian comparison of dark-matter and modified-gravity models on the 175 rotation curves of the SPARC database (Lelli, McGaugh & Schombert 2016).

**Ranked by BIC:** NFW, Burkert, pseudo-isothermal, DC14, MOND (fixed a0), ToyBar
**Reported separately:** MONDfree (a0 free per galaxy), ToyEmp, MONDrar

## Install
    pip install -r requirements.txt

## Run
    python sparc_model_comparison.py all --data-dir data --jobs 4

SPARC data download automatically into `data/` on first run. Stages can also be run one at a time: `run` (MCMC fits), `aggregate` (BIC tables), `btfr` (Tully-Fisher fits), `figures` (paper figures). A full run takes several hours; fits resume if interrupted.

## Outputs (in `data/`)
- `final_results_v2.csv`: all fitted parameters, chi2, BIC and convergence per galaxy and model
- `bic_by_galaxy_v2.csv`: BIC and ΔBIC per galaxy, lowest-BIC model, clear winner or tie
- `bic_summary_v2.csv`: per-model wins, median ΔBIC, median chi2/dof, summed BIC
- `extra_models_v2.csv`: MONDfree / ToyEmp / MONDrar compared with the ranked models
- `btfr_summary_v2.csv`: BTFR slopes and scatter for three quality subsets

## Credits
Author: Aarav Kumar. Mentor: Geetanjali Sethi, who restructured the pipeline into this version.
Parts of the code were written with AI assistance (Claude, Anthropic; ChatGPT, OpenAI), including the MONDrar model, `report_extra_models()` and the figure code.
Data: SPARC, http://astroweb.cwru.edu/SPARC/
