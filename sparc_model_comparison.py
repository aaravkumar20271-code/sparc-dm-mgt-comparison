"""
sparc_model_comparison.py

Bayesian comparison of dark-matter halo, MOND and toy rotation-curve models on the
175 SPARC galaxies (Lelli, McGaugh & Schombert 2016).

Models
------
Ranked (BIC):            NFW, Burkert, pseudo-isothermal (ISO), DC14, MOND (fixed a0), ToyBar
Reported separately:     MONDfree (a0 free), ToyEmp (toy term only, no baryons), MONDrar (RAR function)

Every model with baryons shares V_bar^2 = V_gas|V_gas| + U V_disk|V_disk| + 1.4 U V_bul|V_bul|,
with a Gaussian prior U ~ N(0.5, 0.1) on 0 < U < 2.5. Fits: multi-start L-BFGS-B, then emcee
(48 walkers, 5000 steps, 20% burn-in), fixed per-galaxy seeds. Comparison: BIC = chi2_best + k ln N.

Stages
------
  run        MCMC fits for every galaxy -> data/results/<galaxy>.json (resumes if interrupted)
  aggregate  tables: final_results_v2.csv, bic_by_galaxy_v2.csv, bic_summary_v2.csv,
             wins_by_hubble_type_v2.csv, extra_models_v2.csv
  btfr       baryonic Tully-Fisher fits -> btfr_summary_v2.csv
  figures    paper figures -> figures/*.png and *.pdf (needs run + aggregate first)
  all        run + aggregate + btfr + figures

Usage
-----
  python sparc_model_comparison.py all --data-dir data --jobs 4
  python sparc_model_comparison.py run --data-dir data --galaxies NGC2403 DDO154 --steps 2000   # quick test
  python sparc_model_comparison.py figures --data-dir data --out figures

SPARC data are downloaded automatically into --data-dir on first run
(http://astroweb.cwru.edu/SPARC/).

Requirements: numpy, scipy, pandas, matplotlib, emcee, requests

Credits
-------
Author: Aarav Kumar. Mentor: Geetanjali Sethi (restructured the pipeline into this v2 version).
Parts of this code were written with AI assistance (Claude, Anthropic), including the MONDrar
model, report_extra_models(), and the figure code; see the project's AI-use disclosure.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, json, time, zipfile, zlib, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import hyp2f1

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ----------------------------------------------------------------------------- constants
G = 4.3009e-6                 # kpc (km/s)^2 / Msun
A0 = 3702.8                   # MOND a0 in (km/s)^2/kpc  (1.2e-10 m/s^2)
DELTA = 92.8                  # virial overdensity (DC14 / WMAP3 convention)
HUBBLE_0 = 0.073              # km/s/kpc
UPS_MEAN, UPS_STD = 0.5, 0.1  # Gaussian prior on disk mass-to-light ratio (3.6 micron)

SPARC_ROT_URL = "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
SPARC_TABLE_URL = "https://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt"
HUBBLE_NAMES = {0: "S0", 1: "Sa", 2: "Sab", 3: "Sb", 4: "Sbc", 5: "Sc", 6: "Scd",
                7: "Sd", 8: "Sdm", 9: "Sm", 10: "Im", 11: "BCD"}
TABLE_COLS = ["Galaxy", "HubbleT", "Dist", "e_Dist", "f_D", "Inclination", "e_Inclination", "L36", "e_L36",
              "Reff", "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Quality", "Ref"]


# ----------------------------------------------------------------------------- data loading
def download(url, dest):
    import requests
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def ensure_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    rot_dir = os.path.join(data_dir, "sparc files")
    if not os.path.isdir(rot_dir) or not any(f.endswith("_rotmod.dat") for f in os.listdir(rot_dir)):
        zp = os.path.join(data_dir, "Rotmod_LTG.zip")
        download(SPARC_ROT_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(rot_dir)
    table = os.path.join(data_dir, "SPARC_Lelli2016c.mrt")
    if not os.path.exists(table):
        download(SPARC_TABLE_URL, table)
    return rot_dir, table


def load_master_table(path):
    """Parse the SPARC Table 1 by whitespace: data rows have exactly 19 fields and an integer Hubble type."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.split()
            if len(t) == 19 and t[1].lstrip("-").isdigit():
                rows.append(t)
    df = pd.DataFrame(rows, columns=TABLE_COLS)
    for c in TABLE_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Galaxy"] = df["Galaxy"].str.strip()
    df["HubbleT"] = df["HubbleT"].astype(int)
    df["HubbleType"] = df["HubbleT"].map(HUBBLE_NAMES)
    df["M_bar"] = (0.5 * df["L36"] + 1.33 * df["MHI"]) * 1e9        # Msun, Upsilon_disk = 0.5
    df["log_M_bar"] = np.log10(df["M_bar"])
    return df


def load_galaxy(rot_dir, galaxy, master):
    g = pd.read_csv(os.path.join(rot_dir, galaxy + "_rotmod.dat"), sep=r"\s+", header=None, comment="#")
    g.columns = ["R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    info = master.loc[master["Galaxy"] == galaxy]
    if len(info) == 0:
        return None
    info = info.iloc[0]
    return dict(galaxy=galaxy, r=g["R"].values, V_obs=g["Vobs"].values, V_err=g["e_Vobs"].values,
                V_gas=g["Vgas"].values, V_disk=g["Vdisk"].values, V_bul=g["Vbul"].values,
                R_min=float(g["R"].min()), R_max=float(g["R"].max()), N=len(g),
                L36=float(info["L36"]), L36_solar=float(info["L36"]) * 1e9,
                HubbleT=int(info["HubbleT"]), HubbleType=info["HubbleType"], Quality=int(info["Quality"]))


# ----------------------------------------------------------------------------- physics
def baryon_velocity_sq(V_gas, V_disk, V_bul, ups):
    return V_gas * np.abs(V_gas) + ups * V_disk * np.abs(V_disk) + 1.4 * ups * V_bul * np.abs(V_bul)


def _bar(d, ups):
    return baryon_velocity_sq(d["V_gas"], d["V_disk"], d["V_bul"], ups)


def _total(halo_sq, bar_sq):
    v2 = halo_sq + bar_sq
    if not np.all(np.isfinite(v2)) or np.any(v2 <= 0):
        return None
    return np.sqrt(v2)


def v_nfw_sq(rho_s, r_s, r):
    return G * 4 * np.pi * rho_s * r_s ** 3 * (np.log(1 + r / r_s) - (r / r_s) / (1 + r / r_s)) / r


def v_burkert_sq(rho0, r0, r):
    x = r / r0
    return G * np.pi * rho0 * r0 ** 3 * (2 * np.log(1 + x) + np.log(1 + x ** 2) - 2 * np.arctan(x)) / r


def v_iso_sq(rho0, rc, r):
    x = r / rc
    return G * 4 * np.pi * rho0 * rc ** 3 * (x - np.arctan(x)) / r


def velocity_mond(V_bar_sq, r, a0=None):
    a0 = A0 if a0 is None else a0
    g_n = V_bar_sq / r
    sign = np.sign(g_n)
    g_abs = np.where(np.abs(g_n) == 0, 1e-30, np.abs(g_n))
    g_m = (g_abs / 2) * (1 + np.sqrt(1 + 4 * a0 / g_abs))      # "simple" interpolating function
    return np.sqrt(g_m * r) * sign


def velocity_mond_rar(V_bar_sq, r, a0=None):
    """RAR / 'exponential' interpolating function (McGaugh et al. 2016; Li et al. 2018):
    g = g_N / (1 - exp(-sqrt(g_N / a0))).  expm1 keeps it finite when g_N is tiny."""
    a0 = A0 if a0 is None else a0
    g_n = V_bar_sq / r
    sign = np.sign(g_n)
    g_abs = np.where(np.abs(g_n) == 0, 1e-30, np.abs(g_n))
    g_m = g_abs / (-np.expm1(-np.sqrt(g_abs / a0)))
    return np.sqrt(g_m * r) * sign


# Di Cintio et al. 2014
def di_cintio_greeks(x):
    if x <= -4.1:
        return 1.0, 3.0, 1.0
    if x >= -1.3:
        return None
    alpha = 2.94 - np.log10((10 ** (x + 2.33)) ** -1.08 + (10 ** (x + 2.33)) ** 2.29)
    beta = 4.23 + 1.34 * x + 0.26 * x ** 2
    gamma = -0.06 + np.log10((10 ** (x + 2.56)) ** -0.68 + (10 ** (x + 2.56)))
    return alpha, beta, gamma


def dc14_mass_factor(x, alpha, beta, gamma):
    A = (3 - gamma) / alpha
    B = (beta - gamma) / alpha
    return (x ** (3 - gamma)) / (3 - gamma) * hyp2f1(A, B, A + 1, -(x ** alpha))


def r_virial(M_halo):
    rho_crit = 3 * HUBBLE_0 ** 2 / (8 * np.pi * G)
    return (3 * M_halo / (4 * np.pi * DELTA * rho_crit)) ** (1 / 3)


def dc14_params(params, L36_solar):
    log_m, c, ups = params[0], params[1], params[2]
    m_halo = 10 ** log_m
    m_star = ups * L36_solar
    log_ratio = np.log10(m_star / m_halo)
    greeks = di_cintio_greeks(log_ratio)
    if greeks is None:
        return None
    alpha, beta, gamma = greeks
    r_vir = r_virial(m_halo)
    r_s = (r_vir / c) / (((2 - gamma) / (beta - 2)) ** (1 / alpha))
    norm = 4 * np.pi * r_s ** 3 * dc14_mass_factor(r_vir / r_s, alpha, beta, gamma)
    return dict(M_halo=m_halo, M_star=m_star, log_ratio=log_ratio, alpha=alpha, beta=beta, gamma=gamma,
                r_vir=r_vir, r_s=r_s, rho_s=m_halo / norm)


# ----------------------------------------------------------------------------- model specs
# Every model returns the predicted circular velocity or None if the parameters are unphysical.
def m_nfw(p, d):
    return _total(v_nfw_sq(10 ** p[0], 10 ** p[1], d["r"]), _bar(d, p[2]))


def m_burkert(p, d):
    return _total(v_burkert_sq(10 ** p[0], 10 ** p[1], d["r"]), _bar(d, p[2]))


def m_iso(p, d):
    return _total(v_iso_sq(10 ** p[0], 10 ** p[1], d["r"]), _bar(d, p[2]))


def m_mond_free(p, d):
    v = velocity_mond(_bar(d, p[1]), d["r"], 10 ** p[0])
    return v if np.all(np.isfinite(v)) else None


def m_mond(p, d):
    v = velocity_mond(_bar(d, p[0]), d["r"])
    return v if np.all(np.isfinite(v)) else None


def m_mond_rar(p, d):
    v = velocity_mond_rar(_bar(d, p[0]), d["r"])
    return v if np.all(np.isfinite(v)) else None


def m_dc14(p, d):
    der = dc14_params(p, d["L36_solar"])
    if der is None:
        return None
    enc = 4 * np.pi * der["rho_s"] * der["r_s"] ** 3 * dc14_mass_factor(d["r"] / der["r_s"], der["alpha"], der["beta"], der["gamma"])
    return _total(G * enc / d["r"], _bar(d, p[2]))


def toy_lambda(v2_ext, r_ratio, R_max):
    r_lam = r_ratio * R_max
    return v2_ext * (R_max + r_lam) / R_max, r_lam


def m_toy_emp(p, d):
    lam, r_lam = toy_lambda(p[0], p[1], d["R_max"])
    return _total(lam * d["r"] / (d["r"] + r_lam), 0.0)


def m_toy_bar(p, d):
    lam, r_lam = toy_lambda(p[0], p[1], d["R_max"])
    return _total(lam * d["r"] / (d["r"] + r_lam), _bar(d, p[2]))


def extra_toy(p, d):
    lam, r_lam = toy_lambda(p[0], p[1], d["R_max"])
    return (max(0.1, d["R_min"] / 2) < r_lam < 10 * d["R_max"]) and (0 < lam < 1e6)


def extra_dc14(p, d):
    return np.log10(p[2] * d["L36_solar"] / 10 ** p[0]) < -1.3


def u(rng, lo, hi):
    return rng.uniform(lo, hi)


# name: dict(k, names, bounds, ups_idx, model, extra, guess, rand_start, scales)
MODELS = {
    "NFW": dict(names=["log_rho_s", "log_r_s", "upsilon"], bounds=[(3, 12), (-1, 2), (0, 2.5)], ups_idx=2,
                model=m_nfw, extra=None, guess=lambda d: [7.0, 1.0, 0.5],
                rand=lambda rng, d: [u(rng, 4, 10), u(rng, -0.5, 1.5), u(rng, .2, .8)], scales=lambda s: [.10, .05, .03]),
    "Burkert": dict(names=["log_rho0", "log_r0", "upsilon"], bounds=[(3, 12), (-1, 2), (0, 2.5)], ups_idx=2,
                    model=m_burkert, extra=None, guess=lambda d: [7.5, 0.5, 0.5],
                    rand=lambda rng, d: [u(rng, 4, 10), u(rng, -0.5, 1.5), u(rng, .2, .8)], scales=lambda s: [.10, .05, .03]),
    "ISO": dict(names=["log_rho0", "log_rc", "upsilon"], bounds=[(3, 12), (-1, 2), (0, 2.5)], ups_idx=2,
                model=m_iso, extra=None, guess=lambda d: [7.5, 0.5, 0.5],
                rand=lambda rng, d: [u(rng, 4, 10), u(rng, -0.5, 1.5), u(rng, .2, .8)], scales=lambda s: [.10, .05, .03]),
    "MONDfree": dict(names=["log_a0", "upsilon"], bounds=[(2.0, 4.7), (0, 2.5)], ups_idx=1, model=m_mond_free, extra=None,
                     guess=lambda d: [np.log10(A0), 0.5], rand=lambda rng, d: [u(rng, 3.0, 4.2), u(rng, .2, 1.0)],
                     scales=lambda s: [.05, .03]),
    "MOND": dict(names=["upsilon"], bounds=[(0, 2.5)], ups_idx=0, model=m_mond, extra=None,
                 guess=lambda d: [0.5], rand=lambda rng, d: [u(rng, .2, 1.0)], scales=lambda s: [.03]),
    "MONDrar": dict(names=["upsilon"], bounds=[(0, 2.5)], ups_idx=0, model=m_mond_rar, extra=None,
                    guess=lambda d: [0.5], rand=lambda rng, d: [u(rng, .2, 1.0)], scales=lambda s: [.03]),
    "DC14": dict(names=["log_Mhalo", "c", "upsilon"], bounds=[(9, 13), (1, 40), (0, 2.5)], ups_idx=2,
                 model=m_dc14, extra=extra_dc14,
                 guess=lambda d: [float(np.clip(max(11.0, np.log10(.5 * d["L36_solar"]) + 1.5), 9.2, 12.8)), 10.0, 0.5],
                 rand=lambda rng, d: [u(rng, 10, 12.5), u(rng, 3, 30), u(rng, .2, .8)], scales=lambda s: [.10, .5, .03]),
    "ToyEmp": dict(names=["v2_ext", "r_ratio"], bounds=[(1.0, 1e6), (1e-4, 10)], ups_idx=None,
                   model=m_toy_emp, extra=extra_toy, guess=lambda d: [d["V_obs"][-1] ** 2, 0.5],
                   rand=lambda rng, d: [d["V_obs"][-1] ** 2 * 10 ** u(rng, -.3, .3), 10 ** u(rng, -2, 1)],
                   scales=lambda s: [s[0] * 0.05, 0.05]),
    "ToyBar": dict(names=["v2_ext", "r_ratio", "upsilon"], bounds=[(1.0, 1e6), (1e-4, 10), (0, 2.5)], ups_idx=2,
                   model=m_toy_bar, extra=extra_toy,
                   guess=lambda d: [max(d["V_obs"][-1] ** 2 - _bar(d, .5)[-1], 100.0), 0.5, 0.5],
                   rand=lambda rng, d: [max(d["V_obs"][-1] ** 2 - _bar(d, .5)[-1], 100.0) * 10 ** u(rng, -.3, .3),
                                        10 ** u(rng, -2, 1), u(rng, .2, .8)],
                   scales=lambda s: [s[0] * 0.05, 0.05, 0.03]),
}
for _n, _m in MODELS.items():
    _m["k"] = len(_m["names"])


# ----------------------------------------------------------------------------- statistics
def chi2(spec, p, d):
    v = spec["model"](p, d)
    if v is None or not np.all(np.isfinite(v)):
        return np.inf
    return float(np.sum(((d["V_obs"] - v) / d["V_err"]) ** 2))


def log_prior(spec, p, d):
    for x, (lo, hi) in zip(p, spec["bounds"]):
        if not (lo < x < hi):
            return -np.inf
    if spec["extra"] is not None and not spec["extra"](p, d):
        return -np.inf
    if spec["ups_idx"] is not None:
        return -0.5 * ((p[spec["ups_idx"]] - UPS_MEAN) / UPS_STD) ** 2
    return 0.0


def log_post(p, spec, d):
    lp = log_prior(spec, p, d)
    if not np.isfinite(lp):
        return -np.inf
    c = chi2(spec, p, d)
    return -np.inf if not np.isfinite(c) else lp - 0.5 * c


def neg_log_post(p, spec, d):
    lp = log_post(p, spec, d)
    return 1e30 if not np.isfinite(lp) else -lp


def find_map(spec, d, rng, n_starts=6):
    """Multi-start L-BFGS-B on the posterior (helps with the bimodal DC14 surface)."""
    starts = [np.array(spec["guess"](d), float)] + [np.array(spec["rand"](rng, d), float) for _ in range(n_starts - 1)]
    best = None
    for x0 in starts:
        try:
            res = minimize(neg_log_post, x0, args=(spec, d), bounds=spec["bounds"], method="L-BFGS-B")
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < 1e29 and (best is None or res.fun < best.fun):
            best = res
    return None if best is None else best.x


def init_walkers(center, scales, n, spec, d, rng):
    center, scales = np.asarray(center), np.asarray(scales)
    out, tries, f = [], 0, 1.0
    while len(out) < n:
        cand = center + f * scales * rng.normal(size=len(center))
        if np.isfinite(log_post(cand, spec, d)):
            out.append(cand)
        tries += 1
        if tries % 1000 == 0:
            f *= 0.5                       # shrink the ball if the prior edge keeps rejecting
        if tries > 20000:
            out.append(center.copy() + 1e-6 * rng.normal(size=len(center)))
    return np.array(out)


def run_model(name, d, steps, walkers, rng, seed, save_chain_dir=None):
    import emcee
    spec = MODELS[name]
    t0 = time.time()
    p_map = find_map(spec, d, rng)
    if p_map is None:
        return dict(failed=True)
    p0 = init_walkers(p_map, spec["scales"](p_map), walkers, spec, d, rng)
    sampler = emcee.EnsembleSampler(walkers, spec["k"], log_post, args=(spec, d))
    try:
        sampler.random_state = np.random.RandomState(seed)
    except Exception:
        np.random.seed(seed)
    sampler.run_mcmc(p0, steps, progress=False)
    burn = steps // 5
    flat = sampler.get_chain(discard=burn, flat=True)
    lp = sampler.get_log_prob(discard=burn, flat=True)
    q = np.percentile(flat, [16, 50, 84], axis=0)
    med = q[1]
    p_best = flat[np.argmax(lp)]
    if log_post(p_map, spec, d) >= log_post(p_best, spec, d):
        p_best = p_map
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.nanmax(tau))
        converged = bool(np.all(steps > 50 * tau))
    except Exception:
        tau_max, converged = np.nan, False
    c_best, c_med = chi2(spec, p_best, d), chi2(spec, med, d)
    N, k = d["N"], spec["k"]
    out = dict(failed=False, chi2_best=c_best, chi2_median=c_med, k=k, N=N,
               bic=c_best + k * np.log(N), converged=converged, tau_max=tau_max,
               acceptance=float(np.mean(sampler.acceptance_fraction)), runtime_s=time.time() - t0)
    for i, pn in enumerate(spec["names"]):
        out[f"{pn}_median"], out[f"{pn}_low"], out[f"{pn}_high"], out[f"{pn}_best"] = float(q[1, i]), float(q[0, i]), float(q[2, i]), float(p_best[i])
    # model-specific derived quantities (at the median, as in the earlier tables)
    if name == "DC14":
        der = dc14_params(med, d["L36_solar"])
        if der:
            out.update({f"DC14_{k_}": float(v) for k_, v in der.items()})
    if name.startswith("Toy"):
        lam, r_lam = toy_lambda(med[0], med[1], d["R_max"])
        v = spec["model"](p_best, d)
        out.update(lam_median=float(lam), r_lambda_median=float(r_lam),
                   V_Rmax_best=float(v[-1]) if v is not None else np.nan)
    if save_chain_dir:
        np.save(os.path.join(save_chain_dir, f"{d['galaxy']}_{name}.npy"), flat[:: max(1, len(flat) // 5000)])
    return out


def fit_galaxy(args):
    galaxy, rot_dir, table, steps, walkers, outdir, models, save_chains = args
    out_path = os.path.join(outdir, "results", f"{galaxy}.json")
    if os.path.exists(out_path):
        return galaxy, "cached"
    master = load_master_table(table)
    d = load_galaxy(rot_dir, galaxy, master)
    if d is None:
        return galaxy, "no metadata"
    row = dict(Galaxy=galaxy, Datapoints=d["N"], L36=d["L36"], HubbleT=d["HubbleT"], HubbleType=d["HubbleType"],
               Quality=d["Quality"], R_min=d["R_min"], R_max=d["R_max"])
    chain_dir = None
    if save_chains:
        chain_dir = os.path.join(outdir, "chains")
        os.makedirs(chain_dir, exist_ok=True)
    for mi, name in enumerate(models):
        seed = zlib.crc32(galaxy.encode()) % (2 ** 31) + 1000 * mi
        rng = np.random.default_rng(seed)
        try:
            res = run_model(name, d, steps, walkers, rng, seed, chain_dir)
        except Exception as e:           # a failing model must not kill the galaxy
            res = dict(failed=True, error=str(e))
        for key, val in res.items():
            row[f"{name}_{key}"] = val
    with open(out_path, "w") as f:
        json.dump(row, f, default=lambda o: None if o is None else float(o))
    return galaxy, "done"


# ----------------------------------------------------------------------------- stage: run
def stage_run(a):
    rot_dir, table = ensure_data(a.data_dir)
    os.makedirs(os.path.join(a.data_dir, "results"), exist_ok=True)
    master = load_master_table(table)
    avail = sorted(f.replace("_rotmod.dat", "") for f in os.listdir(rot_dir) if f.endswith("_rotmod.dat"))
    gals = a.galaxies if a.galaxies else [g for g in avail if g in set(master["Galaxy"])]
    models = a.models.split(",")
    jobs = [(g, rot_dir, table, a.steps, a.walkers, a.data_dir, models, a.save_chains) for g in gals]
    print(f"{len(jobs)} galaxies, models={models}, steps={a.steps}, walkers={a.walkers}, jobs={a.jobs}", flush=True)
    t0 = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(fit_galaxy, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(jobs)}] {f.result()}  {time.time() - t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {fit_galaxy(j)}  {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- stage: aggregate
def load_results(data_dir):
    rows = []
    rdir = os.path.join(data_dir, "results")
    for f in sorted(os.listdir(rdir)):
        if f.endswith(".json"):
            with open(os.path.join(rdir, f)) as fh:
                rows.append(json.load(fh))
    return pd.DataFrame(rows)


def stage_aggregate(a):
    df = load_results(a.data_dir)
    df.to_csv(os.path.join(a.data_dir, "final_results_v2.csv"), index=False)
    models = a.bic_models.split(",")
    ok = np.ones(len(df), bool)
    for m in models:
        ok &= df[f"{m}_bic"].notna().values & np.isfinite(df[f"{m}_bic"].astype(float).values)
        if f"{m}_failed" in df:
            ok &= ~df[f"{m}_failed"].fillna(True).astype(bool).values
    d = df.loc[ok].copy()
    print(f"{len(df)} galaxies fitted; {len(d)} have a finite BIC for all of {models}")
    bic = d[[f"{m}_bic" for m in models]].astype(float)
    bic.columns = models
    dbic = bic.sub(bic.min(axis=1), axis=0)
    srt = np.sort(bic.values, axis=1)
    gap = srt[:, 1] - srt[:, 0]
    best = bic.idxmin(axis=1)
    per = pd.DataFrame({"Galaxy": d["Galaxy"].values, "HubbleType": d["HubbleType"].values,
                        "Datapoints": d["Datapoints"].values})
    for m in models:
        per[f"chi2_{m}"] = d[f"{m}_chi2_best"].values
        per[f"BIC_{m}"] = bic[m].values
        per[f"dBIC_{m}"] = dbic[m].values
    per["lowest_BIC"] = best.values
    per["clear_winner"] = np.where(gap > 2, best.values, "tie")
    per["gap_to_second"] = gap
    per.to_csv(os.path.join(a.data_dir, "bic_by_galaxy_v2.csv"), index=False)

    summ = []
    for m in models:
        n_par = MODELS[m]["k"]
        summ.append(dict(Model=m, k=n_par,
                         lowest_BIC=int((best == m).sum()), clear_wins=int((per["clear_winner"] == m).sum()),
                         within_2=int((dbic[m] < 2).sum()), worse_than_10=int((dbic[m] > 10).sum()),
                         median_dBIC=float(dbic[m].median()),
                         median_chi2_dof=float(np.median(d[f"{m}_chi2_best"] / (d["Datapoints"] - n_par))),
                         summed_BIC=float(bic[m].sum()),
                         converged=int(d[f"{m}_converged"].fillna(False).astype(bool).sum())))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(a.data_dir, "bic_summary_v2.csv"), index=False)
    print(s.to_string(index=False))
    print("ties (top two within dBIC 2):", int((per["clear_winner"] == "tie").sum()))
    # median-vs-best chi2 check: shows how much the old 'chi2 at median' inflated BIC
    chk = {m: float((d[f"{m}_chi2_median"] - d[f"{m}_chi2_best"]).median()) for m in models}
    print("median(chi2_at_median - chi2_best):", {k: round(v, 2) for k, v in chk.items()})
    # wins by Hubble type
    wt = pd.crosstab(per["HubbleType"], per["clear_winner"])
    wt.to_csv(os.path.join(a.data_dir, "wins_by_hubble_type_v2.csv"))
    # toy classes (r_lambda relative to the data range)
    for tm in ("ToyEmp", "ToyBar"):
        if f"{tm}_r_lambda_median" in df:
            rl, rmin, rmax = df[f"{tm}_r_lambda_median"], df["R_min"], df["R_max"]
            cls = np.where(rl < rmin, "saturated", np.where(rl <= rmax, "real transition", "degenerate"))
            print(tm, "classes:", dict(pd.Series(cls).value_counts()))
    report_extra_models(a, d, bic, models)


def report_extra_models(a, d, bic, models):
    """Separately-reported variants (e.g. MONDrar, MONDfree, ToyEmp): fitted, but NOT in the main ranking.
    For each: median chi2/dof, how it does against the best of the ranked models, and (for MOND variants)
    against fixed-a0 simple-function MOND.  Uses the same galaxies as the main comparison."""
    best_ranked = bic.min(axis=1)
    rows = []
    for m in [x for x in MODELS if x not in models and f"{x}_bic" in d.columns]:
        bm = d[f"{m}_bic"].astype(float)
        ok = np.isfinite(bm.values)
        if not ok.any():
            continue
        k = MODELS[m]["k"]
        diff = (bm - best_ranked)[ok]
        row = dict(Model=m, k=k, galaxies=int(ok.sum()),
                   median_chi2_dof=float(np.median((d[f"{m}_chi2_best"].astype(float) / (d["Datapoints"] - k))[ok])),
                   better_than_best_ranked=int((diff < -2).sum()), worse_than_best_ranked=int((diff > 2).sum()),
                   tie_with_best_ranked=int((diff.abs() <= 2).sum()),
                   converged=int(d[f"{m}_converged"].fillna(False).astype(bool)[ok].sum()))
        if m.startswith("MOND") and m != "MOND" and "MOND" in bic.columns:
            dm = (bm - bic["MOND"])[ok]
            row.update(better_than_MOND=int((dm < -2).sum()), worse_than_MOND=int((dm > 2).sum()),
                       tie_with_MOND=int((dm.abs() <= 2).sum()))
        if f"{m}_upsilon_median" in d.columns:
            ups = d[f"{m}_upsilon_median"].astype(float)[ok]
            row.update(upsilon_median=float(ups.median()),
                       frac_upsilon_outside_0p3_0p7=float(((ups < 0.3) | (ups > 0.7)).mean()))
        if f"{m}_log_a0_median" in d.columns:
            a0 = 10 ** d[f"{m}_log_a0_median"].astype(float)[ok]
            row.update(a0_median=float(a0.median()), a0_p16=float(a0.quantile(.16)), a0_p84=float(a0.quantile(.84)))
        rows.append(row)
    if rows:
        ex = pd.DataFrame(rows)
        ex.to_csv(os.path.join(a.data_dir, "extra_models_v2.csv"), index=False)
        print("\nSeparately reported variants (not in the main ranking):")
        print(ex.to_string(index=False))


# ----------------------------------------------------------------------------- stage: BTFR
def tls_fit(x, y, sx, sy):
    """Straight line y=a*x+b with errors in both axes (effective-variance / orthogonal-type weighted fit)."""
    a0, b0 = np.polyfit(x, y, 1)
    def cost(p):
        a, b = p
        return np.sum((y - a * x - b) ** 2 / (sy ** 2 + a ** 2 * sx ** 2))
    r = minimize(cost, [a0, b0], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000))
    return r.x


def stage_btfr(a):
    rot_dir, table = ensure_data(a.data_dir)
    m = load_master_table(table)
    base = m[(m["Vflat"] > 0) & np.isfinite(m["log_M_bar"])]
    subsets = {"all with Vflat>0": base,
               "Q<3 & Inc>30": base[(base["Quality"] < 3) & (base["Inclination"] > 30)],
               "Q=1": base[base["Quality"] == 1]}
    rows = []
    for name, t in subsets.items():
        x, y = np.log10(t["Vflat"].values), t["log_M_bar"].values
        sx = (t["e_Vflat"].values / t["Vflat"].values / np.log(10))
        sx = np.where(np.isfinite(sx) & (sx > 0), sx, 0.02)
        sy = np.full_like(y, 0.1)                       # ASSUMED 0.1 dex uncertainty on log M_bar
        a_ols, b_ols = np.polyfit(x, y, 1)
        a_tls, b_tls = tls_fit(x, y, sx, sy)
        rows.append(dict(subset=name, n=len(t), slope_OLS=a_ols, slope_orthogonal=a_tls,
                         scatter_dex_OLS=float(np.std(y - a_ols * x - b_ols)),
                         scatter_dex_orthogonal=float(np.std(y - a_tls * x - b_tls))))
        if name == "all with Vflat>0":
            plt.figure(figsize=(6, 5))
            plt.scatter(x, y, s=12, c="C0")
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, a_ols * xs + b_ols, "r-", label=f"OLS slope {a_ols:.2f}")
            plt.plot(xs, a_tls * xs + b_tls, "g-", label=f"orthogonal slope {a_tls:.2f}")
            plt.plot(xs, 4 * xs + (y.mean() - 4 * x.mean()), "k--", label="MOND (slope 4)")
            plt.xlabel("log V_flat (km/s)"); plt.ylabel("log M_bar (Msun), observed")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(a.data_dir, "btfr_observed.png"), dpi=200); plt.close()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.data_dir, "btfr_summary_v2.csv"), index=False)
    print(out.to_string(index=False))


# ----------------------------------------------------------------------------- figures
STYLE = {  # model -> (label, colour, linestyle)
    "Burkert": ("Burkert", "#2a78d6", "-"),
    "ISO": ("Pseudo-isothermal", "#eb6834", "--"),
    "DC14": ("DC14", "#1baf7a", "-."),
    "ToyBar": ("Toy model (with baryons)", "#e87ba4", ":"),
    "NFW": ("NFW", "#6553d6", (0, (3, 1, 1, 1, 1, 1))),
    "MOND": ("MOND", "#e34b4b", (0, (1, 1.5))),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def ranked_models(summary):
    return [m for m in summary.sort_values("lowest_BIC", ascending=False)["Model"] if m in STYLE]


# ----------------------------------------------------------------------------- Figure 1
def fig1_example_fits(P, res, per, models, data_dir, out, galaxies):
    rot_dir = os.path.join(data_dir, "sparc files")
    master = P.load_master_table(os.path.join(data_dir, "SPARC_Lelli2016c.mrt"))
    avail = [g for g in galaxies if g in set(res["Galaxy"])]
    if len(avail) < 4:  # fall back to four galaxies spanning the range of data points
        pool = per.sort_values("Datapoints")["Galaxy"].tolist()
        idx = np.linspace(0, len(pool) - 1, 4).round().astype(int)
        avail = (avail + [pool[i] for i in idx if pool[i] not in avail])[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, g, tag in zip(axes.flat, avail, "abcd"):
        d = P.load_galaxy(rot_dir, g, master)
        row = res.loc[res["Galaxy"] == g].iloc[0]
        ax.errorbar(d["r"], d["V_obs"], yerr=d["V_err"], fmt="o", color="k", ms=4, lw=1, zorder=5)
        for m in models:
            names = P.MODELS[m]["names"]
            cols = [f"{m}_{n}_best" for n in names]
            if not all(c in row and np.isfinite(row[c]) for c in cols):
                continue
            v = P.MODELS[m]["model"]([row[c] for c in cols], d)
            if v is None:
                continue
            lab, col, ls = STYLE[m]
            ax.plot(d["r"], v, color=col, ls=ls, lw=1.6, label=lab)
        prow = per.loc[per["Galaxy"] == g]
        win = ""
        if len(prow):
            best, cw = prow["lowest_BIC"].iloc[0], prow["clear_winner"].iloc[0]
            win = f", lowest BIC: {STYLE.get(best, (best,))[0]}" + (" (tie)" if cw == "tie" else "")
        ax.set_title(f"({tag}) {g}, {d['N']} points{win}", fontsize=10, loc="left")
        ax.set_xlabel("Radius (kpc)"); ax.set_ylabel("Rotation velocity (km/s)")
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.3)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, out, "fig1_example_fits")


# ----------------------------------------------------------------------------- Figure 2
def fig2_wins(summary, per, models, out):
    s = summary.set_index("Model").loc[models]
    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for yi, m in zip(y, models):
        col = STYLE[m][1]
        ax.barh(yi + .2, s.loc[m, "lowest_BIC"], .36, color=col, alpha=.35)
        ax.barh(yi - .2, s.loc[m, "clear_wins"], .36, color=col)
        ax.text(s.loc[m, "lowest_BIC"] + .8, yi + .2, int(s.loc[m, "lowest_BIC"]), va="center")
        ax.text(s.loc[m, "clear_wins"] + .8, yi - .2, int(s.loc[m, "clear_wins"]), va="center")
    ax.set_yticks(y); ax.set_yticklabels([STYLE[m][0] for m in models])
    n, ties = len(per), int((per["clear_winner"] == "tie").sum())
    ax.set_xlabel(f"Number of galaxies (of {n})"); ax.grid(axis="x", alpha=.3)
    ax.text(.98, .3, f"{ties} of {n} galaxies are ties\n(top two models within 2 BIC units)",
            transform=ax.transAxes, ha="right", color="#555")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="grey", alpha=.35), Patch(color="grey")], ["Lowest BIC", "Clear winner (gap > 2)"],
              loc="lower right", frameon=False)
    save(fig, out, "fig2_wins")


# ----------------------------------------------------------------------------- Figure 3
def fig3_dbic(per, models, out):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(models):
        v = per[f"dBIC_{m}"].astype(float).dropna().values
        col = STYLE[m][1]
        ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=6, color=col, alpha=.45, lw=0)
        q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
        ax.add_patch(plt.Rectangle((i - .15, q25), .3, q75 - q25, fill=False, lw=1.2))
        ax.plot([i - .15, i + .15], [q50, q50], color="k", lw=3)
        ax.plot([i, i], [q10, q25], color="k", lw=1.2); ax.plot([i, i], [q75, q90], color="k", lw=1.2)
    ax.set_yscale("symlog", linthresh=2, linscale=1)
    for t, lab in ((2, "2: tie"), (10, "10: strongly worse")):
        ax.axhline(t, color="#444", ls="--", lw=1)
        ax.text(len(models) - .45, t * 1.15, lab, color="#444", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([STYLE[m][0].replace(" (", "\n(").replace("Pseudo-", "Pseudo-\n") for m in models])
    ax.set_ylabel("ΔBIC relative to the best model"); ax.set_ylim(bottom=-.1)
    ax.set_xlim(-.6, len(models) + .3); ax.grid(axis="y", alpha=.3)
    save(fig, out, "fig3_dbic")


# ----------------------------------------------------------------------------- Figure 4
def fig4_by_points(per, models, out):
    bins = [(4, 10, "4 to 10"), (11, 20, "11 to 20"), (21, 40, "21 to 40"), (41, 10 ** 6, "more than 40")]
    cats = ["tie"] + models
    colours = {"tie": "#d0cfca", **{m: STYLE[m][1] for m in models}}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ticklabels = []
    for j, (lo, hi, lab) in enumerate(bins):
        sub = per[(per["Datapoints"] >= lo) & (per["Datapoints"] <= hi)]
        n, bottom = len(sub), 0.0
        ticklabels.append(f"{lab}\n(n = {n})")
        if n == 0:
            continue
        for c in cats:
            share = 100 * (sub["clear_winner"] == c).sum() / n
            if share == 0:
                continue
            ax.bar(j, share, .62, bottom=bottom, color=colours[c])
            if share >= 8:
                ax.text(j, bottom + share / 2, f"{share:.0f}%", ha="center", va="center",
                        color="k" if c == "tie" else "white", fontsize=9)
            bottom += share
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(ticklabels)
    ax.set_ylim(0, 100); ax.set_ylabel("Share of galaxies (%)")
    ax.set_xlabel("Points on the rotation curve")
    hs = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in cats]
    ax.legend(hs, ["Tie"] + [STYLE[m][0] for m in models], frameon=False, loc="center left", bbox_to_anchor=(1, .5))
    save(fig, out, "fig4_by_points")


# ----------------------------------------------------------------------------- Figure 5
def fig5_toy(res, out, model="ToyBar"):
    need = [f"{model}_r_ratio_median", f"{model}_v2_ext_median", f"{model}_r_lambda_median", "R_min", "R_max"]
    if not all(c in res for c in need):
        print("skip fig5: no", model, "columns"); return
    t = res[need].astype(float).dropna()
    x, y = t[need[0]], np.sqrt(t[need[1]])
    rl = t[need[2]]
    cls = np.where(rl < t["R_min"], "Saturated", np.where(rl <= t["R_max"], "Transition", "Degenerate"))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for c, col in (("Degenerate", "#eb6834"), ("Transition", "#2a78d6"), ("Saturated", "#1baf7a")):
        s = cls == c
        ax.scatter(x[s], y[s], s=22, color=col, edgecolor="white", lw=.5, label=f"{c} (n = {int(s.sum())})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(1, color="#444", ls="--", lw=1)
    ax.text(1.08, ax.get_ylim()[0] * 1.2, "r_λ = outermost point", rotation=90, color="#444", fontsize=9, va="bottom")
    ax.set_xlabel(r"$r_\lambda / R_{\rm max}$ (fitted scale radius over outermost radius)")
    ax.set_ylabel(r"Extra velocity at $R_{\rm max}$ (km/s)")
    ax.legend(frameon=False, loc="lower left"); ax.grid(alpha=.3, which="major")
    save(fig, out, "fig5_toy")


# ----------------------------------------------------------------------------- Figure 6
def fig6_upsilon(res, out, P):
    vals = {m: res[f"{m}_upsilon_median"].astype(float).dropna() for m in ("MOND", "Burkert")
            if f"{m}_upsilon_median" in res}
    top = max([1.4] + [float(np.ceil(v.max() * 20) / 20) for v in vals.values() if len(v)])
    bins = np.arange(0, top + .05, .05)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(.3, .7, color="#d0cfca", alpha=.4, lw=0)
    n = 0
    for m, kw in (("MOND", dict(color="#e34b4b", alpha=.85, label="MOND")),
                  ("Burkert", dict(histtype="step", lw=2, color="#2a78d6", label="Burkert"))):
        if m in vals:
            n = max(n, len(vals[m]))
            ax.hist(vals[m], bins=bins, **kw)
    xs = np.linspace(0, top, 300)
    prior = n * .05 * np.exp(-.5 * ((xs - P.UPS_MEAN) / P.UPS_STD) ** 2) / (P.UPS_STD * np.sqrt(2 * np.pi))
    ax.plot(xs, prior, "k--", lw=1.3, label=f"Prior ({P.UPS_MEAN} ± {P.UPS_STD})")
    ax.set_xlabel(r"Fitted stellar mass-to-light ratio $\Upsilon$"); ax.set_ylabel("Number of galaxies")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
    save(fig, out, "fig6_upsilon")


# ----------------------------------------------------------------------------- Figure 7
def fig7_btfr(P, data_dir, out):
    m = P.load_master_table(os.path.join(data_dir, "SPARC_Lelli2016c.mrt"))
    base = m[(m["Vflat"] > 0) & np.isfinite(m["log_M_bar"])]
    good = (base["Quality"] < 3) & (base["Inclination"] > 30)
    x, y = np.log10(base["Vflat"].values), base["log_M_bar"].values
    sx = base["e_Vflat"].values / base["Vflat"].values / np.log(10)
    sx = np.where(np.isfinite(sx) & (sx > 0), sx, .02)
    a_ols, b_ols = np.polyfit(x, y, 1)
    a_tls, b_tls = P.tls_fit(x, y, sx, np.full_like(y, .1))
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(x[~good], y[~good], s=16, color="#b8b6ae", edgecolor="white", lw=.4,
               label="Quality 3 or inclination < 30°")
    ax.scatter(x[good], y[good], s=18, color="#2a78d6", edgecolor="white", lw=.4,
               label="Quality < 3, inclination > 30°")
    xs = np.linspace(x.min() - .05, x.max() + .05, 100)
    ax.plot(xs, a_ols * xs + b_ols, color="#eb6834", lw=1.8, label=f"Ordinary fit, slope {a_ols:.2f}")
    ax.plot(xs, a_tls * xs + b_tls, color="#1baf7a", ls="--", lw=1.8, label=f"Errors on both axes, slope {a_tls:.2f}")
    ax.plot(xs, 4 * xs + (y.mean() - 4 * x.mean()), "k:", lw=1.3, label="MOND prediction, slope 4")
    ax.set_xlabel(r"$\log_{10} V_{\rm flat}$ (km/s)"); ax.set_ylabel(r"$\log_{10} M_{\rm bar}$ ($M_\odot$)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.3)
    save(fig, out, "fig7_btfr")


# ----------------------------------------------------------------------------- Figure 8
def fig8_a0(res, out, P, model="MONDfree"):
    c = f"{model}_log_a0_median"
    if c not in res:
        print("skip fig8: no", c); return
    la = res[c].astype(float).dropna()
    canon, med = np.log10(P.A0), la.median()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(la, bins=np.arange(2.0, 4.75, .1), color="#e34b4b", alpha=.85)
    ax.axvline(canon, color="k", ls="--", lw=1.5, label=f"Canonical $a_0$ = {P.A0:.0f}")
    ax.axvline(med, color="#2a78d6", lw=1.8, label=f"Median of fits = {10 ** med:.0f}")
    ax.set_xlabel(r"$\log_{10} a_0$  [(km/s)$^2$ kpc$^{-1}$]  (MOND with $a_0$ free)")
    ax.set_ylabel("Number of galaxies"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
    save(fig, out, "fig8_a0_distribution")


# ----------------------------------------------------------------------------- stage: figures
def stage_figures(a):
    P = sys.modules[__name__]                      # figure code reuses this file's model functions
    os.makedirs(a.out, exist_ok=True)
    res = pd.read_csv(os.path.join(a.data_dir, "final_results_v2.csv"))
    per = pd.read_csv(os.path.join(a.data_dir, "bic_by_galaxy_v2.csv"))
    summary = pd.read_csv(os.path.join(a.data_dir, "bic_summary_v2.csv"))
    models = ranked_models(summary)
    fig1_example_fits(P, res, per, models, a.data_dir, a.out, a.fig1_galaxies)
    fig2_wins(summary, per, models, a.out)
    fig3_dbic(per, models, a.out)
    fig4_by_points(per, models, a.out)
    fig5_toy(res, a.out)
    fig6_upsilon(res, a.out, P)
    fig7_btfr(P, a.data_dir, a.out)
    fig8_a0(res, a.out, P)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Bayesian rotation-curve model comparison on SPARC.")
    ap.add_argument("stage", choices=["run", "aggregate", "btfr", "figures", "all"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="figures", help="output folder for figures")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--walkers", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--galaxies", nargs="*", default=None)
    # order matters: it sets each model's random seed, so keep new models at the end
    ap.add_argument("--models", default="NFW,Burkert,ISO,MOND,DC14,ToyEmp,ToyBar,MONDfree,MONDrar")
    ap.add_argument("--bic-models", default="NFW,Burkert,ISO,DC14,MOND,ToyBar")
    ap.add_argument("--fig1-galaxies", nargs="*", default=["DDO154", "NGC2903", "NGC3198", "UGC02953"])
    ap.add_argument("--save-chains", action="store_true")
    a = ap.parse_args()
    if a.stage in ("run", "all"):
        stage_run(a)
    if a.stage in ("aggregate", "all"):
        stage_aggregate(a)
    if a.stage in ("btfr", "all"):
        stage_btfr(a)
    if a.stage in ("figures", "all"):
        stage_figures(a)


if __name__ == "__main__":
    main()
