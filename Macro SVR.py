import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


# Parameters ___________________________________________________
target_col       = 'GS10'  # df_O column to predict/trade, e.g. 'GS10', ln(GSCI), ln(SP500), 'ln(Gold)'
# Option to Windsorize X and y Variable
windsor_value    = 0 #y
windsor_value_X  = 0 # X
# Option to remove Covid Data
exclude_covid    = False    # True = drop 2020-01-01 to 2021-01-01
show_plots       = True   # False = skip all matplotlib plots
# Visualize splitting long vs short over specific percentiles of each variable
# This is more so used to visually see how each individual variable split performed 
run_gs10_splits   = False    # True = percentile-split backtests for GS10 across all X variables (This is more descriptive than anything else)
# used as risk adjsuted return benchmark
use_sortino       = False    # True = downside-deviation (Sortino) ratio; False = standard-deviation (Sharpe) ratio
direction         = 'long_short'   # 'long' = 0/1, 'long_short' = ±1, 'short' = 0/−1
#cpcv parameters
cpcv_n_groups  = 6   # CPCV: split data into N groups; C(N, k) total folds
cpcv_k_test    = 2   # CPCV: number of groups held out as test per fold
#pls parameters
run_gs_pls            = True  # True = PLS regression on GS10 with CPCV, 1 components
pls_direction_filter  = True  # True = long only when GS10_diff12 < 0, short only when > 0
pls_pos_power         = .5   # position = sign(pos) * |pos|^power  (1.0 = no adjustment)

def _risk_denom(x):
    """Standard deviation (Sharpe) or downside deviation vs. 0 (Sortino),
    selected by use_sortino."""
    if use_sortino:
        return np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    return x.std()

def _score_label():
    return "SoR" if use_sortino else "SR"

# Load Data ____________________________________________________
df = pd.read_csv('Macro Trading Factors.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# df is only known 1 period after the date it's stamped with, so lag every
# column by 1 before it can be used as a contemporaneous feature.
_df_cols = [c for c in df.columns if c != 'Date']
df[[f'lag1_{c}' for c in _df_cols]] = df[_df_cols].shift(1)
df = df.drop(columns=_df_cols)

df_O = pd.read_csv('Live Variables.csv', parse_dates=['Date'])
df_O = df_O.sort_values('Date')

# Create X and y variables_______________________________________

# y: forward 1-month cumulative return less 1 month of risk-free rate
df_O_idx    = df_O.set_index('Date')
fwd_returns = df_O.iloc[:, 2:].diff(1).shift(-1)
y           = fwd_returns.subtract(df_O.iloc[:, 1].values / 1200, axis=0)
y.index     = df_O['Date']

dtb3 = df_O_idx['DTB3']
for col, tenor in [('GS5', 5), ('GS10', 10)]:
    if col in y.columns:
        price = 1 / (1 + df_O_idx[col] / 100) ** tenor
        fwd_price_return = price.pct_change(1).shift(-1)
        y[col] = fwd_price_return + df_O_idx[col] / 1200 - dtb3 / 1200

for col in ['DTB3', 'BAA_AAA']:
    if col not in y.columns and col in df_O_idx.columns:
        y[col] = df_O_idx[col].diff(1).shift(-1)

# 1-month forward return reference — used for Sharpe/P&L,
# so model targets can use a longer horizon while performance is always scored on
# the actual realized 1-month return.
fwd_returns_1m = df_O.iloc[:, 2:].diff(1).shift(-1)
y_1m           = fwd_returns_1m.subtract(df_O.iloc[:, 1].values / 1200, axis=0)
y_1m.index     = df_O['Date']

for col, tenor in [('GS5', 5), ('GS10', 10)]:
    if col in y_1m.columns:
        price_1m = 1 / (1 + df_O_idx[col] / 100) ** tenor
        fwd_price_return_1m = price_1m.pct_change(1).shift(-1)
        y_1m[col] = fwd_price_return_1m + df_O_idx[col] / 1200 - dtb3 / 1200

for col in ['DTB3', 'BAA_AAA']:
    if col not in y_1m.columns and col in df_O_idx.columns:
        y_1m[col] = df_O_idx[col].diff(1).shift(-1)

# Align on common dates, drop NaN
X = df.set_index('Date')
X = X.join(df_O_idx[['BAA_AAA','GS10_DTB3', 'DTB3']], how='left')

# Add 3-, 6-, 9-, and 12-period differences of all df_O variables (exclude GS10_DTB3)
lag_periods = [3, 6, 9, 12]
_lag_base = df_O_idx.drop(columns=['GS10_DTB3'], errors='ignore')
lag_diffs = pd.concat(
    [_lag_base.diff(k).add_suffix(f'_diff{k}') for k in lag_periods],
    axis=1,
)
X = X.join(lag_diffs, how='left')

# CPCV embargo width: test obs within max(lag_periods) periods of training data
# have features (backward-looking diffs) or labels (1 period ahead) that
# reach into training data, so trim that much off every test-block edge that
# borders training.
cpcv_embargo_periods = max(1, max(lag_periods))

def _trim_test_edges(test_idx_full, n_obs_, embargo):
    """Drop the first/last `embargo` obs of each contiguous run in
    test_idx_full, unless that edge is the very start/end of the series."""
    breaks = np.where(np.diff(test_idx_full) != 1)[0] + 1
    keep = np.ones(len(test_idx_full), dtype=bool)
    for run in np.split(np.arange(len(test_idx_full)), breaks):
        lo, hi = test_idx_full[run[0]], test_idx_full[run[-1]]
        trim_lo = lo if lo == 0 else lo + embargo
        trim_hi = hi if hi == n_obs_ - 1 else hi - embargo
        keep[run] &= (test_idx_full[run] >= trim_lo) & (test_idx_full[run] <= trim_hi)
    return test_idx_full[keep]

y = y.drop(columns=['GS10_DTB3'], errors='ignore')
combined = X.join(y, how='inner', lsuffix='_x').dropna()
y_clean = combined[y.columns]
y_clean_1m = y_1m.reindex(combined.index)   # 1-month return aligned to same dates, unwindsorized

if exclude_covid:
    mask = (y_clean.index >= '2020-01-01') & (y_clean.index <= '2021-01-01')
    y_clean    = y_clean[~mask]
    y_clean_1m = y_clean_1m[~mask]
y_p10 = y_clean.quantile(windsor_value)
y_p90 = y_clean.quantile(1 - windsor_value)
y_clean = y_clean.clip(lower=y_p10, upper=y_p90, axis=1)

X_feat_cols = [c for c in combined.columns if c not in y.columns]
X_feat = combined[X_feat_cols].loc[y_clean.index]

X_p_lo = X_feat.quantile(windsor_value_X)
X_p_hi = X_feat.quantile(1 - windsor_value_X)
X_feat = X_feat.clip(lower=X_p_lo, upper=X_p_hi, axis=1)



# --- Standardize X_feat once (shared scaler for fitting + grid transforms) ---
scaler_X = StandardScaler()
X_feat_scaled = pd.DataFrame(
    scaler_X.fit_transform(X_feat),
    columns=X_feat_cols,
    index=X_feat.index,
)
if run_gs10_splits and target_col in y_clean.columns:
    gs10_ret      = y_clean_1m[target_col].values   # Sharpe/P&L always uses 1-month forward return
    gs10_index    = y_clean[target_col].index
    ann_factor    = np.sqrt(12)
    score_lbl     = _score_label()
    always_ret    = gs10_ret
    sharpe_always = always_ret.mean() / (_risk_denom(always_ret) + 1e-12) * ann_factor
    cum_always    = np.cumsum(always_ret)
    percentiles   = [20, 40, 60, 80]
    split_rows    = []

    for x_col in X_feat_cols:
        x_vals = X_feat[x_col].values

        if show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
            fig.suptitle(f'{target_col} Cumulative Real Return — Percentile Splits: {x_col}', fontsize=11)

        for p_idx, pct in enumerate(percentiles):
            threshold  = np.percentile(x_vals, pct)
            high_mask  = x_vals > threshold
            frac_high  = high_mask.mean()
            frac_low   = 1.0 - frac_high
            high_ret   = np.where(high_mask,  gs10_ret, 0.0)
            low_ret    = np.where(~high_mask, gs10_ret, 0.0)
            ls_raw     = np.where(high_mask,  gs10_ret, -gs10_ret)
            # flip sign if long/short ends negative so it always ends positive
            ls_sign    = 1.0 if ls_raw.sum() >= 0 else -1.0
            ls_ret     = ls_raw * ls_sign

            sh_high    = high_ret.mean() / (_risk_denom(high_ret) + 1e-12) * ann_factor
            sh_low     = low_ret.mean()  / (_risk_denom(low_ret)  + 1e-12) * ann_factor
            sh_ls      = ls_ret.mean()   / (_risk_denom(ls_ret)   + 1e-12) * ann_factor

            # Rescale each series' vol to match the always-invested vol, so the
            # cumulative-return plot compares returns at a common risk level
            # (Sharpe ratios above are unaffected — they're scale-invariant).
            std_target = always_ret.std() + 1e-12
            high_ret_scaled = high_ret * (std_target / (high_ret.std() + 1e-12))
            low_ret_scaled  = low_ret  * (std_target / (low_ret.std()  + 1e-12))
            ls_ret_scaled   = ls_ret   * (std_target / (ls_ret.std()   + 1e-12))

            split_rows.append({'x': x_col, 'pct_split': pct,
                                'frac_high': frac_high, 'frac_low': frac_low,
                                'sharpe_high': sh_high, 'sharpe_low': sh_low,
                                'sharpe_long_short': sh_ls,
                                'ls_flipped': ls_sign < 0})

            if show_plots:
                ls_lbl = 'L/S (flipped)' if ls_sign < 0 else 'L/S'
                ax = axes[p_idx // 2][p_idx % 2]
                ax.plot(gs10_index, cum_always,
                        label=f'Always  {score_lbl}={sharpe_always:.2f}',
                        color='steelblue', linewidth=1.2, alpha=0.7)
                ax.plot(gs10_index, np.cumsum(high_ret_scaled),
                        label=f'High >p{pct} ({frac_high:.0%})  {score_lbl}={sh_high:.2f}',
                        color='darkgreen', linewidth=1.3)
                ax.plot(gs10_index, np.cumsum(low_ret_scaled),
                        label=f'Low ≤p{pct} ({frac_low:.0%})  {score_lbl}={sh_low:.2f}',
                        color='firebrick', linewidth=1.3)
                ax.plot(gs10_index, np.cumsum(ls_ret_scaled),
                        label=f'{ls_lbl}  {score_lbl}={sh_ls:.2f}',
                        color='darkorange', linewidth=1.3, linestyle='--')
                ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
                ax.set_title(f'Split at {pct}th pct  (threshold = {threshold:.3f})', fontsize=9)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

        if show_plots:
            plt.tight_layout()
            plt.show()

    split_df = (pd.DataFrame(split_rows)
                  .sort_values('sharpe_long_short', ascending=False)
                  .reset_index(drop=True))
    print(f"\n{target_col} Percentile-Split Backtest Summary  "
          f"(always-invested {score_lbl} = {sharpe_always:.2f}):")
    print(split_df.to_string(index=False))


# ── PLS regression on target_col with CPCV ───────────────────────────────────
if run_gs_pls and target_col in y_clean.columns:

    gs10_ret_p   = y_clean[target_col].values        # 1-period-ahead target used to fit PLS
    gs10_pnl_p   = y_clean_1m[target_col].values     # 1-month forward return used for Sharpe/P&L
    gs10_index_p = y_clean[target_col].index
    ann_factor_p = np.sqrt(12)
    n_obs_p      = len(gs10_ret_p)
    score_lbl_p  = _score_label()

    def _score_p(pos, ret, af):
        s = pos * ret
        return s.mean() / (_risk_denom(s) + 1e-12) * af

    # Build CPCV folds 
    from itertools import combinations as _comb_p
    _gs_p  = n_obs_p // cpcv_n_groups
    _grps_p = [
        np.arange(i * _gs_p,
                  n_obs_p if i == cpcv_n_groups - 1 else (i + 1) * _gs_p)
        for i in range(cpcv_n_groups)
    ]
    pls_folds = []
    for _tg in _comb_p(range(cpcv_n_groups), cpcv_k_test):
        _tg_set    = set(_tg)
        _te_full   = np.concatenate([_grps_p[g] for g in sorted(_tg)])
        _te_idx    = _trim_test_edges(_te_full, n_obs_p, cpcv_embargo_periods)
        _tr_idx    = np.concatenate([_grps_p[g] for g in range(cpcv_n_groups)
                                     if g not in _tg_set])
        if len(_tr_idx) > 0 and len(_te_idx) > 0:
            pls_folds.append((_tr_idx, _te_idx))

    n_pls_folds  = len(pls_folds)
    X_pls        = X_feat_scaled.values   # already standardised

    def _idx_blocks(idx):
        """Collapse a sorted index array into contiguous (start, end) date ranges."""
        idx = np.sort(idx)
        blocks = []
        start = prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            blocks.append((start, prev))
            start = prev = i
        blocks.append((start, prev))
        return blocks

    # Direction filter: clamp position sign based on target_col_diff12 level
    _diff12_col = f'{target_col}_diff12'
    _d12 = (X_feat[_diff12_col].values
            if pls_direction_filter and _diff12_col in X_feat_cols
            else None)

    def _apply_direction(pos, idx):
        """Long only when target_col_diff12[idx] < 0; short only when > 0."""
        if _d12 is None:
            return pos
        d = _d12[idx]
        return np.where(d < 0, np.maximum(pos, 0.0),
               np.where(d > 0, np.minimum(pos, 0.0), pos))

    def _apply_power(pos):
        """Non-linear position scaling: sign(pos) * |pos|^pls_pos_power."""
        if pls_pos_power == 1.0:
            return pos
        return np.sign(pos) * np.abs(pos) ** pls_pos_power

    def _size_position(pred, var):
        """Continuous Kelly position size: pred / realized_var."""
        return pred / var

    nc = 1
    best_oos_sc   = -np.inf
    best_nc       = 1
    best_oos_pos  = None
    best_oos_mask = None
    best_pos_is   = None
    best_sc_is    = None
    best_pls_is   = None

    oos_pos_sum = np.zeros(n_obs_p)
    oos_pos_cnt = np.zeros(n_obs_p)
    is_sc_folds  = []
    oos_sc_folds = []
    fold_pos_records = []   # (tr_idx, te_idx, pos_tr, pos_te) per fold, for the per-fold plot

    print(f"\nCPCV folds for PLS {target_col} — C({cpcv_n_groups},{cpcv_k_test})={n_pls_folds} folds, "
          f"{nc} component, n_obs={n_obs_p}:")
    for _fk, (tr_idx, te_idx) in enumerate(pls_folds, 1):
        X_tr = X_pls[tr_idx]
        y_tr = gs10_ret_p[tr_idx].reshape(-1, 1)
        X_te = X_pls[te_idx]

        nc_safe = min(nc, X_tr.shape[1], X_tr.shape[0])
        pls = PLSRegression(n_components=nc_safe, scale=False)
        pls.fit(X_tr, y_tr)

        pred_tr  = pls.predict(X_tr).ravel()
        pred_te  = pls.predict(X_te).ravel()
        tr_var   = gs10_pnl_p[tr_idx].var() + 1e-12   # realized return variance, training period only
        pos_te   = _apply_power(_apply_direction(_size_position(pred_te, tr_var), te_idx))

        oos_pos_sum[te_idx] += pos_te
        oos_pos_cnt[te_idx] += 1

        pos_tr      = _apply_power(_apply_direction(_size_position(pred_tr, tr_var), tr_idx))
        is_sc_fold  = _score_p(pos_tr, gs10_pnl_p[tr_idx], ann_factor_p)
        oos_sc_fold = _score_p(pos_te, gs10_pnl_p[te_idx], ann_factor_p)
        is_sc_folds.append(is_sc_fold)
        oos_sc_folds.append(oos_sc_fold)
        fold_pos_records.append((tr_idx, te_idx, pos_tr, pos_te))

        print(f"\n  Fold {_fk}: train n={len(tr_idx)}, test n={len(te_idx)}")
        print("    Train (in-sample):")
        for s, e in _idx_blocks(tr_idx):
            print(f"      {gs10_index_p[s].date()}  to  {gs10_index_p[e].date()}  (n={e - s + 1})")
        print("    Test (out-of-sample):")
        for s, e in _idx_blocks(te_idx):
            print(f"      {gs10_index_p[s].date()}  to  {gs10_index_p[e].date()}  (n={e - s + 1})")
        print(f"    IS {score_lbl_p} = {is_sc_fold:.4f}   OOS {score_lbl_p} = {oos_sc_fold:.4f}")

    oos_mask_p = oos_pos_cnt > 0
    oos_pos    = np.where(oos_mask_p,
                              oos_pos_sum / np.maximum(oos_pos_cnt, 1), 0.0)

    is_sc_std  = np.std(is_sc_folds)
    oos_sc_std = np.std(oos_sc_folds)

    print(f"\nPLS {target_col} — CPCV C({cpcv_n_groups},{cpcv_k_test})={n_pls_folds} folds, "
          f"1 component, n_obs={n_obs_p}:")
    print(f"  {'n_comp':>6}  {'IS '+score_lbl_p:>8}  {'OOS '+score_lbl_p:>9}"
          f"  {'IS std':>8}  {'OOS std':>8}")

    # IS reference: fit on full data
    nc_safe_is = min(nc, X_pls.shape[1], X_pls.shape[0])
    pls_is = PLSRegression(n_components=nc_safe_is, scale=False)
    pls_is.fit(X_pls, gs10_ret_p.reshape(-1, 1))
    pred_is    = pls_is.predict(X_pls).ravel()
    full_var   = gs10_pnl_p.var() + 1e-12   # realized return variance, full dataset
    pos_is     = _apply_power(_apply_direction(_size_position(pred_is, full_var), np.arange(n_obs_p)))

    sc_is  = _score_p(pos_is,               gs10_pnl_p,              ann_factor_p)
    sc_oos = _score_p(oos_pos[oos_mask_p],  gs10_pnl_p[oos_mask_p],  ann_factor_p)

    print(f"  {nc:>6}  {sc_is:>8.4f}  {sc_oos:>9.4f}"
          f"  {is_sc_std:>8.4f}  {oos_sc_std:>8.4f}")

    if sc_oos > best_oos_sc:
        best_oos_sc   = sc_oos
        best_nc       = nc
        best_oos_pos  = oos_pos.copy()
        best_oos_mask = oos_mask_p.copy()
        best_pos_is   = pos_is.copy()
        best_sc_is    = sc_is
        best_pls_is   = pls_is

    print(f"\n  Best PLS OOS: 1 component"
          f"IS {score_lbl_p}={best_sc_is:.4f}  OOS {score_lbl_p}={best_oos_sc:.4f}")

    # Final PLS components, fit on the full dataset, for the best-OOS n_components
    coefs      = best_pls_is.coef_.ravel()
    intercept  = float(best_pls_is.predict(np.zeros((1, X_pls.shape[1]))).ravel()[0])
    coef_df    = pd.DataFrame({'variable': list(X_feat_cols) + ['const'],
                                'coef': list(coefs) + [intercept]})
    coef_df    = coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=False).index)
    print(f"\n  Final PLS coefficients (full-data fit, {best_nc} components):")
    print(coef_df.to_string(index=False))

    if show_plots:
        always_sc_p  = _score_p(np.ones(n_obs_p), gs10_pnl_p, ann_factor_p)
        strat_is_p   = best_pos_is * gs10_pnl_p

        # Long-only version: clamp positions to >= 0
        lo_is_pos   = np.maximum(best_pos_is, 0.0)
        strat_lo_is = lo_is_pos * gs10_pnl_p
        sc_lo_is    = _score_p(lo_is_pos, gs10_pnl_p, ann_factor_p)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)

        ax1.plot(gs10_index_p, np.cumsum(gs10_pnl_p),
                 label=f'Always long  {score_lbl_p}={always_sc_p:.3f}',
                 color='steelblue', linewidth=1.2, alpha=0.6)
        ax1.plot(gs10_index_p, np.cumsum(strat_is_p),
                 label=f'IS ({best_nc} comps)  {score_lbl_p}={best_sc_is:.3f}',
                 color='gray', linewidth=1.5)
        ax1.plot(gs10_index_p, np.cumsum(strat_lo_is),
                 label=f'Long-only IS  {score_lbl_p}={sc_lo_is:.3f}',
                 color='orange', linewidth=1.5)
        for g in _grps_p:
            ax1.axvline(gs10_index_p[g[0]], color='gray', linewidth=0.4, linestyle=':')
        ax1.axhline(0, color='black', linewidth=0.4, linestyle='--')
        ax1.set_title(f'{target_col} PLS Regression — trained on full dataset, '
                      f'{best_nc} components (in-sample)')
        ax1.set_ylabel('Cumulative Real Return')
        ax1.legend(fontsize=8)

        ax2.plot(gs10_index_p, best_pos_is,
                 color='black', linewidth=0.9, alpha=0.8, label='IS position (L/S)')
        ax2.plot(gs10_index_p, lo_is_pos,
                 color='darkorange', linewidth=0.9, alpha=0.8, label='IS position (long only)')
        ax2.axhline(0, color='black', linewidth=0.5)
        for g in _grps_p:
            ax2.axvline(gs10_index_p[g[0]], color='gray', linewidth=0.4, linestyle=':')
        ax2.set_ylabel('Position (pred / σ_pred)')
        ax2.legend(fontsize=8, loc='upper left')
        ax2.set_xlabel('Date')

        plt.tight_layout()
        plt.show()

        # Per-fold train/test map: shows train/test date ranges shaded, plus each
        # fold's own long-short and long-only strategy cumulative return, split
        # into its IS (train) and OOS (test) segments, with Sharpes in the title.
        from matplotlib.patches import Patch as _Patch
        from matplotlib.lines import Line2D as _Line2D

        cum_always_p = np.cumsum(gs10_pnl_p)

        def _anchored_segment_curve(idx, pos):
            """Cumulative return of `pos` over `idx` only, as a full-length series
            that's NaN outside idx (so plotting shows nothing there) and, at the
            start of each contiguous run, anchored to the always-long cumulative
            return level just before that run — so the strategy curve begins
            level with the benchmark and visibly diverges from there."""
            curve = np.full(n_obs_p, np.nan)
            for s, e in _idx_blocks(idx):
                sel      = (idx >= s) & (idx <= e)
                block_pts = idx[sel]
                block_pos = pos[sel]
                offset    = cum_always_p[s - 1] if s > 0 else 0.0
                curve[block_pts] = offset + np.cumsum(block_pos * gs10_pnl_p[block_pts])
            return curve

        _n_cols = 3
        _n_rows = int(np.ceil(n_pls_folds / _n_cols))
        fig3, axes3 = plt.subplots(_n_rows, _n_cols,
                                    figsize=(5.5 * _n_cols, 3.2 * _n_rows),
                                    sharex=True, sharey=True)
        axes3 = np.atleast_1d(axes3).ravel()

        for _fk, (tr_idx, te_idx, pos_tr, pos_te) in enumerate(fold_pos_records):
            ax = axes3[_fk]

            lo_pos_tr = np.maximum(pos_tr, 0.0)
            lo_pos_te = np.maximum(pos_te, 0.0)

            is_ls_sc     = is_sc_folds[_fk]
            oos_ls_sc    = oos_sc_folds[_fk]
            is_lo_sc     = _score_p(lo_pos_tr, gs10_pnl_p[tr_idx], ann_factor_p)
            oos_lo_sc    = _score_p(lo_pos_te, gs10_pnl_p[te_idx], ann_factor_p)
            is_always_sc  = _score_p(np.ones(len(tr_idx)), gs10_pnl_p[tr_idx], ann_factor_p)
            oos_always_sc = _score_p(np.ones(len(te_idx)), gs10_pnl_p[te_idx], ann_factor_p)

            curve_is_ls  = _anchored_segment_curve(tr_idx, pos_tr)
            curve_oos_ls = _anchored_segment_curve(te_idx, pos_te)
            curve_is_lo  = _anchored_segment_curve(tr_idx, lo_pos_tr)
            curve_oos_lo = _anchored_segment_curve(te_idx, lo_pos_te)

            ax.plot(gs10_index_p, cum_always_p, color='steelblue', linewidth=0.8, alpha=0.4)
            ax.plot(gs10_index_p, curve_is_ls,  color='gray',       linewidth=1.0, linestyle='--')
            ax.plot(gs10_index_p, curve_oos_ls, color='darkgreen', linewidth=1.3)
            ax.plot(gs10_index_p, curve_is_lo,  color='orange',    linewidth=1.0, linestyle='--')
            ax.plot(gs10_index_p, curve_oos_lo, color='darkorange',linewidth=1.3)

            for s, e in _idx_blocks(tr_idx):
                ax.axvspan(gs10_index_p[s], gs10_index_p[e], color='tab:blue', alpha=0.10)
            for s, e in _idx_blocks(te_idx):
                ax.axvspan(gs10_index_p[s], gs10_index_p[e], color='tab:orange', alpha=0.15)

            ax.set_title(f'Fold {_fk + 1}   LS: IS {is_ls_sc:.2f} / OOS {oos_ls_sc:.2f}    '
                         f'LO: IS {is_lo_sc:.2f} / OOS {oos_lo_sc:.2f}    '
                         f'AL: IS {is_always_sc:.2f} / OOS {oos_always_sc:.2f}', fontsize=7)
            ax.tick_params(labelsize=7)

        for ax in axes3[n_pls_folds:]:
            ax.axis('off')

        _legend_handles = [
            _Line2D([0], [0], color='steelblue',  linewidth=1.0, alpha=0.4, label='Always long'),
            _Line2D([0], [0], color='gray',        linewidth=1.0, linestyle='--', label='Long-short IS'),
            _Line2D([0], [0], color='darkgreen',   linewidth=1.3, label='Long-short OOS'),
            _Line2D([0], [0], color='orange',      linewidth=1.0, linestyle='--', label='Long-only IS'),
            _Line2D([0], [0], color='darkorange',  linewidth=1.3, label='Long-only OOS'),
            _Patch(color='tab:blue',   alpha=0.10, label='Train period'),
            _Patch(color='tab:orange', alpha=0.15, label='Test period'),
        ]
        fig3.legend(handles=_legend_handles, loc='upper center', ncol=4, fontsize=8)
        fig3.suptitle(f'{target_col} — CPCV C({cpcv_n_groups},{cpcv_k_test})={n_pls_folds} folds: '
                      f'IS/OOS long-short vs long-only performance', y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

