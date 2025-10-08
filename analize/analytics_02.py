# full analysis with extra metrics: volatility, max drawdown, avg win/loss per period, Sharpe, profit factor, etc.
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

input_path = Path(r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx")
xls = pd.read_excel(input_path, sheet_name=None)
df_raw = xls['Data'].copy()

# Normalize columns to strip whitespace
df_raw.columns = [c.strip() if isinstance(c, str) else c for c in df_raw.columns]

# Check expected columns
expected = ['Дата', 'Вводы', 'Выводы', 'Всего на счетах']
for c in expected:
    if c not in df_raw.columns:
        raise RuntimeError(f"Expected column '{c}' not found in sheet. Columns: {df_raw.columns.tolist()}")

df = pd.DataFrame()
df['date'] = pd.to_datetime(df_raw['Дата'], errors='coerce')
df['deposit'] = pd.to_numeric(df_raw['Вводы'], errors='coerce').fillna(0.0)
df['withdraw'] = pd.to_numeric(df_raw['Выводы'], errors='coerce').fillna(0.0)
df['nav'] = pd.to_numeric(df_raw['Всего на счетах'], errors='coerce').ffill()  # forward fill if some rows missing
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

# Define cashflow as deposits positive, withdrawals negative
df['cashflow'] = df['deposit'] - df['withdraw']

# --- Utility functions -------------------------------------------------
def xirr(cashflows, guess=0.1):
    """Compute XIRR using Newton method. cashflows: list of (date (datetime.date), amount)"""
    if len(cashflows) < 2:
        return None
    dates = [d for d, _ in cashflows]
    amounts = [a for _, a in cashflows]
    t0 = min(dates)
    def f(r):
        return sum([a / ((1 + r) ** (((d - t0).days) / 365.0)) for d, a in cashflows])
    def dfun(r):
        return sum([-a * (((d - t0).days) / 365.0) / ((1 + r) ** ((((d - t0).days) / 365.0) + 1)) for d, a in cashflows])
    r = guess
    for i in range(200):
        fv = f(r)
        dfv = dfun(r)
        if abs(fv) < 1e-8:
            return r
        if dfv == 0:
            r += 0.1
            continue
        r_new = r - fv/dfv
        if abs(r_new - r) < 1e-9:
            return r_new
        r = r_new
    return r

def cagr(start_val, end_val, start_date, end_date):
    years = (end_date - start_date).days / 365.0
    if years <= 0 or start_val <= 0:
        return None
    return (end_val / start_val) ** (1.0/years) - 1.0

def period_returns_and_dates(navs, cashflows, dates):
    """
    Return list of (date_end, return) for each period i->i+1 using:
    r_i = (nav[i] - nav[i-1] - cf[i]) / (nav[i-1] + cf[i-1] if cf[i-1]!=0 else nav[i-1])
    This matches the TWR period-return approach used earlier.
    Also returns list of period lengths in days between dates.
    """
    navs = np.array(navs, dtype=float)
    cfs = np.array(cashflows, dtype=float)
    rets = []
    period_days = []
    for i in range(1, len(navs)):
        denom = navs[i-1] + (cfs[i-1] if cfs[i-1] != 0 else 0.0)
        if denom == 0:
            continue
        ri = (navs[i] - navs[i-1] - cfs[i]) / denom
        rets.append(ri)
        period_days.append((pd.to_datetime(dates[i]) - pd.to_datetime(dates[i-1])).days)
    return rets, period_days

def modified_dietz(start_val, end_val, cashflows, start_date, end_date, dates):
    BMV = start_val
    EMV = end_val
    T = (end_date - start_date).days
    if T <= 0:
        return None
    cf = np.array(cashflows, dtype=float)
    numerator = EMV - BMV - cf.sum()
    denom = BMV
    for d, c in zip(dates, cf):
        t_i = (end_date - d).days
        w_i = t_i / T
        denom += w_i * c
    if denom == 0:
        return None
    return numerator / denom

def twr_from_series(navs, cashflows):
    rets, _ = period_returns_and_dates(navs, cashflows, [None]*len(navs))
    return np.prod([1 + r for r in rets]) - 1 if rets else 0.0

# --- Исправленная функция TWR ---------------------------------------------
def twr_from_series(navs, cashflows, dates):
    """Вычисляет TWR и возвращает итоговую доходность"""
    rets, _ = period_returns_and_dates(navs, cashflows, dates)
    return np.prod([1 + r for r in rets]) - 1 if rets else 0.0

# --- Overall metrics ---------------------------------------------------
start_date = df['date'].iloc[0]
end_date = df['date'].iloc[-1]
start_nav = df['nav'].iloc[0]
end_nav = df['nav'].iloc[-1]

# Prepare cashflows list for XIRR: add starting NAV as initial deposit if no explicit first cashflow
cashflows = []
if df['cashflow'].iloc[0] == 0 and start_nav != 0:
    cashflows.append((start_date.date(), -float(start_nav)))
for d, amt in zip(df['date'], df['cashflow']):
    if abs(amt) > 1e-9:
        cashflows.append((d.date(), float(amt)))
cashflows.append((end_date.date(), float(end_nav)))

overall = {}
overall['TWR'] = twr_from_series(df['nav'].values, df['cashflow'].values, df['date'].values)
try:
    overall['XIRR'] = xirr(cashflows)
except Exception:
    overall['XIRR'] = None
overall['CAGR'] = cagr(start_nav, end_nav, start_date, end_date)
overall['ModifiedDietz'] = modified_dietz(start_nav, end_nav, df['cashflow'].values, start_date, end_date, df['date'].values)

# --- Extra metrics: volatility, max drawdown, avg win/loss, Sharpe, profit factor ----
# Compute period returns and corresponding days
period_rets, period_days = period_returns_and_dates(df['nav'].values, df['cashflow'].values, df['date'].values)

# Determine annualization factor from median period length (works for irregular dates)
if len(period_days) > 0:
    median_period_days = np.median([d for d in period_days if d > 0]) if np.median(period_days) > 0 else 1.0
else:
    median_period_days = 1.0
annual_factor = 365.0 / float(median_period_days)  # multiply mean return by this, volatility by sqrt(this)

# Volatility (annualized)
if len(period_rets) > 1:
    vol_period = np.std(period_rets, ddof=1)
    vol_annual = vol_period * np.sqrt(annual_factor)
else:
    vol_period = None
    vol_annual = None

# Win/Loss stats (by period)
wins = [r for r in period_rets if r > 0]
losses = [r for r in period_rets if r < 0]
win_count = len(wins)
loss_count = len(losses)
total_periods = len(period_rets)
avg_win = np.mean(wins) if wins else None
avg_loss = np.mean(losses) if losses else None
win_rate = (win_count / total_periods) if total_periods > 0 else None
sum_wins = sum(wins) if wins else 0.0
sum_losses = sum(losses) if losses else 0.0
profit_factor = (sum_wins / abs(sum_losses)) if (abs(sum_losses) > 0) else (np.inf if sum_wins>0 else None)

# Sharpe (annualized) using arithmetic mean of period returns
rf = 0.0  # default risk-free rate (annual)
if len(period_rets) > 0 and vol_annual not in (None, 0):
    mean_period = np.mean(period_rets)
    ann_return_from_periods = mean_period * annual_factor
    sharpe = (ann_return_from_periods - rf) / vol_annual
else:
    sharpe = None

# Max drawdown computed on NAV series
nav_series = df['nav'].values.astype(float)
if len(nav_series) > 0:
    running_max = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series / running_max) - 1.0
    max_drawdown = float(np.min(drawdowns))  # negative value
else:
    max_drawdown = None

# Save these overall extras
overall['Volatility'] = vol_annual
overall['MaxDrawdown'] = max_drawdown
overall['AvgWin'] = avg_win
overall['AvgLoss'] = avg_loss
overall['WinRate'] = win_rate
overall['ProfitFactor'] = profit_factor
overall['Sharpe'] = sharpe

# --- Monthly metrics (same as before) including extra stats per month -------------
df['month'] = df['date'].dt.to_period('M')
rows = []
for name, g in df.groupby('month', sort=True):
    g = g.sort_values('date')
    sd = g['date'].iloc[0]; ed = g['date'].iloc[-1]
    sv = g['nav'].iloc[0]; ev = g['nav'].iloc[-1]
    # cashflow list for xirr in the month
    cfs = [(sd.date(), -float(sv))] if (g['cashflow'].iloc[0] == 0 and sv != 0) else []
    for d, amt in zip(g['date'], g['cashflow']):
        if abs(amt) > 1e-9:
            cfs.append((d.date(), float(amt)))
    cfs.append((ed.date(), float(ev)))
    try:
        x = xirr(cfs)
    except Exception:
        x = None
    twr = twr_from_series(g['nav'].values, g['cashflow'].values, g['date'].values)
    cagr_m = cagr(sv, ev, sd, ed)
    md = modified_dietz(sv, ev, g['cashflow'].values, sd, ed, g['date'].values)

    # extra stats per month
    rets_m, days_m = period_returns_and_dates(g['nav'].values, g['cashflow'].values, g['date'].values)
    if len(days_m) > 0:
        med_days_m = np.median([d for d in days_m if d>0]) if np.median(days_m)>0 else 1.0
    else:
        med_days_m = 1.0
    ann_factor_m = 365.0 / float(med_days_m)
    if len(rets_m) > 1:
        vol_m = np.std(rets_m, ddof=1) * np.sqrt(ann_factor_m)
    else:
        vol_m = None
    wins_m = [r for r in rets_m if r>0]; losses_m = [r for r in rets_m if r<0]
    avg_win_m = np.mean(wins_m) if wins_m else None
    avg_loss_m = np.mean(losses_m) if losses_m else None
    win_rate_m = (len(wins_m)/len(rets_m)) if len(rets_m)>0 else None
    sum_wins_m = sum(wins_m) if wins_m else 0.0
    sum_losses_m = sum(losses_m) if losses_m else 0.0
    profit_factor_m = (sum_wins_m / abs(sum_losses_m)) if (abs(sum_losses_m) > 0) else (np.inf if sum_wins_m>0 else None)
    if len(rets_m)>0 and vol_m not in (None, 0):
        sharpe_m = (np.mean(rets_m) * ann_factor_m - rf) / vol_m
    else:
        sharpe_m = None

    # max drawdown for month
    nav_m = g['nav'].values.astype(float)
    if len(nav_m)>0:
        runmax_m = np.maximum.accumulate(nav_m)
        dd_m = (nav_m / runmax_m) - 1.0
        maxdd_m = float(np.min(dd_m))
    else:
        maxdd_m = None

    rows.append({'period': str(name), 'start_date': sd, 'end_date': ed, 'start_nav': sv, 'end_nav': ev,
                 'total_cashflow': g['cashflow'].sum(), 'TWR': twr, 'XIRR': x, 'CAGR': cagr_m, 'ModifiedDietz': md,
                 'Volatility': vol_m, 'MaxDrawdown': maxdd_m, 'AvgWin': avg_win_m, 'AvgLoss': avg_loss_m,
                 'WinRate': win_rate_m, 'ProfitFactor': profit_factor_m, 'Sharpe': sharpe_m})

# Build result DataFrame and prepend overall row
res_df = pd.DataFrame(rows)
ov_row = {'period': 'overall', 'start_date': start_date, 'end_date': end_date, 'start_nav': start_nav,
          'end_nav': end_nav, 'total_cashflow': df['cashflow'].sum(), 'TWR': overall['TWR'],
          'XIRR': overall['XIRR'], 'CAGR': overall['CAGR'], 'ModifiedDietz': overall['ModifiedDietz'],
          'Volatility': overall['Volatility'], 'MaxDrawdown': overall['MaxDrawdown'], 'AvgWin': overall['AvgWin'],
          'AvgLoss': overall['AvgLoss'], 'WinRate': overall['WinRate'], 'ProfitFactor': overall['ProfitFactor'],
          'Sharpe': overall['Sharpe']}
res_df = pd.concat([pd.DataFrame([ov_row]), res_df], ignore_index=True)

# === Форматирование числовых значений ===
def format_number(x):
    if pd.isna(x):
        return ""
    try:
        if abs(x) >= 1:
            return f"{x:.2f}"
        elif abs(x) >= 0.01:
            return f"{x:.4f}"
        # else:
        #     return f"{x:.2e}"  # очень маленькие значения — научная нотация
    except Exception:
        return str(x)

res_df = res_df.applymap(format_number)

# Save outputs
res_csv = r"C:\Users\Alkor\PycharmProjects\beget_rss\analize\futures_analysis_results.csv"
res_xlsx = r"C:\Users\Alkor\PycharmProjects\beget_rss\analize\futures_analysis_results.xlsx"
res_png = r"C:\Users\Alkor\PycharmProjects\beget_rss\analize\futures_analysis_cumreturn.png"
res_df.to_csv(res_csv, index=False)
with pd.ExcelWriter(res_xlsx) as w:
    res_df.to_excel(w, index=False, sheet_name='summary')
    df.to_excel(w, index=False, sheet_name='series')

# Cumulative return plot
df['cum_return'] = df['nav'] / df['nav'].iloc[0] - 1.0
plt.figure(figsize=(10,6))
plt.plot(df['date'], df['cum_return'])
plt.xlabel("Date")
plt.ylabel("Cumulative Return (nav / start - 1)")
plt.title("Cumulative return")
plt.tight_layout()
plt.savefig(res_png)

# Display results table (first rows) and list generated files
print("Saved files:")
print(res_csv)
print(res_xlsx)
print(res_png)

try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Monthly and overall metrics (with extra stats)", res_df)
except Exception:
    print(res_df.head().to_string())

# end of script
