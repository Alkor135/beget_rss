# Now run the full analysis using the detected Russian columns: 'Вводы', 'Выводы', 'Всего на счетах' from the chosen sheet.
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
df['nav'] = pd.to_numeric(df_raw['Всего на счетах'], errors='coerce').fillna(method='ffill')  # forward fill if some rows missing
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

# Define cashflow as deposits positive, withdrawals negative
df['cashflow'] = df['deposit'] - df['withdraw']

# Functions (same as before)
def xnpv(rate, cashflows):
    if rate <= -1.0:
        return float('inf')
    t0 = cashflows[0][0]
    return sum([amt / ((1 + rate) ** (((d - t0).days) / 365.0)) for d, amt in cashflows])

def xirr(cashflows, guess=0.1):
    dates = [d for d, _ in cashflows]
    amounts = [a for _, a in cashflows]
    if len(cashflows) < 2:
        return None
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

def twr_from_series(navs, cashflows):
    navs = np.array(navs, dtype=float)
    cfs = np.array(cashflows, dtype=float)
    returns = []
    for i in range(1, len(navs)):
        denom = navs[i-1] + (cfs[i-1] if cfs[i-1] != 0 else 0.0)
        if denom == 0:
            continue
        ri = (navs[i] - navs[i-1] - cfs[i]) / denom
        returns.append(ri)
    twr = np.prod([1 + r for r in returns]) - 1 if returns else 0.0
    return twr

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

# Overall metrics
start_date = df['date'].iloc[0]
end_date = df['date'].iloc[-1]
start_nav = df['nav'].iloc[0]
end_nav = df['nav'].iloc[-1]

# Prepare cashflows for xirr: if no explicit initial deposit, assume start_nav is initial investment
cashflows = []
if df['cashflow'].iloc[0] == 0 and start_nav != 0:
    cashflows.append((start_date.date(), -float(start_nav)))
for d, amt in zip(df['date'], df['cashflow']):
    if abs(amt) > 1e-9:
        cashflows.append((d.date(), float(amt)))
cashflows.append((end_date.date(), float(end_nav)))

overall = {}
overall['TWR'] = twr_from_series(df['nav'].values, df['cashflow'].values)
try:
    overall['XIRR'] = xirr(cashflows)
except Exception:
    overall['XIRR'] = None
overall['CAGR'] = cagr(start_nav, end_nav, start_date, end_date)
overall['ModifiedDietz'] = modified_dietz(start_nav, end_nav, df['cashflow'].values, start_date, end_date, df['date'].values)

# Monthly metrics
df['month'] = df['date'].dt.to_period('M')
rows = []
for name, g in df.groupby('month', sort=True):
    g = g.sort_values('date')
    sd = g['date'].iloc[0]; ed = g['date'].iloc[-1]
    sv = g['nav'].iloc[0]; ev = g['nav'].iloc[-1]
    cfs = [(sd.date(), -float(sv))] if g['cashflow'].iloc[0] == 0 and sv != 0 else []
    for d, amt in zip(g['date'], g['cashflow']):
        if abs(amt) > 1e-9:
            cfs.append((d.date(), float(amt)))
    cfs.append((ed.date(), float(ev)))
    try:
        x = xirr(cfs)
    except Exception:
        x = None
    twr = twr_from_series(g['nav'].values, g['cashflow'].values)
    cagr_m = cagr(sv, ev, sd, ed)
    md = modified_dietz(sv, ev, g['cashflow'].values, sd, ed, g['date'].values)
    rows.append({'period': str(name), 'start_date': sd, 'end_date': ed, 'start_nav': sv, 'end_nav': ev,
                 'total_cashflow': g['cashflow'].sum(), 'TWR': twr, 'XIRR': x, 'CAGR': cagr_m, 'ModifiedDietz': md})

res_df = pd.DataFrame(rows)
# Add overall as first row
ov_row = {'period': 'overall', 'start_date': start_date, 'end_date': end_date, 'start_nav': start_nav,
          'end_nav': end_nav, 'total_cashflow': df['cashflow'].sum(), 'TWR': overall['TWR'],
          'XIRR': overall['XIRR'], 'CAGR': overall['CAGR'], 'ModifiedDietz': overall['ModifiedDietz']}
res_df = pd.concat([pd.DataFrame([ov_row]), res_df], ignore_index=True)

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
    display_dataframe_to_user("Monthly and overall metrics", res_df)
except Exception:
    print(res_df.head().to_string())

{"generated_files": {"csv": res_csv, "xlsx": res_xlsx, "png": res_png}}
