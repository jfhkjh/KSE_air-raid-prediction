import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

DEFAULT_FILE = "data/volunteer_data_en.csv"  
TZ = "Europe/Kyiv"

# 0. Параметри командного рядка
parser = argparse.ArgumentParser(description="Аналіз часових рядів повітряних тривог")
parser.add_argument("region", nargs="?", default="Kyiv City",
                    help='назва регіону з колонки region, напр. "Kyiv City"')
parser.add_argument("--file", default=DEFAULT_FILE, help="шлях до CSV")
parser.add_argument("--skip-existing", action="store_true",
                    help="пропустити регіон, для якого результати вже є")
args = parser.parse_args()

REGION = args.region
PREFIX = re.sub(r"[^A-Za-z0-9]+", "_", REGION).strip("_")   # "Kyiv City" -> "Kyiv_City"


def out(name):
    """Шлях до файлу результату з назвою регіону: outputs/Kyiv_City_daily.png"""
    return f"outputs/{PREFIX}_{name}"


os.makedirs("outputs", exist_ok=True)

if args.skip_existing and os.path.exists(out("anomalies.csv")):
    print(f"Результати для {REGION} вже існують, пропускаю.")
    sys.exit(0)

# 1. Завантаження і очищення
df = pd.read_csv(args.file)
regions = sorted(df["region"].dropna().unique())
if REGION not in regions:
    print(f'Регіон "{REGION}" не знайдено. Доступні регіони:')
    for r in regions:
        print("  ", r)
    sys.exit(1)

df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")
df = df[df["region"] == REGION].dropna(subset=["started_at", "finished_at"])
df = df[df["finished_at"] > df["started_at"]].copy()

df["started_at"] = df["started_at"].dt.tz_convert(TZ)
df["finished_at"] = df["finished_at"].dt.tz_convert(TZ)
df["duration_h"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600

print(f"Регіон: {REGION}, тривог у вибірці: {len(df)}")
print(f"Період: {df['started_at'].min()} - {df['started_at'].max()}")

# 2. Часовий ряд: кількість і сумарна тривалість тривог за добу
daily = (df.set_index("started_at")
           .resample("D")
           .agg(count=("duration_h", "size"), hours=("duration_h", "sum")))
daily = daily.iloc[:-1]   # остання доба неповна: відкидаємо її скрізь           

# 3. Графік + ковзне середнє
daily["count_ma7"] = daily["count"].rolling(7).mean()
ax = daily[["count", "count_ma7"]].plot(figsize=(12, 4), alpha=0.8)
ax.set_title(f"Тривоги за добу, {REGION}")
plt.savefig(out("daily.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4. Теплокарта «день тижня × година» (київський час)
df["hour"] = df["started_at"].dt.hour
df["weekday"] = df["started_at"].dt.dayofweek
heat = (df.pivot_table(index="weekday", columns="hour",
                       values="duration_h", aggfunc="size", fill_value=0)
          .reindex(index=range(7), columns=range(24), fill_value=0))
plt.figure(figsize=(10, 3))
plt.imshow(heat, aspect="auto", cmap="Reds")
plt.title(f"Початки тривог, {REGION}")
plt.xlabel("Година (київський час)")
plt.ylabel("День тижня (0=пн)")
plt.colorbar(label="Кількість початків тривог")
plt.savefig(out("heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# 5. Розклад на тренд, сезонність, шум
res = STL(daily["count"], period=7, robust=True).fit()
fig = res.plot()
fig.savefig(out("stl.png"), dpi=150)
plt.close(fig)

# 6. Аномалії (вгору і вниз)
resid = res.resid
mask = resid.abs() > 3 * resid.std()
anomalies = daily[mask].copy()
anomalies["resid"] = resid[mask]
anomalies["type"] = anomalies["resid"].gt(0).map(
    {True: "більше за норму", False: "менше за норму"})
anomalies.to_csv(out("anomalies.csv"))
print(anomalies["type"].value_counts())

# 7. Ковзна перевірка: багато відрізків по 14 днів
y = daily["count"]   
H = 14
STEP = 14
MIN_TRAIN = 365

rows = []
for end in range(MIN_TRAIN, len(y) - H + 1, STEP):
    train = y.iloc[:end]
    test = y.iloc[end:end + H]
    try:
        f_trend = ExponentialSmoothing(train, trend="add", seasonal="add",
                                       seasonal_periods=7).fit().forecast(H)
        f_season = ExponentialSmoothing(train, trend=None, seasonal="add",
                                        seasonal_periods=7).fit().forecast(H)
    except Exception:
        continue
    naive = train.iloc[-7:].mean()
    rows.append({
        "forecast_start": test.index[0].date(),
        "mae_ets_trend": np.abs(test.values - f_trend.values).mean(),
        "mae_ets_season": np.abs(test.values - f_season.values).mean(),
        "mae_naive": np.abs(test.values - naive).mean(),
    })

cv = pd.DataFrame(rows)
cv.to_csv(out("rolling_validation.csv"), index=False)

print(f"Кількість відрізків перевірки: {len(cv)} (по {H} днів)")
print(cv[["mae_ets_trend", "mae_ets_season", "mae_naive"]].agg(["mean", "median"]).round(2))
for col in ["mae_ets_trend", "mae_ets_season"]:
    share = (cv[col] < cv["mae_naive"]).mean() * 100
    print(f"{col} точніша за наївний прогноз у {share:.0f}% відрізків")

cv.set_index("forecast_start")[["mae_ets_trend", "mae_ets_season", "mae_naive"]].plot(
    figsize=(12, 4), alpha=0.8)
plt.title(f"MAE на кожному відрізку перевірки, {REGION}")
plt.ylabel("MAE (тривог на добу)")
plt.savefig(out("rolling_validation.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"Готово. Результати збережено в outputs/ з префіксом {PREFIX}_")