"""
PySpark Weather Data Processing & Visualization Pipeline
=========================================================
Processes weather CSV files from multiple Philippine stations,
performs aggregations, and generates insightful visualizations.

Usage:
    spark-submit weather_pipeline.py
    # or run interactively in PySpark shell / Jupyter
"""

import os
import glob
import re

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, TimestampType,
    DoubleType, IntegerType,
)

# ──────────────────────────────────────────────
# 0.  CONFIG
# ──────────────────────────────────────────────
DATA_DIR    = "./csv"          # folder that holds all CSV files
OUTPUT_DIR  = "./output_results"    # aggregated CSVs & chart images
import shutil
if os.path.exists(OUTPUT_DIR) and not os.path.isdir(OUTPUT_DIR):
    os.remove(OUTPUT_DIR)  # remove if it was created as a file by mistake
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = [                     # dark minimalist palette
    "#1a1a2e", "#16213e", "#0f3460",
    "#533483", "#2b2d42", "#5c6b7a",
    "#8d99ae", "#3d405b",
]

# ──────────────────────────────────────────────
# 1.  SPARK SESSION
# ──────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("WeatherPipeline")
        .config("spark.sql.session.timeZone", "Asia/Manila")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


# ──────────────────────────────────────────────
# 2.  EXPLICIT SCHEMA  (avoids full-scan inference)
# ──────────────────────────────────────────────
WEATHER_SCHEMA = StructType([
    StructField("station_id",  StringType(),  True),
    StructField("datetime",    StringType(),  True),   # parsed later
    StructField("lat",         DoubleType(),  True),
    StructField("lon",         DoubleType(),  True),
    StructField("temp",        DoubleType(),  True),
    StructField("pressure",    DoubleType(),  True),
    StructField("humidity",    DoubleType(),  True),
    StructField("wind_speed",  DoubleType(),  True),
    StructField("wind_dir",    DoubleType(),  True),
    StructField("cloud_cover", DoubleType(),  True),
    StructField("visibility",  DoubleType(),  True),
    StructField("rain_3h",     DoubleType(),  True),
])


# ──────────────────────────────────────────────
# 3.  DATA LOADING
# ──────────────────────────────────────────────
def extract_location(filepath: str) -> str:
    """
    Derive a human-readable location label from the filename.
    'path/to/DecodedCamNorte.csv'  →  'CamNorte'
    Falls back to the bare stem if the pattern doesn't match.
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r"Decoded(.+)", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def load_all_csvs(spark: SparkSession, data_dir: str):
    """
    Load every CSV in *data_dir*, add a `location` column derived from
    the filename, and union all DataFrames into one.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

    frames = []
    for path in csv_files:
        location = extract_location(path)
        df = (
            spark.read
            .option("header", "true")
            .option("nullValue", "")
            .schema(WEATHER_SCHEMA)
            .csv(path)
            .withColumn("location", F.lit(location))
        )
        frames.append(df)
        print(f"  Loaded  {os.path.basename(path)}  →  location='{location}'  "
              f"rows={df.count()}")

    combined = frames[0]
    for df in frames[1:]:
        combined = combined.unionByName(df)

    return combined


# ──────────────────────────────────────────────
# 4.  CLEANING & PREPARATION
# ──────────────────────────────────────────────
NUMERIC_COLS = [
    "temp", "pressure", "humidity",
    "wind_speed", "wind_dir",
    "cloud_cover", "visibility", "rain_3h",
]

def clean(df):
    """
    • Parse datetime string → TimestampType
    • Cast numeric columns (already typed in schema; re-cast for safety)
    • Drop rows where all key measurements are null
    • Fill remaining nulls with 0 for rain_3h (missing = no rain)
    """
    df = df.withColumn(
        "ts",
        F.coalesce(
            F.try_to_timestamp(F.col("datetime"), F.lit("yyyy-MM-dd HH:mm:ss")),
            F.try_to_timestamp(F.col("datetime"), F.lit("yyyy-MM-dd HH:mm")),
            F.try_to_timestamp(F.col("datetime")),   # auto-detect fallback
        ),
    )

    # safety cast – no-op if already DoubleType
    for c in NUMERIC_COLS:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # drop rows where every measurement is null
    df = df.dropna(
        how="all",
        subset=["temp", "pressure", "humidity", "wind_speed"],
    )

    # rain null → 0  (sensor doesn't log zero-rain events)
    df = df.fillna({"rain_3h": 0.0})

    # extract date for daily aggregations
    df = df.withColumn("date", F.to_date(F.col("ts")))

    return df.cache()   # cache – reused by multiple aggregations


# ──────────────────────────────────────────────
# 5.  AGGREGATIONS
# ──────────────────────────────────────────────
def agg_daily_temp(df):
    """Average temperature per location per day."""
    return (
        df.groupBy("location", "date")
        .agg(F.round(F.avg("temp"), 2).alias("avg_temp"))
        .orderBy("location", "date")
    )

def agg_avg_humidity(df):
    """Average humidity per location."""
    return (
        df.groupBy("location")
        .agg(F.round(F.avg("humidity"), 2).alias("avg_humidity"))
        .orderBy("avg_humidity", ascending=False)
    )

def agg_total_rain(df):
    """Total rainfall per location."""
    return (
        df.groupBy("location")
        .agg(F.round(F.sum("rain_3h"), 2).alias("total_rain_mm"))
        .orderBy("total_rain_mm", ascending=False)
    )

def agg_wind_trend(df):
    """Average wind speed per location per day."""
    return (
        df.groupBy("location", "date")
        .agg(F.round(F.avg("wind_speed"), 2).alias("avg_wind_speed"))
        .orderBy("location", "date")
    )

def run_aggregations(df):
    print("\n[Aggregations]")
    daily_temp   = agg_daily_temp(df)
    avg_humidity = agg_avg_humidity(df)
    total_rain   = agg_total_rain(df)
    wind_trend   = agg_wind_trend(df)

    # Save via Pandas (avoids Hadoop/winutils requirement on Windows)
    for name, sdf in [
        ("daily_avg_temp",   daily_temp),
        ("avg_humidity",     avg_humidity),
        ("total_rain",       total_rain),
        ("wind_trend",       wind_trend),
    ]:
        out = os.path.join(OUTPUT_DIR, f"{name}.csv")
        sdf.toPandas().to_csv(out, index=False)
        print(f"  Saved → {out}")

    return daily_temp, avg_humidity, total_rain, wind_trend


# ──────────────────────────────────────────────
# 6.  VISUALIZATIONS
# ──────────────────────────────────────────────
def _colour_map(locations):
    return {loc: PALETTE[i % len(PALETTE)] for i, loc in enumerate(locations)}


# 6a. Line chart – temperature trend over time
def plot_temp_trends(daily_temp_sdf):
    pdf = daily_temp_sdf.toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])

    locations = sorted(pdf["location"].unique())
    cmap = _colour_map(locations)

    fig, ax = plt.subplots(figsize=(14, 5))
    for loc in locations:
        sub = pdf[pdf["location"] == loc].sort_values("date")
        ax.plot(sub["date"], sub["avg_temp"],
                label=loc, color=cmap[loc], linewidth=1.8, marker="o",
                markersize=3, alpha=0.85)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()

    ax.set_title("Daily Average Temperature by Location", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "temp_trends.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {path}")


# 6b. Bar chart – average humidity per location
def plot_humidity_bar(avg_humidity_sdf):
    pdf = avg_humidity_sdf.toPandas().sort_values("avg_humidity", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        pdf["location"], pdf["avg_humidity"],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(pdf))],
        edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    ax.set_title("Average Humidity by Location", fontsize=14, fontweight="bold")
    ax.set_xlabel("Location")
    ax.set_ylabel("Average Humidity (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "avg_humidity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {path}")


# 6c. Scatter plot – wind speed vs temperature
def plot_wind_vs_temp(df_spark):
    # sample to avoid overloaded scatter (≤ 5 000 pts)
    sample_pdf = (
        df_spark
        .select("location", "wind_speed", "temp")
        .dropna()
        .sample(fraction=0.3, seed=42)
        .limit(5000)
        .toPandas()
    )

    locations = sorted(sample_pdf["location"].unique())
    cmap = _colour_map(locations)

    fig, ax = plt.subplots(figsize=(10, 6))
    for loc in locations:
        sub = sample_pdf[sample_pdf["location"] == loc]
        ax.scatter(sub["wind_speed"], sub["temp"],
                   label=loc, color=cmap[loc],
                   alpha=0.55, s=18, edgecolors="none")

    ax.set_title("Wind Speed vs Temperature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "wind_vs_temp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {path}")


# 6d. Heatmap – correlation matrix of weather variables
def plot_correlation_heatmap(df_spark):
    corr_pdf = (
        df_spark
        .select(*NUMERIC_COLS)
        .dropna()
        .sample(fraction=0.5, seed=0)
        .limit(10000)
        .toPandas()
    )

    corr_matrix = corr_pdf.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
    # mask upper triangle for cleaner look
    import numpy as np
    mask_arr = mask.values.copy()
    mask_arr[np.triu_indices_from(mask_arr, k=1)] = True
    mask = pd.DataFrame(mask_arr, index=corr_matrix.index, columns=corr_matrix.columns)

    sns.heatmap(
        corr_matrix, mask=mask,
        annot=True, fmt=".2f", linewidths=0.5,
        cmap="mako", center=0,
        vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Weather Variable Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {path}")


def run_visualizations(daily_temp, avg_humidity, _total_rain, _wind_trend, raw_df):
    print("\n[Visualizations]")
    plot_temp_trends(daily_temp)
    plot_humidity_bar(avg_humidity)
    plot_wind_vs_temp(raw_df)
    plot_correlation_heatmap(raw_df)


# ──────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Philippine Weather Data Pipeline")
    print("=" * 60)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # --- Load ---
    print(f"\n[Loading CSVs from '{DATA_DIR}']")
    raw = load_all_csvs(spark, DATA_DIR)

    # --- Clean ---
    print("\n[Cleaning]")
    clean_df = clean(raw)
    print(f"  Total rows after cleaning: {clean_df.count():,}")

    # --- Aggregate ---
    daily_temp, avg_humidity, total_rain, wind_trend = run_aggregations(clean_df)

    # --- Visualize ---
    run_visualizations(daily_temp, avg_humidity, total_rain, wind_trend, clean_df)

    # --- Summary prints (small, so .show() is fine) ---
    print("\n[Sample Aggregations]")
    print("-- Daily Avg Temp (top 10) --")
    daily_temp.show(10, truncate=False)

    print("-- Avg Humidity per Location --")
    avg_humidity.show(truncate=False)

    print("-- Total Rainfall per Location --")
    total_rain.show(truncate=False)

    clean_df.unpersist()
    spark.stop()
    print("\nPipeline complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()