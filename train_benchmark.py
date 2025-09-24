#!/usr/bin/env python3
import argparse, json, math, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- helpers ----------
def norm_duration(s: str) -> str:
    if not s: return ""
    s = str(s).strip().lower()
    s = s.replace("hours","hour").replace("hrs","hr").replace(" ", "")
    if s in ("1h","1hr","1hour"):  return "1h"
    if s in ("1d","1day"):         return "1 day"
    if s in ("1w","1wk","1week"):  return "1 week"
    if s in ("1m","1mo","1month"): return "1 month"
    return s

def num(x):
    if x is None: return np.nan
    try:
        return float(str(x).replace("$","").strip())
    except Exception:
        return np.nan

TS_CANDIDATES = ["fetched_at_utc","timestamp","fetched_at","when","ts","collected_at_utc"]
PRICE_CANDS   = ["effective_price_usd_per_gpu_hr","price_hourly_usd","price_usd","price","$/gpu-hr"]
GPU_CANDS     = ["gpu_model","gpu","model"]
PROV_CANDS    = ["provider","vendor"]
TYPE_CANDS    = ["type","instance_type","category"]
REGION_CANDS  = ["region","used_region","geo"]
DUR_CANDS     = ["duration","term","runtime"]
COUNT_CANDS   = ["gpu_count","count","gpus"]

def load_history_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    def pick(cols):
        for c in cols:
            if c in df.columns: return c
        return None
    ts_col   = pick(TS_CANDIDATES)
    pr_col   = pick(PRICE_CANDS)
    gpu_col  = pick(GPU_CANDS)
    prov_col = pick(PROV_CANDS)
    type_col = pick(TYPE_CANDS)
    reg_col  = pick(REGION_CANDS)
    dur_col  = pick(DUR_CANDS)
    cnt_col  = pick(COUNT_CANDS)

    if ts_col is None or pr_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
        "price": df[pr_col].map(num)
    })
    out["provider"] = df[prov_col] if prov_col in df.columns else path.stem
    out["gpu"]      = df[gpu_col] if gpu_col in df.columns else None
    out["type"]     = df[type_col] if type_col in df.columns else None
    out["region"]   = df[reg_col] if reg_col in df.columns else None
    out["duration"] = df[dur_col].map(norm_duration) if dur_col in df.columns else None
    out["count"]    = pd.to_numeric(df[cnt_col], errors="coerce") if cnt_col in df.columns else np.nan
    out = out.dropna(subset=["ts","price"])
    return out

def winsorize(s: pd.Series, lo=0.01, hi=0.99):
    if s.size < 5: return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=ql, upper=qh)

def hourly_index(df_slice: pd.DataFrame) -> pd.Series:
    # median across providers per hour
    s = (df_slice
         .set_index("ts")
         .groupby(pd.Grouper(freq="1H"))["price"]
         .median()
         .dropna())
    return s

def ewma_log(s: pd.Series, halflife_hours: float = 12.0) -> pd.Series:
    # EWMA on log-price, back to level
    sl = np.log(s)
    ew = sl.ewm(halflife=halflife_hours, min_periods=3).mean()
    return np.exp(ew)

def quantiles(arr: np.ndarray, qs=(0.1,0.25,0.5,0.75,0.9)):
    if len(arr)==0: return [np.nan]*len(qs)
    return [float(np.quantile(arr, q)) for q in qs]

# ---------- core ----------
def train_slice(df_all: pd.DataFrame, gpu: str, region: str, typ: str, duration: str, count: int,
                lookback_days=14, min_points=24):
    # Filter to lookback
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=lookback_days)
    df = df_all[df_all["ts"] >= cutoff].copy()

    # Normalize filters
    duration = norm_duration(duration)
    cnt_str  = None if count is None else str(int(count))

    # strict slice
    f = (df["gpu"].astype(str)==gpu) & \
        (df["type"].astype(str).str.lower()==str(typ).lower())
    if region and region.lower()!="global":
        f &= (df["region"].astype(str)==region)
    if duration:
        f &= (df["duration"].astype(str).str.lower()==duration.lower())
    if cnt_str:
        f &= (df["count"].astype("Int64").astype(str)==cnt_str)

    df_slice = df[f].copy()
    # relax if too small
    note = "exact"
    if df_slice.shape[0] < min_points:
        df_slice = df[(df["gpu"].astype(str)==gpu) & (df["type"].astype(str).str.lower()==str(typ).lower())].copy()
        note = "relaxed(gpu+type)"

    if df_slice.empty:
        return None, {"note":"no-data"}

    # clean
    df_slice["price"] = winsorize(df_slice["price"])
    # build index
    idx = hourly_index(df_slice)
    if idx.size < 8:
        return None, {"note":"too-few-points", "n":int(idx.size)}

    # smooth
    idx_sm = ewma_log(idx, halflife_hours=12)
    # residuals on log-scale
    res_log = np.log(idx[idx_sm.index]) - np.log(idx_sm.reindex(idx.index))
    res_log = res_log.replace([np.inf,-np.inf], np.nan).dropna()
    if res_log.size < 6:
        return None, {"note":"few-residuals", "n":int(res_log.size)}

    # drift (local trend) on log-scale
    dlog = np.log(idx_sm).diff().dropna()
    if dlog.size:
        # emphasize last day
        drift = float(dlog.ewm(halflife=12, min_periods=3).mean().iloc[-1])
    else:
        drift = 0.0

    # snapshot quantiles (now)
    last_level = float(idx_sm.iloc[-1])
    q_res = np.quantile(res_log, [0.10,0.25,0.50,0.75,0.90])
    p10,p25,p50,p75,p90 = [float(math.exp(math.log(last_level)+q)) for q in q_res]

    # forecast next 72h hourly
    horizon = 72
    base0 = math.log(last_level)
    qs = [0.10,0.25,0.50,0.75,0.90]
    res_q = [float(np.quantile(res_log, q)) for q in qs]
    future = []
    start = pd.Timestamp.utcnow().tz_localize("UTC").floor("H")
    for h in range(1, horizon+1):
        t = start + pd.Timedelta(hours=h)
        mu = base0 + drift*h      # mean log-price path
        vals = [math.exp(mu + z) for z in res_q]
        future.append({
            "date_utc": t.isoformat(),
            "horizon_hr": h,
            "p10": vals[0], "p25": vals[1], "p50": vals[2], "p75": vals[3], "p90": vals[4]
        })

    meta = {
        "note": note,
        "n_raw": int(df_slice.shape[0]),
        "n_hourly": int(idx.size),
        "residuals_n": int(res_log.size),
        "drift_per_hour_log": drift
    }

    snapshot = {"p10":p10, "p25":p25, "p50":p50, "p75":p75, "p90":p90}
    return (snapshot, future), meta

def main():
    ap = argparse.ArgumentParser(description="Train simple EWMA+bootstrap price bands from history")
    ap.add_argument("--history-root", default="docs/data/history", help="folder with *_history.csv")
    ap.add_argument("--out-root", default="docs/data/derived", help="where to write outputs")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--region", default="Global")
    ap.add_argument("--type", dest="typ", default="On-Demand")
    ap.add_argument("--duration", default="1h")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--lookback-days", type=int, default=14)
    args = ap.parse_args()

    hist_dir = Path(args.history_root)
    out_dir  = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load all history
    files = list(hist_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No history files in {hist_dir}")

    parts = [load_history_table(p) for p in files]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit("No valid rows parsed from history.")
    df_all = pd.concat(parts, ignore_index=True)
    # normalize some fields
    df_all["duration"] = df_all["duration"].map(norm_duration)
    # training
    out, meta = train_slice(
        df_all,
        gpu=args.gpu, region=args.region, typ=args.typ, duration=args.duration, count=args.count,
        lookback_days=args.lookback_days
    )

    as_of = pd.Timestamp.utcnow().tz_localize("UTC").isoformat()
    if not out:
        # write a minimal snapshot so UI doesn't break
        snap = {
            "gpu_model": args.gpu, "region": args.region, "type": args.typ,
            "duration": args.duration, "gpu_count": args.count,
            "asof_utc": as_of, "note": meta.get("note","no-data")
        }
        (out_dir / "price_predict_snapshot.json").write_text(json.dumps(snap, indent=2))
        print("No forecast produced:", meta)
        return

    snapshot, future = out

    # write snapshot JSON
    snap = {
        "gpu_model": args.gpu, "region": args.region, "type": args.typ,
        "duration": args.duration, "gpu_count": args.count,
        "asof_utc": as_of, **snapshot, **meta
    }
    (out_dir / "price_predict_snapshot.json").write_text(json.dumps(snap, indent=2))

    # write forecast CSV
    fdf = pd.DataFrame(future)
    fdf.insert(0,"gpu_model", args.gpu)
    fdf.insert(1,"region", args.region)
    fdf.insert(2,"type", args.typ)
    fdf.insert(3,"duration", args.duration)
    fdf.insert(4,"gpu_count", args.count)
    fdf.to_csv(out_dir / "forecast.csv", index=False)

    print("Wrote:")
    print(" -", out_dir / "price_predict_snapshot.json")
    print(" -", out_dir / "forecast.csv")

if __name__ == "__main__":
    main()
