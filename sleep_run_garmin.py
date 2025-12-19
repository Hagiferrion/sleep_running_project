# sleep_run_garmin.py
# -----------------------------------------
# Garmin Sleep -> Next-day Running analysis
# -----------------------------------------

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Config
# -----------------------------
DATA_DIR  = "data"
SLEEP_CSV = "sleep_total_1017_1219_FEATURES.csv"
RUN_CSV   = "Activities_Running_Only.csv"
OUT_DIR   = "outputs"
RANDOM_STATE = 42

def project_root() -> str:
    # sleep_run_garmin.py가 src/에 있다고 가정하면, 프로젝트 루트는 한 단계 위
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sleep_path = os.path.join(root, "data", SLEEP_CSV)
    run_path   = os.path.join(root, "data", RUN_CSV)

    sleep = pd.read_csv(sleep_path)
    run   = pd.read_csv(run_path)
    return sleep, run


TARGETS = {
    # 러닝 CSV의 원본 컬럼명(한글/기호) -> 내부 타깃명
    "평균 속도": "avg_speed",
    "평균 심박수": "avg_hr",
    "Training Stress Score®": "tss",
}

RUN_RENAME_MAP = {
    "날짜": "date",
    "거리": "distance",
    "시간": "duration",
    "평균 속도": "avg_speed",
    "평균 심박수": "avg_hr",
    "Training Stress Score®": "tss",
}

def time_str_to_seconds(x):
    """
    'MM:SS' or 'HH:MM:SS' -> seconds
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return x

    parts = str(x).split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None

    if len(parts) == 2:      # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:    # HH:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        return None

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sleep = pd.read_csv(os.path.join(DATA_DIR, SLEEP_CSV))
    run   = pd.read_csv(os.path.join(DATA_DIR, RUN_CSV))
    return sleep, run


SHIFT_SLEEP_BY_1DAY = False  # True면 전날 수면(D-1) -> 다음날 러닝(D)

def preprocess_sleep(sleep: pd.DataFrame) -> pd.DataFrame:
    sleep = sleep.copy()
    if "date" not in sleep.columns:
        raise ValueError("sleep CSV에 'date' 컬럼이 없습니다.")

    sleep["date"] = safe_to_datetime(sleep["date"])
    sleep = sleep.dropna(subset=["date"])

    # (선택) 전날 수면을 다음날 러닝에 매칭하고 싶을 때만 사용
    if SHIFT_SLEEP_BY_1DAY:
        sleep["date"] = sleep["date"] + pd.Timedelta(days=1)

    # ★핵심: 시간 제거해서 날짜만 남기기 (merge 성공률 1순위)
    sleep["date"] = sleep["date"].dt.normalize()

    # Feature engineering
    if "sleep_need_min" in sleep.columns and "duration_min" in sleep.columns:
        sleep["sleep_debt"] = sleep["sleep_need_min"] - sleep["duration_min"]
    else:
        sleep["sleep_debt"] = np.nan

    if "duration_min" in sleep.columns:
        sleep["short_sleep"] = (sleep["duration_min"] < 360).astype(int)  # < 6h
    else:
        sleep["short_sleep"] = np.nan

    if "sleep_start_hour" in sleep.columns:
        sleep["late_sleep"] = (sleep["sleep_start_hour"] > 24).astype(int)
    else:
        sleep["late_sleep"] = np.nan

    return sleep



def preprocess_run(run: pd.DataFrame) -> pd.DataFrame:
    run = run.copy()

    if "날짜" not in run.columns and "date" not in run.columns:
        raise ValueError("running CSV에 '날짜' 또는 'date' 컬럼이 없습니다.")

    run = run.rename(columns=RUN_RENAME_MAP)
    run["date"] = safe_to_datetime(run["date"])
    run = run.dropna(subset=["date"])
    run["date"] = run["date"].dt.normalize()


    # -----------------------------
    # [핵심 추가] "4:27" 같은 pace/time 문자열 처리
    # -----------------------------
    def time_str_to_seconds(x):
        # 'MM:SS' or 'HH:MM:SS' -> seconds
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (int, float)):
            return float(x)

        s = str(x).strip()
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return pd.NA

        if len(parts) == 2:      # MM:SS
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:      # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return pd.NA

    # avg_speed 컬럼이 숫자가 아니라 문자열(예: '4:27')이면 pace로 간주하고 변환
    if "avg_speed" in run.columns:
        if run["avg_speed"].dtype == "object":
            # 1) pace 문자열 -> 초(= seconds per km)로 변환 (가정: '분:초 / km')
            run["avg_pace_sec_per_km"] = run["avg_speed"].apply(time_str_to_seconds)

            # 2) speed로도 만들기 (km/h).  pace(sec/km) -> km/h = 3600 / pace
            run["avg_speed_kmh"] = 3600 / run["avg_pace_sec_per_km"]
        else:
            # 숫자면 그대로 사용(단위는 데이터에 따라 km/h 또는 m/s일 수 있음)
            run["avg_speed_num"] = pd.to_numeric(run["avg_speed"], errors="coerce")

    # -----------------------------
    # 하루 단위 요약
    # -----------------------------
    agg_map = {}
    # avg_speed는 "숫자 컬럼"만 평균내도록 변경
    if "avg_speed_kmh" in run.columns:
        agg_map["avg_speed_kmh"] = "mean"
    elif "avg_speed_num" in run.columns:
        agg_map["avg_speed_num"] = "mean"

    if "avg_hr" in run.columns:
        agg_map["avg_hr"] = "mean"
    if "tss" in run.columns:
        agg_map["tss"] = "sum"
    if "distance" in run.columns:
        agg_map["distance"] = "sum"

    # 안전장치: agg 대상이 하나도 없으면 에러
    if not agg_map:
        raise ValueError("집계할 컬럼이 없습니다. RUN_RENAME_MAP 및 CSV 컬럼을 확인하세요.")

    run_day = run.groupby("date", as_index=False).agg(agg_map)
    if "avg_speed_kmh" in run_day.columns:
        run_day["avg_speed"] = run_day["avg_speed_kmh"]
    elif "avg_speed_num" in run_day.columns:
        run_day["avg_speed"] = run_day["avg_speed_num"]

    return run_day


def merge_sleep_run(sleep: pd.DataFrame, run_day: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(run_day, sleep, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_feature_list(df: pd.DataFrame) -> list[str]:
    # 기본 수면 feature 후보들 (존재하는 것만 사용)
    candidates = [
        "score",
        "duration_min",
        "sleep_need_min",
        "sleep_debt",
        "resting_hr",
        "pulse_ox",
        "sleep_start_hour",
        "short_sleep",
        "late_sleep",
        "weekday",
        "is_weekend",
        "sleep_score_4w",
        "sleep_duration_4w",
    ]
    return [c for c in candidates if c in df.columns]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # squared 인자 사용 안 함
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def plot_pred_vs_true(y_true: pd.Series, y_pred: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(y_true, y_pred, alpha=0.75)
    ymin = float(min(y_true.min(), np.min(y_pred)))
    ymax = float(max(y_true.max(), np.max(y_pred)))
    plt.plot([ymin, ymax], [ymin, ymax], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_feature_importance(importances: pd.Series, title: str, outpath: str) -> None:
    # 수평 bar plot
    imp = importances.sort_values(ascending=True)
    plt.figure(figsize=(6.5, max(3.5, 0.35 * len(imp))))
    plt.barh(imp.index, imp.values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def train_and_report(df: pd.DataFrame, features: list[str], target_col: str) -> dict:
    dfm = df[["date"] + features + [target_col]].dropna().copy()
    if len(dfm) < 15:
        return {"error": f"유효 데이터가 너무 적습니다: {len(dfm)} rows (target={target_col})"}

    X = dfm[features]
    y = dfm[target_col]

    # 시간 순서를 지키는 split (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            random_state=RANDOM_STATE
        ),
    }

    out = {"rows_used": int(len(dfm)), "target": target_col, "features": features, "models": {}}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        out["models"][name] = eval_metrics(y_test.values, pred)

        # pred vs true plot
        plot_path = os.path.join(OUT_DIR, f"pred_vs_true__{target_col}__{name}.png")
        plot_pred_vs_true(y_test, pred, f"{target_col} | {name}", plot_path)

        # feature importance for RF
        if name == "RandomForest":
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            imp_csv = os.path.join(OUT_DIR, f"feature_importance__{target_col}__{name}.csv")
            importances.to_csv(imp_csv, header=["importance"])
            imp_png = os.path.join(OUT_DIR, f"feature_importance__{target_col}__{name}.png")
            plot_feature_importance(importances, f"Feature Importance ({target_col})", imp_png)

    # metrics table csv
    metrics_rows = []
    for m, met in out["models"].items():
        row = {"target": target_col, "model": m, **met}
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, f"metrics__{target_col}.csv"), index=False)

    return out


def main() -> None:
    ensure_outdir(OUT_DIR)

    sleep, run = load_data()
    sleep = preprocess_sleep(sleep)
    run_day = preprocess_run(run)

   

    df = merge_sleep_run(sleep, run_day)

    # 저장: 병합된 원본(검증용)
    df.to_csv(os.path.join(OUT_DIR, "merged_sleep_run.csv"), index=False)

    features = get_feature_list(df)
    if not features:
        raise ValueError("사용 가능한 수면 feature가 없습니다. (score/duration_min 등 컬럼 확인 필요)")

    summary = {
        "merged_rows": int(len(df)),
        "features_used": features,
        "targets": {},
    }

    for raw_name, tcol in TARGETS.items():
        if tcol in df.columns:
            summary["targets"][tcol] = train_and_report(df, features, tcol)
        else:
            summary["targets"][tcol] = {"error": f"러닝 데이터에 타깃 컬럼이 없습니다: {tcol} (원본명: {raw_name})"}

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"[Saved] {os.path.join(OUT_DIR, 'merged_sleep_run.csv')}")
    print(f"[Saved] {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"Features: {features}")
    print("Targets processed:", ", ".join(summary["targets"].keys()))


if __name__ == "__main__":
    main()
