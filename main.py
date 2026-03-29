from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
import io
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.ai_analyst import AIAnalyst
from src.rag_engine import RAGEngine

# Opsiyonel filtre importu (projede sınıf adı değişken olabilir)
try:
    from src import filters as _filters_module  # noqa: F401
except Exception:
    _filters_module = None


app = FastAPI(title="TUA-ASTRO Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_analyst = AIAnalyst()
rag_engine = RAGEngine()


def _pick_numeric_telemetry_column(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("CSV içinde analiz edilebilir sayısal (numeric) kolon bulunamadı.")
    return numeric_cols[0]


def _compute_saa_status(df: pd.DataFrame) -> bool:
    """
    Basit SAA bayrağı:
    - lat/lon kolonları varsa son örneğe göre yaklaşık SAA bounding-box kontrolü yapar.
    - yoksa False döner.
    """
    lower_map = {c.lower(): c for c in df.columns}
    lat_col: Optional[str] = lower_map.get("lat") or lower_map.get("latitude")
    lon_col: Optional[str] = lower_map.get("lon") or lower_map.get("longitude")

    if not lat_col or not lon_col:
        return False

    try:
        lat = float(df[lat_col].dropna().iloc[-1])
        lon = float(df[lon_col].dropna().iloc[-1])

        # Yaklaşık SAA kutusu (temsilî): lat [-50, 10], lon [-90, -20]
        in_saa = (-50.0 <= lat <= 10.0) and (-90.0 <= lon <= -20.0)
        return bool(in_saa)
    except Exception:
        return False


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Lütfen .csv uzantılı bir dosya yükleyin.")

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV okuma hatası: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV dosyası satır içermiyor.")

    try:
        telemetry_col = _pick_numeric_telemetry_column(df)
        telemetry_series = df[telemetry_col].astype(float)

        stuck_mask, seu_mask, ai_report = ai_analyst.analyze(telemetry_series)

        stuck_indices = np.where(stuck_mask)[0].tolist()
        seu_indices = np.where(seu_mask)[0].tolist()

        analysis_payload = {
            "raw_report": ai_report,
            "stuck_count": len(stuck_indices),
            "seu_count": len(seu_indices),
            "stuck_mask": stuck_mask.tolist(),
            "seu_mask": seu_mask.tolist(),
        }

        expert_report = rag_engine.generate_expert_report(analysis_payload)
        status_flag = _compute_saa_status(df)

        return {
            "telemetry_data": telemetry_series.tolist(),
            "seu_indices": seu_indices,
            "stuck_indices": stuck_indices,
            "expert_report": expert_report,
            "status": status_flag,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analiz servis hatası: {exc}",
        ) from exc
