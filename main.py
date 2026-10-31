"""NIKITA Enterprise API.

Model, kontrak fitur, dan jalur SHAP dimuat sekali saat startup. Preprocessing
memakai logika numerik yang sama dengan training; tidak ada LabelEncoder atau
OneHotEncoder karena model dilatih langsung pada kode kategori numerik.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from preprocessor import (
    FEATURES,
    PREPROCESSING_VERSION,
    PreprocessingError,
    buat_alasan,
    buat_dataframe,
    hitung_fitur,
)
from schemas import (
    ErrorResponse,
    FeatureVector,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.getenv("NIKITA_MODEL_DIR", str(BASE_DIR / "artifacts"))
).resolve()
MODEL_PATH = Path(
    os.getenv("NIKITA_MODEL_PATH", str(MODEL_DIR / "model_xgb_calibrated.pkl"))
)
SHAP_PATH = Path(
    os.getenv("NIKITA_SHAP_PATH", str(MODEL_DIR / "shap_explainer.pkl"))
)

AMBANG_TINGGI = float(os.getenv("NIKITA_AMBANG_TINGGI", "0.70"))
AMBANG_SEDANG = float(os.getenv("NIKITA_AMBANG_SEDANG", "0.40"))

if not (0.0 <= AMBANG_SEDANG < AMBANG_TINGGI <= 1.0):
    raise RuntimeError(
        "Ambang tidak valid: butuh 0 <= AMBANG_SEDANG < AMBANG_TINGGI <= 1"
    )

logger = logging.getLogger("nikita.api")


def _status(skor: float) -> str:
    if skor >= AMBANG_TINGGI:
        return "Tinggi"
    if skor >= AMBANG_SEDANG:
        return "Sedang"
    return "Rendah"


def _base_model(model):
    if hasattr(model, "calibrated_classifiers_"):
        estimator = model.calibrated_classifiers_[0].estimator
        return getattr(estimator, "estimator", estimator)
    return model


def _booster(model):
    if not hasattr(model, "get_booster"):
        raise RuntimeError("Model dasar bukan XGBoost; pred_contribs tidak tersedia")
    return model.get_booster()


def _shap_dari_explainer(explainer, X: pd.DataFrame) -> np.ndarray:
    nilai = explainer.shap_values(X)
    if isinstance(nilai, list):
        nilai = nilai[-1]
    return np.asarray(nilai, dtype=float).reshape(-1)


def _shap_native(booster, X: pd.DataFrame) -> np.ndarray:
    matriks = xgb.DMatrix(X, feature_names=list(X.columns))
    kontribusi = booster.predict(matriks, pred_contribs=True)
    return np.asarray(kontribusi[0, :-1], dtype=float)


def _margin_native(booster, X: pd.DataFrame) -> float:
    matriks = xgb.DMatrix(X, feature_names=list(X.columns))
    return float(booster.predict(matriks, output_margin=True)[0])


def _sigmoid(nilai: float) -> float:
    if nilai >= 0:
        eksp = math.exp(-nilai)
        return 1.0 / (1.0 + eksp)
    eksp = math.exp(nilai)
    return eksp / (1.0 + eksp)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mulai = time.monotonic()

    if tuple(FeatureVector.model_fields) != FEATURES:
        raise RuntimeError(
            "FeatureVector tidak sinkron dengan features.json"
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model tidak ditemukan: {MODEL_PATH}")

    model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    with warnings.catch_warnings(record=True) as ditangkap:
        warnings.simplefilter("always")
        model = joblib.load(MODEL_PATH)

    for peringatan in ditangkap:
        logger.warning("Peringatan memuat model: %s", peringatan.message)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Artefak model tidak memiliki predict_proba")

    jumlah_fitur = int(getattr(model, "n_features_in_", -1))
    if jumlah_fitur != len(FEATURES):
        raise RuntimeError(
            f"Model membutuhkan {jumlah_fitur} fitur, features.json berisi "
            f"{len(FEATURES)}"
        )

    dasar = _base_model(model)

    nama_fitur = getattr(model, "feature_names_in_", None)
    if nama_fitur is None:
        nama_fitur = getattr(dasar, "feature_names_in_", None)
    if nama_fitur is not None and [str(n) for n in nama_fitur] != list(FEATURES):
        raise RuntimeError(
            "Urutan fitur model tidak sama dengan features.json"
        )

    booster = _booster(dasar)
    if int(booster.num_features()) != len(FEATURES):
        raise RuntimeError(
            f"Booster memiliki {booster.num_features()} fitur, "
            f"features.json berisi {len(FEATURES)}"
        )

    explainer = None
    if SHAP_PATH.exists():
        try:
            explainer = joblib.load(SHAP_PATH)
        except Exception as exc:
            logger.warning(
                "Gagal memuat SHAP explainer (%s); memakai pred_contribs native",
                exc,
            )
    else:
        logger.warning(
            "SHAP explainer tidak ditemukan di %s; memakai pred_contribs native",
            SHAP_PATH,
        )

    X_pemanasan = pd.DataFrame(
        [[0.0] * len(FEATURES)], columns=list(FEATURES), dtype="float64"
    )

    try:
        model.predict_proba(X_pemanasan)
    except Exception as exc:
        raise RuntimeError(f"Warm-up model gagal: {exc}") from exc

    use_explainer = False
    if explainer is not None:
        try:
            _shap_dari_explainer(explainer, X_pemanasan)
            use_explainer = True
        except Exception as exc:
            logger.warning(
                "SHAP TreeExplainer tidak dapat dipakai (%s); fallback ke "
                "pred_contribs native XGBoost",
                exc,
            )

    app.state.model = model
    app.state.model_sha256 = model_sha256
    app.state.booster = booster
    app.state.explainer = explainer
    app.state.use_explainer = use_explainer
    app.state.shap_mode = (
        "shap_treeexplainer" if use_explainer else "native_xgboost"
    )
    app.state.started_at = mulai

    logger.info(
        "Model siap: %d fitur, SHAP=%s, sha256=%s, startup %.2f dtk",
        len(FEATURES),
        app.state.shap_mode,
        model_sha256[:12],
        time.monotonic() - mulai,
    )

    try:
        yield
    finally:
        app.state.model = None
        app.state.model_sha256 = ""
        app.state.booster = None
        app.state.explainer = None
        logger.info("Artefak model dilepas")


app = FastAPI(
    title="NIKITA - Identity Continuity Risk Score API",
    version="1.1.0",
    description=(
        "API skor risiko penyalahgunaan identitas peserta BPJS Kesehatan. "
        "Input adalah satu pasangan kunjungan FKTP-FKRTL pada hari yang sama; "
        "preprocessing menghasilkan 41 fitur numerik yang sama dengan training."
    ),
    openapi_tags=[
        {"name": "prediksi", "description": "Skoring risiko per transaksi"},
        {"name": "operasional", "description": "Health check dan kesiapan model"},
    ],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.exception_handler(RequestValidationError)
async def tangani_validasi(request: Request, exc: RequestValidationError):
    detail = [
        {k: v for k, v in kesalahan.items() if k not in ("ctx", "url", "input")}
        for kesalahan in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "validasi_input",
            "detail": jsonable_encoder(detail),
        },
    )


@app.exception_handler(PreprocessingError)
async def tangani_preprocessing(request: Request, exc: PreprocessingError):
    return JSONResponse(
        status_code=422,
        content={"error": "preprocessing", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def tangani_internal(request: Request, exc: Exception):
    logger.exception(
        "rid=%s kesalahan internal pada %s %s",
        getattr(request.state, "request_id", "-"),
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "Terjadi kesalahan internal",
        },
    )


@app.middleware("http")
async def catat_permintaan(request: Request, call_next):
    request_id = "".join(
        karakter
        for karakter in request.headers.get("X-Request-ID", "")
        if karakter.isalnum() or karakter in "-_."
    )[:64] or uuid.uuid4().hex
    request.state.request_id = request_id
    mulai = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "rid=%s %s %s -> %d (%.1f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - mulai) * 1000,
    )
    return response


@app.get("/", include_in_schema=False)
def akar():
    return {
        "nama": "NIKITA API",
        "versi": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/api/v1/model-info",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["operasional"],
    summary="Status kesiapan layanan dan model",
)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        model_sha256=state.model_sha256,
        shap_explainer_loaded=state.explainer is not None,
        shap_mode=state.shap_mode,
        jumlah_fitur=len(FEATURES),
        preprocessing=PREPROCESSING_VERSION,
        uptime_detik=round(time.monotonic() - state.started_at, 1),
    )


@app.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    tags=["operasional"],
    summary="Keterbatasan dan konteks penggunaan model",
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        metode_label="weak_supervision_deterministic",
        kalibrasi="isotonic",
        catatan=(
            "Model dilatih pada label sintetis yang diturunkan dari aturan. "
            "Skor kalibrasi bersifat biner (0/1) karena perfect separation. "
            "Gunakan skor_risiko_relatif untuk visualisasi ordinal. Untuk "
            "produksi, dibutuhkan label fraud riil dan rekalibrasi proper "
            "(Platt/temperature scaling)."
        ),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["prediksi"],
    summary="Hitung skor risiko penyalahgunaan identitas",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Input atau hasil preprocessing tidak valid",
        },
        500: {
            "model": ErrorResponse,
            "description": "Kesalahan internal layanan",
        },
    },
)
def predict(
    payload: TransactionRequest, request: Request
) -> PredictionResponse:
    fitur = hitung_fitur(payload.model_dump())

    try:
        vektor = FeatureVector(**fitur)
    except ValidationError as exc:
        raise PreprocessingError(
            f"Fitur hasil preprocessing tidak valid: {exc}"
        ) from exc

    X = buat_dataframe(vektor.model_dump())

    state = request.app.state
    skor = float(state.model.predict_proba(X)[0, 1])
    if not np.isfinite(skor):
        raise RuntimeError("Model menghasilkan skor tidak valid")

    margin = _margin_native(state.booster, X)
    if not np.isfinite(margin):
        raise RuntimeError("Model menghasilkan margin tidak valid")
    skor_relatif = _sigmoid(margin / 4.0)

    if state.use_explainer:
        shap_values = _shap_dari_explainer(state.explainer, X)
    else:
        shap_values = _shap_native(state.booster, X)

    if shap_values.size != len(FEATURES):
        raise RuntimeError(
            f"SHAP menghasilkan {shap_values.size} nilai, "
            f"butuh {len(FEATURES)}"
        )
    if not np.isfinite(shap_values).all():
        raise RuntimeError("SHAP menghasilkan nilai tidak valid")

    urutan = np.argsort(-np.abs(shap_values))[:3]
    alasan = [
        buat_alasan(
            FEATURES[int(i)],
            float(X.iloc[0, int(i)]),
            float(shap_values[int(i)]),
        )
        for i in urutan
    ]

    return PredictionResponse(
        skor_risiko=round(skor, 6),
        skor_margin=round(margin, 6),
        skor_risiko_relatif=round(skor_relatif, 6),
        status=_status(skor),
        alasan=alasan,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("NIKITA_WORKERS", "1")),
        reload=False,
    )
