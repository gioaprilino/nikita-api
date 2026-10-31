"""Kontrak request/response API NIKITA."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransactionRequest(BaseModel):
    """Data mentah satu pasangan kunjungan FKTP-FKRTL pada hari yang sama."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "peserta_id": "DEMO-ANOMALI-001",
                "tanggal_lahir": "2002-01-13",
                "tanggal_kunjungan": "2024-02-06",
                "tanggal_pulang": "2024-02-08",
                "jenis_kelamin": 1,
                "tahun_sampel": 2024,
                "provinsi_peserta": 74,
                "kabupaten_peserta": 7405,
                "provinsi_fktp": 74,
                "kabupaten_fktp": 7405,
                "provinsi_fkrtl": 74,
                "kabupaten_fkrtl": 7471,
                "jenis_fktp": 1,
                "tipe_fktp": 1,
                "tingkat_fktp": 1,
                "poli_fktp": 1,
                "kepemilikan_fktp": 1,
                "jenis_fkrtl": 1,
                "tipe_fkrtl": 1,
                "tingkat_fkrtl": 1,
                "poli_fkrtl": 1,
                "kepemilikan_fkrtl": 1,
                "kelas_fkrtl": 1,
                "status_pulang_fkrtl": 1,
                "biaya_tagih": 5000000,
                "biaya_verifikasi": 4800000,
                "inacbgs_kelompok": 1,
                "inacbgs_tipe": 1,
                "inacbgs_spesifikasi": 1,
                "inacbgs_keparahan": 1,
                "jenis_kunjungan_fktp": 1,
                "diagnosa_fktp": "O42",
                "diagnosa_fkrtl": "O42",
                "frekuensi_hari_itu": 2,
            }
        },
    )

    peserta_id: str | None = Field(
        default=None, max_length=64, description="ID peserta untuk audit/log"
    )
    tanggal_lahir: date = Field(
        ..., description="Tanggal lahir peserta (PSTV03)"
    )
    tanggal_kunjungan: date = Field(
        ...,
        description="Tanggal kunjungan same-day FKTP/FKRTL (FKP03/FKL03)",
    )
    tanggal_pulang: date = Field(
        ..., description="Tanggal pulang FKRTL (FKL04)"
    )
    jenis_kelamin: Literal[1, 2] = Field(
        ..., description="1=Laki-laki, 2=Perempuan (PSTV05)"
    )
    tahun_sampel: int = Field(
        ..., ge=2000, le=2100, description="Tahun sampel/klaim (PSTV16)"
    )
    hubungan_keluarga: int = Field(default=99, ge=0, le=99, description="PSTV04")
    kelas_rawat: int = Field(default=99, ge=0, le=99, description="PSTV07")
    segmen_peserta: int = Field(default=99, ge=0, le=99, description="PSTV08")
    status_kepesertaan: int = Field(
        default=99, ge=0, le=99, description="PSTV17"
    )
    provinsi_peserta: int = Field(
        default=99, ge=0, le=99, description="PSTV09_NEW"
    )
    kabupaten_peserta: int = Field(
        default=9999, ge=0, le=9999, description="PSTV10"
    )
    provinsi_fktp: int = Field(
        default=99, ge=0, le=99, description="FKP05_NEW"
    )
    kabupaten_fktp: int = Field(
        default=9999, ge=0, le=9999, description="FKP06"
    )
    jenis_fktp: int = Field(default=99, ge=0, le=99, description="FKP08")
    tipe_fktp: int = Field(default=99, ge=0, le=99, description="FKP09")
    tingkat_fktp: int = Field(default=99, ge=0, le=99, description="FKP10")
    poli_fktp: int = Field(default=99, ge=0, le=99, description="FKP11")
    kepemilikan_fktp: int = Field(
        default=99, ge=0, le=99, description="FKP07"
    )
    provinsi_fkrtl: int = Field(
        default=99, ge=0, le=99, description="FKL05_NEW"
    )
    kabupaten_fkrtl: int = Field(
        default=9999, ge=0, le=9999, description="FKL06"
    )
    jenis_fkrtl: int = Field(default=99, ge=0, le=99, description="FKL08")
    tipe_fkrtl: int = Field(default=99, ge=0, le=99, description="FKL09")
    tingkat_fkrtl: int = Field(default=99, ge=0, le=99, description="FKL10")
    poli_fkrtl: int = Field(default=99, ge=0, le=99, description="FKL11")
    kepemilikan_fkrtl: int = Field(
        default=99, ge=0, le=99, description="FKL07"
    )
    kelas_fkrtl: int = Field(
        default=9, ge=0, le=99, description="FKL13; 9 atau 99=nilai hilang"
    )
    status_pulang_fkrtl: int = Field(
        default=99, ge=0, le=99, description="FKL14"
    )
    biaya_tagih: float = Field(..., ge=0, description="FKL47")
    biaya_verifikasi: float = Field(..., ge=0, description="FKL48")
    inacbgs_kelompok: int = Field(
        default=99, ge=0, le=99, description="FKL20"
    )
    inacbgs_tipe: int = Field(default=99, ge=0, le=99, description="FKL21")
    inacbgs_spesifikasi: int = Field(
        default=99, ge=0, le=99, description="FKL22"
    )
    inacbgs_keparahan: int = Field(
        default=99, ge=0, le=99, description="FKL23"
    )
    jenis_kunjungan_fktp: int = Field(
        default=99, ge=0, le=99, description="FKP22"
    )
    diagnosa_fktp: str | None = Field(
        default=None, max_length=10, description="Kode ICD-10 FKTP (FKP14A)"
    )
    diagnosa_fkrtl: str | None = Field(
        default=None, max_length=10, description="Kode ICD-10 FKRTL (FKL17A)"
    )
    frekuensi_hari_itu: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Jumlah kunjungan peserta pada tanggal tersebut (agregat harian)",
    )

    @field_validator("diagnosa_fktp", "diagnosa_fkrtl")
    @classmethod
    def rapikan_diagnosa(cls, nilai: str | None) -> str | None:
        if nilai is None:
            return None
        hasil = nilai.strip().upper()
        return hasil or None

    @model_validator(mode="after")
    def validasi_konsistensi(self) -> "TransactionRequest":
        if self.tanggal_lahir > self.tanggal_kunjungan:
            raise ValueError(
                "tanggal_lahir tidak boleh setelah tanggal_kunjungan"
            )
        if self.tanggal_pulang < self.tanggal_kunjungan:
            raise ValueError(
                "tanggal_pulang tidak boleh sebelum tanggal_kunjungan"
            )
        usia = (
            self.tanggal_kunjungan.year
            - self.tanggal_lahir.year
            - (
                (self.tanggal_kunjungan.month, self.tanggal_kunjungan.day)
                < (self.tanggal_lahir.month, self.tanggal_lahir.day)
            )
        )
        if usia > 130:
            raise ValueError("usia peserta melebihi 130 tahun")
        return self


class FeatureVector(BaseModel):
    """Vektor 41 fitur hasil preprocessing; urutan sama dengan features.json."""

    model_config = ConfigDict(extra="forbid")

    is_spatial_anomaly: bool
    is_gender_mismatch: bool
    is_age_mismatch: bool
    frekuensi_hari_itu: int = Field(ge=1, le=100)
    biaya_ratio: float = Field(ge=0)
    usia_saat_kunjungan: int = Field(ge=0, le=130)
    jenis_kelamin: int = Field(ge=-1, le=2)
    hubungan_keluarga: int = Field(ge=-1, le=99)
    kelas_rawat: int = Field(ge=-1, le=99)
    segmen_peserta: int = Field(ge=-1, le=99)
    status_kepesertaan: int = Field(ge=-1, le=99)
    tahun_sampel: int = Field(ge=2000, le=2100)
    provinsi_peserta: int = Field(ge=-1, le=99)
    kabupaten_peserta: int = Field(ge=-1, le=9999)
    provinsi_fktp: int = Field(ge=-1, le=99)
    kabupaten_fktp: int = Field(ge=-1, le=9999)
    provinsi_fkrtl: int = Field(ge=-1, le=99)
    kabupaten_fkrtl: int = Field(ge=-1, le=9999)
    jenis_fktp: int = Field(ge=-1, le=99)
    tipe_fktp: int = Field(ge=-1, le=99)
    tingkat_fktp: int = Field(ge=-1, le=99)
    poli_fktp: int = Field(ge=-1, le=99)
    kepemilikan_fktp: int = Field(ge=-1, le=99)
    jenis_fkrtl: int = Field(ge=-1, le=99)
    tipe_fkrtl: int = Field(ge=-1, le=99)
    tingkat_fkrtl: int = Field(ge=-1, le=99)
    poli_fkrtl: int = Field(ge=-1, le=99)
    kepemilikan_fkrtl: int = Field(ge=-1, le=99)
    kelas_fkrtl: int = Field(ge=-1, le=9)
    status_pulang_fkrtl: int = Field(ge=-1, le=99)
    durasi_rawat: int = Field(ge=0, le=3650)
    bulan_kunjungan: int = Field(ge=1, le=12)
    hari_minggu: int = Field(ge=0, le=6)
    akhir_pekan: bool
    biaya_tagih: float = Field(ge=0)
    biaya_verifikasi: float = Field(ge=0)
    inacbgs_kelompok: int = Field(ge=-1, le=99)
    inacbgs_tipe: int = Field(ge=-1, le=99)
    inacbgs_spesifikasi: int = Field(ge=-1, le=99)
    inacbgs_keparahan: int = Field(ge=-1, le=99)
    jenis_kunjungan_fktp: int = Field(ge=-1, le=99)


class PredictionResponse(BaseModel):
    """Hasil prediksi satu transaksi."""

    skor_risiko: float = Field(
        ge=0,
        le=1,
        description=(
            "Probabilitas risiko terkalibrasi (isotonic). Pada artefak demo ini "
            "bersifat biner 0/1 karena weak label deterministik dan perfect "
            "separation; gunakan skor_risiko_relatif untuk visualisasi ordinal."
        ),
    )
    skor_margin: float = Field(
        description=(
            "Log-odds mentah dari booster XGBoost "
            "(booster.predict(DMatrix, output_margin=True)). Bukan probabilitas; "
            "berguna untuk ranking dan diagnosis."
        )
    )
    skor_risiko_relatif: float = Field(
        ge=0,
        le=1,
        description=(
            "Sigmoid(margin/4), skor ordinal 0-1 untuk visualisasi dashboard. "
            "Bukan probabilitas terkalibrasi; jangan dipakai sebagai pengganti "
            "skor_risiko dalam keputusan kebijakan."
        ),
    )
    status: Literal["Tinggi", "Sedang", "Rendah"] = Field(
        description=(
            "Kategori risiko berdasarkan ambang kebijakan dan skor_risiko "
            "terkalibrasi (bukan skor_risiko_relatif)."
        )
    )
    alasan: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Top-3 fitur dengan |SHAP| terbesar",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "skor_risiko": 1.0,
                "skor_margin": 10.561622,
                "skor_risiko_relatif": 0.933417,
                "status": "Tinggi",
                "alasan": [
                    "Inkonsistensi gender dengan diagnosis = ya (kontribusi +4.618, menaikkan risiko)",
                    "Frekuensi kunjungan pada hari yang sama = 2 (kontribusi +4.269, menaikkan risiko)",
                    "Anomali spasial (kabupaten FKTP dan FKRTL berbeda) = ya (kontribusi +1.378, menaikkan risiko)",
                ],
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Dokumentasi publik tentang metode dan keterbatasan model."""

    metode_label: Literal["weak_supervision_deterministic"] = Field(
        description="Cara label training diturunkan"
    )
    kalibrasi: Literal["isotonic"] = Field(
        description="Metode kalibrasi probabilitas yang dipakai artefak"
    )
    catatan: str = Field(
        description=(
            "Penjelasan transparan tentang keterbatasan skor dan rekomendasi "
            "penggunaan untuk demo/produksi"
        )
    )


class HealthResponse(BaseModel):
    """Status kesiapan layanan."""

    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_sha256: str = Field(description="SHA-256 artefak model yang dimuat")
    shap_explainer_loaded: bool
    shap_mode: Literal["shap_treeexplainer", "native_xgboost"]
    jumlah_fitur: int
    preprocessing: str
    uptime_detik: float


class ErrorResponse(BaseModel):
    """Envelope error yang stabil untuk klien."""

    error: str
    detail: object


__all__ = [
    "ErrorResponse",
    "FeatureVector",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictionResponse",
    "TransactionRequest",
]
