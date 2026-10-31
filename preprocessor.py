"""Preprocessing fitur NIKITA untuk training dan serving.

Tidak ada artefak LabelEncoder/OneHotEncoder. Model dilatih langsung pada kode
numerik kategori (nilai hilang dipetakan ke -1) dan XGBoost mempelajari split
numerik pada kode tersebut. Modul ini adalah implementasi tunggal preprocessing
yang dipakai training dan API; menambahkan encoder baru akan mengubah 41 fitur
dan membuat model menolak input.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PREPROCESSING_VERSION = "numeric-codec-v1"

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = Path(
    os.getenv(
        "NIKITA_FEATURES_PATH",
        str(BASE_DIR / "artifacts" / "features.json"),
    )
)

with FEATURES_PATH.open(encoding="utf-8") as _berkas:
    _fitur = json.load(_berkas)

if (
    not isinstance(_fitur, list)
    or len(_fitur) != 41
    or len(set(_fitur)) != 41
    or not all(isinstance(nama, str) for nama in _fitur)
):
    raise RuntimeError(
        f"features.json tidak valid: butuh 41 nama fitur unik, dapat {_fitur!r}"
    )

FEATURES: tuple[str, ...] = tuple(_fitur)

POLA_KEHAMILAN = ("O", "Z34", "Z37")
POLA_PROSTAT = ("N40", "C61")
POLA_DEGENERATIF = ("I10", "E11", "I25")
USIA_ANAK_TAHUN = 10


class PreprocessingError(ValueError):
    """Input tidak dapat diubah menjadi vektor 41 fitur."""


def _tanggal(nilai: Any) -> date | None:
    if isinstance(nilai, datetime):
        return nilai.date()
    if isinstance(nilai, date):
        return nilai
    if isinstance(nilai, str) and len(nilai) >= 10:
        try:
            return date.fromisoformat(nilai[:10])
        except ValueError:
            return None
    return None


def _usia_tahun(lahir: date | None, kunjungan: date | None) -> int:
    if lahir is None or kunjungan is None:
        return -1
    return (
        kunjungan.year
        - lahir.year
        - ((kunjungan.month, kunjungan.day) < (lahir.month, lahir.day))
    )


def _kode(nilai: Any, missing: tuple[int, ...] = (99,)) -> int:
    try:
        v = int(nilai)
    except (TypeError, ValueError):
        return -1
    return -1 if v in missing else v


def _angka(nilai: Any) -> float:
    try:
        v = float(nilai)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v


def _awal(kode: Any, prefixes: tuple[str, ...]) -> bool:
    return isinstance(kode, str) and kode.strip().upper().startswith(prefixes)


def _ambil(raw: Mapping[str, Any], *kunci: str, default: Any = None) -> Any:
    for k in kunci:
        if k in raw and raw[k] is not None:
            return raw[k]
    return default


def hitung_fitur(raw: Mapping[str, Any]) -> dict[str, int | float]:
    lahir = _tanggal(_ambil(raw, "tanggal_lahir", "PSTV03"))
    tanggal = _tanggal(
        _ambil(raw, "tanggal_kunjungan", "tanggal", "FKL03", "FKP03")
    )
    if lahir is None:
        raise PreprocessingError("tanggal_lahir wajib diisi dengan format YYYY-MM-DD")
    if tanggal is None:
        raise PreprocessingError(
            "tanggal_kunjungan wajib diisi dengan format YYYY-MM-DD"
        )

    masuk = tanggal
    pulang = _tanggal(_ambil(raw, "tanggal_pulang", "FKL04"))
    usia = _usia_tahun(lahir, tanggal)
    if usia < 0:
        raise PreprocessingError("tanggal_lahir tidak boleh setelah tanggal_kunjungan")
    if usia > 130:
        raise PreprocessingError("usia peserta melebihi 130 tahun")
    if pulang is not None and pulang < masuk:
        raise PreprocessingError(
            "tanggal_pulang tidak boleh sebelum tanggal_kunjungan"
        )

    dx_fktp = _ambil(raw, "diagnosa_fktp", "FKP14A")
    dx_fkrtl = _ambil(raw, "diagnosa_fkrtl", "FKL17A")
    sex = _kode(_ambil(raw, "jenis_kelamin", "PSTV05"))

    hamil = _awal(dx_fktp, POLA_KEHAMILAN) or _awal(dx_fkrtl, POLA_KEHAMILAN)
    prostat = _awal(dx_fktp, POLA_PROSTAT) or _awal(dx_fkrtl, POLA_PROSTAT)
    gender_mismatch = int((sex == 1 and hamil) or (sex == 2 and prostat))

    degeneratif = _awal(dx_fktp, POLA_DEGENERATIF) or _awal(
        dx_fkrtl, POLA_DEGENERATIF
    )
    age_mismatch = int(0 <= usia < USIA_ANAK_TAHUN and degeneratif)

    kab_fktp = _kode(_ambil(raw, "kabupaten_fktp", "FKP06"), missing=(9999, 99))
    kab_fkrtl = _kode(_ambil(raw, "kabupaten_fkrtl", "FKL06"), missing=(9999, 99))
    spatial = int(kab_fktp > 0 and kab_fkrtl > 0 and kab_fktp != kab_fkrtl)

    tagih = _angka(_ambil(raw, "biaya_tagih", "FKL47"))
    verifikasi = _angka(_ambil(raw, "biaya_verifikasi", "FKL48"))
    if tagih < 0 or verifikasi < 0:
        raise PreprocessingError(
            "biaya_tagih dan biaya_verifikasi tidak boleh negatif"
        )
    biaya_ratio = round(verifikasi / tagih, 6) if tagih > 0 else 1.0

    frekuensi = max(
        1,
        _kode(
            _ambil(raw, "frekuensi_hari_itu", "frekuensi_kunjungan_hari_itu"),
            missing=(),
        ),
    )

    tahun = _kode(_ambil(raw, "tahun_sampel", "PSTV16"), missing=())
    if not (1900 <= tahun <= 2100):
        raise PreprocessingError("tahun_sampel wajib diisi (1900-2100)")

    durasi = (pulang - masuk).days if (masuk and pulang) else 0
    hari = tanggal.weekday()

    fitur: dict[str, int | float] = {
        "is_spatial_anomaly": spatial,
        "is_gender_mismatch": gender_mismatch,
        "is_age_mismatch": age_mismatch,
        "frekuensi_hari_itu": frekuensi,
        "biaya_ratio": biaya_ratio,
        "usia_saat_kunjungan": usia,
        "jenis_kelamin": sex,
        "hubungan_keluarga": _kode(_ambil(raw, "hubungan_keluarga", "PSTV04")),
        "kelas_rawat": _kode(_ambil(raw, "kelas_rawat", "PSTV07")),
        "segmen_peserta": _kode(_ambil(raw, "segmen_peserta", "PSTV08")),
        "status_kepesertaan": _kode(
            _ambil(raw, "status_kepesertaan", "PSTV17")
        ),
        "tahun_sampel": tahun,
        "provinsi_peserta": _kode(
            _ambil(raw, "provinsi_peserta", "PSTV09_NEW")
        ),
        "kabupaten_peserta": _kode(
            _ambil(raw, "kabupaten_peserta", "PSTV10"), missing=(9999,)
        ),
        "provinsi_fktp": _kode(_ambil(raw, "provinsi_fktp", "FKP05_NEW")),
        "kabupaten_fktp": kab_fktp,
        "provinsi_fkrtl": _kode(_ambil(raw, "provinsi_fkrtl", "FKL05_NEW")),
        "kabupaten_fkrtl": kab_fkrtl,
        "jenis_fktp": _kode(_ambil(raw, "jenis_fktp", "FKP08")),
        "tipe_fktp": _kode(_ambil(raw, "tipe_fktp", "FKP09")),
        "tingkat_fktp": _kode(_ambil(raw, "tingkat_fktp", "FKP10")),
        "poli_fktp": _kode(_ambil(raw, "poli_fktp", "FKP11")),
        "kepemilikan_fktp": _kode(
            _ambil(raw, "kepemilikan_fktp", "FKP07")
        ),
        "jenis_fkrtl": _kode(_ambil(raw, "jenis_fkrtl", "FKL08")),
        "tipe_fkrtl": _kode(_ambil(raw, "tipe_fkrtl", "FKL09")),
        "tingkat_fkrtl": _kode(_ambil(raw, "tingkat_fkrtl", "FKL10")),
        "poli_fkrtl": _kode(_ambil(raw, "poli_fkrtl", "FKL11")),
        "kepemilikan_fkrtl": _kode(
            _ambil(raw, "kepemilikan_fkrtl", "FKL07")
        ),
        "kelas_fkrtl": _kode(
            _ambil(raw, "kelas_fkrtl", "FKL13"), missing=(9, 99)
        ),
        "status_pulang_fkrtl": _kode(
            _ambil(raw, "status_pulang_fkrtl", "FKL14")
        ),
        "durasi_rawat": durasi,
        "bulan_kunjungan": tanggal.month,
        "hari_minggu": hari,
        "akhir_pekan": int(hari >= 5),
        "biaya_tagih": tagih,
        "biaya_verifikasi": verifikasi,
        "inacbgs_kelompok": _kode(
            _ambil(raw, "inacbgs_kelompok", "FKL20")
        ),
        "inacbgs_tipe": _kode(_ambil(raw, "inacbgs_tipe", "FKL21")),
        "inacbgs_spesifikasi": _kode(
            _ambil(raw, "inacbgs_spesifikasi", "FKL22")
        ),
        "inacbgs_keparahan": _kode(
            _ambil(raw, "inacbgs_keparahan", "FKL23")
        ),
        "jenis_kunjungan_fktp": _kode(
            _ambil(raw, "jenis_kunjungan_fktp", "FKP22")
        ),
    }

    if tuple(fitur) != FEATURES:
        raise RuntimeError(
            "Urutan fitur hasil hitung_fitur tidak sama dengan features.json"
        )
    return fitur


def buat_dataframe(fitur: Mapping[str, Any]) -> pd.DataFrame:
    if set(fitur) != set(FEATURES):
        hilang = sorted(set(FEATURES) - set(fitur))
        lebih = sorted(set(fitur) - set(FEATURES))
        raise PreprocessingError(
            f"Vektor fitur tidak lengkap. Hilang={hilang}, lebih={lebih}"
        )

    baris: dict[str, float] = {}
    for nama in FEATURES:
        nilai = fitur[nama]
        if isinstance(nilai, bool):
            nilai = int(nilai)
        try:
            baris[nama] = float(nilai)
        except (TypeError, ValueError) as exc:
            raise PreprocessingError(
                f"Fitur {nama} bukan angka: {nilai!r}"
            ) from exc

    return pd.DataFrame([baris], columns=list(FEATURES), dtype="float64")


LABEL_FITUR: dict[str, str] = {
    "is_spatial_anomaly": "Anomali spasial (kabupaten FKTP dan FKRTL berbeda)",
    "is_gender_mismatch": "Inkonsistensi gender dengan diagnosis",
    "is_age_mismatch": "Inkonsistensi usia dengan diagnosis degeneratif",
    "frekuensi_hari_itu": "Frekuensi kunjungan pada hari yang sama",
    "biaya_ratio": "Rasio biaya verifikasi terhadap tagihan",
    "usia_saat_kunjungan": "Usia saat kunjungan",
    "jenis_kelamin": "Jenis kelamin",
    "hubungan_keluarga": "Hubungan keluarga",
    "kelas_rawat": "Kelas rawat",
    "segmen_peserta": "Segmen peserta",
    "status_kepesertaan": "Status kepesertaan",
    "tahun_sampel": "Tahun sampel",
    "provinsi_peserta": "Provinsi peserta",
    "kabupaten_peserta": "Kabupaten peserta",
    "provinsi_fktp": "Provinsi FKTP",
    "kabupaten_fktp": "Kabupaten FKTP",
    "provinsi_fkrtl": "Provinsi FKRTL",
    "kabupaten_fkrtl": "Kabupaten FKRTL",
    "jenis_fktp": "Jenis FKTP",
    "tipe_fktp": "Tipe FKTP",
    "tingkat_fktp": "Tingkat FKTP",
    "poli_fktp": "Poli FKTP",
    "kepemilikan_fktp": "Kepemilikan FKTP",
    "jenis_fkrtl": "Jenis FKRTL",
    "tipe_fkrtl": "Tipe FKRTL",
    "tingkat_fkrtl": "Tingkat FKRTL",
    "poli_fkrtl": "Poli FKRTL",
    "kepemilikan_fkrtl": "Kepemilikan FKRTL",
    "kelas_fkrtl": "Kelas FKRTL",
    "status_pulang_fkrtl": "Status pulang FKRTL",
    "durasi_rawat": "Durasi rawat (hari)",
    "bulan_kunjungan": "Bulan kunjungan",
    "hari_minggu": "Hari dalam minggu",
    "akhir_pekan": "Kunjungan akhir pekan",
    "biaya_tagih": "Biaya tagihan",
    "biaya_verifikasi": "Biaya verifikasi",
    "inacbgs_kelompok": "Kelompok INACBGs",
    "inacbgs_tipe": "Tipe INACBGs",
    "inacbgs_spesifikasi": "Spesifikasi INACBGs",
    "inacbgs_keparahan": "Tingkat keparahan INACBGs",
    "jenis_kunjungan_fktp": "Jenis kunjungan FKTP",
}

if set(LABEL_FITUR) != set(FEATURES):
    raise RuntimeError("LABEL_FITUR tidak sinkron dengan features.json")

NAMA_HARI = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu")
FITUR_FLAG = {
    "is_spatial_anomaly",
    "is_gender_mismatch",
    "is_age_mismatch",
    "akhir_pekan",
}
FITUR_BIAYA = {"biaya_tagih", "biaya_verifikasi"}


def format_nilai(nama: str, nilai: float) -> str:
    if nama in FITUR_FLAG:
        return "ya" if float(nilai) >= 0.5 else "tidak"
    if nama == "jenis_kelamin":
        return {1: "Laki-laki", 2: "Perempuan"}.get(
            int(nilai), "tidak diketahui"
        )
    if nama == "hari_minggu":
        return NAMA_HARI[int(nilai)] if 0 <= int(nilai) <= 6 else "tidak diketahui"
    if nama in FITUR_BIAYA:
        return f"Rp {float(nilai):,.0f}"
    if nama == "biaya_ratio":
        return f"{float(nilai):.2f}"
    if float(nilai).is_integer():
        return str(int(nilai))
    return f"{float(nilai):.2f}"


def buat_alasan(nama: str, nilai_fitur: float, shap_value: float) -> str:
    arah = "menaikkan" if shap_value >= 0 else "menurunkan"
    return (
        f"{LABEL_FITUR.get(nama, nama)} = {format_nilai(nama, nilai_fitur)} "
        f"(kontribusi {shap_value:+.3f}, {arah} risiko)"
    )


__all__ = [
    "FEATURES",
    "FEATURES_PATH",
    "PREPROCESSING_VERSION",
    "PreprocessingError",
    "buat_alasan",
    "buat_dataframe",
    "format_nilai",
    "hitung_fitur",
    "LABEL_FITUR",
]
