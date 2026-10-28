from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_raw() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root: Path = ROOT

    @property
    def gee_project(self) -> str:
        return self.raw["project"]["gee_project"]

    @property
    def city_name(self) -> str:
        return self.raw["study"]["city_name"]

    @property
    def city_label(self) -> str:
        return self.raw["study"]["city_label"]

    @property
    def focus_year(self) -> int:
        return int(self.raw["study"]["focus_year"])

    @property
    def years(self) -> list[int]:
        return [int(y) for y in self.raw["study"]["years"]]

    @property
    def crs_metric(self) -> str:
        return self.raw["study"]["crs_metric"]

    @property
    def analysis_m(self) -> int:
        return int(self.raw["scales"]["analysis_m"])

    def analysis_m_for(self, city: str | None = None) -> int:
        by = (self.raw.get("scales") or {}).get("analysis_m_by_city") or {}
        key = city or self.city_name
        if key in by:
            return int(by[key])
        return self.analysis_m

    @property
    def lst_m(self) -> int:
        return int(self.raw["scales"]["lst_m"])

    @property
    def feature_m(self) -> int:
        return int(self.raw["scales"]["feature_m"])

    @property
    def train_sample_n(self) -> int:
        return int(self.raw["scales"]["train_sample_n"])

    @property
    def block_cv_m(self) -> int:
        return int(self.raw["scales"]["block_cv_m"])

    @property
    def feature_cols(self) -> list[str]:
        return list(self.raw["features"])

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def out_maps(self) -> Path:
        return self.root / "outputs" / "maps"

    @property
    def out_tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def out_models(self) -> Path:
        return self.root / "outputs" / "models"

    @property
    def out_figures(self) -> Path:
        return self.root / "outputs" / "figures"

    def summer_range(self, year: int) -> tuple[str, str]:
        start = self.raw["study"]["summer_start"]
        end = self.raw["study"]["summer_end"]
        return f"{year}-{start}", f"{year}-{end}"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.out_maps,
            self.out_tables,
            self.out_models,
            self.out_figures,
        ):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings(raw=load_raw())
