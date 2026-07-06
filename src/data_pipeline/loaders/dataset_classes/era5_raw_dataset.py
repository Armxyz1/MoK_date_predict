from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

import dask
dask.config.set(scheduler='threads')


class ERA5RawDataset(Dataset):
    """
    Optimized PyTorch Dataset for loading raw 6-hourly ERA5 data
    with temporal aggregation and per-year caching.
    """

    def __init__(
        self,
        base_dir: str = "/gdata2/ERA5",
        years: Optional[List[int]] = None,
        time_steps: Optional[List[int]] = None,
        temporal_aggregation: str = 'daily',
        pressure_levels: Optional[List[int]] = None,
        transform: Optional[callable] = None,
        target_file: Optional[str] = None,
        input_geo_var_surf: Optional[List[str]] = None,
        input_geo_var_press: Optional[List[str]] = None,
        input_geo_var_surf_src: Optional[List[str]] = None,
        input_geo_var_press_src: Optional[List[str]] = None,
        include_lat: bool = True,
        include_lon: bool = True,
        include_landsea: bool = False,
        is_classification: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.years = years if years is not None else []
        self.time_steps = time_steps if time_steps is not None else [0, 1, 2]
        self.temporal_aggregation = temporal_aggregation.lower()
        self.pressure_levels = pressure_levels if pressure_levels is not None else [0, 1]
        self.transform = transform
        self.is_classification = is_classification

        # Static flags retained but ignored (no static channels added)
        self.include_lat = False
        self.include_lon = False
        self.include_landsea = False

        self.input_geo_var_surf = input_geo_var_surf or ['t2m', 'sst', 'msl', 'tcc']
        self.input_geo_var_press = input_geo_var_press or ['u', 'v']

        self.input_geo_var_surf_src = input_geo_var_surf_src or [
            f"{v}_60S_60N_all_lon_6h" for v in self.input_geo_var_surf
        ]
        self.input_geo_var_press_src = input_geo_var_press_src or [
            f"{v}_60S_60N_all_lon_6h" for v in self.input_geo_var_press
        ]

        valid_aggregations = ['daily', 'weekly', 'monthly', '3day']
        if self.temporal_aggregation not in valid_aggregations:
            raise ValueError(f"temporal_aggregation must be one of {valid_aggregations}")

        self.targets = self._load_targets(target_file, is_classification) if target_file else {}

        self.data_index = []
        self._build_index()

        # 🔥 Per-year cache (key speedup)
        self._year_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------

    def _load_targets(self, target_file: str, is_classification: bool) -> Dict[int, float]:
        df = pd.read_csv(target_file)

        target_col = 'OnsetBinCode' if is_classification else 'DateRelJun01'
        if target_col not in df.columns:
            raise ValueError(f"Missing target column {target_col}")

        return dict(zip(df['Year'].values, df[target_col].values))

    # ------------------------------------------------------------------

    def _build_index(self):
        for year in self.years:
            missing = False

            for src in self.input_geo_var_surf_src + self.input_geo_var_press_src:
                if not (self.base_dir / src / f"{year}.nc").exists():
                    missing = True
                    break

            if not missing:
                self.data_index.append((year, 0))

    # ------------------------------------------------------------------

    def _aggregate_temporal(self, da: xr.DataArray) -> xr.DataArray:
        if self.temporal_aggregation == 'daily':
            return da.resample(valid_time='1D').mean()
        if self.temporal_aggregation == '3day':
            return da.resample(valid_time='3D').mean()
        if self.temporal_aggregation == 'weekly':
            return da.resample(valid_time='7D').mean()
        if self.temporal_aggregation == 'monthly':
            return da.resample(valid_time='1MS').mean()
        raise RuntimeError

    # ------------------------------------------------------------------

    def _load_variable(self, year: int, var: str, src: str) -> xr.DataArray:
        path = self.base_dir / src / f"{year}.nc"
        with xr.open_dataset(path) as ds:
            if var in ds.data_vars:
                da = ds[var]
            else:
                da = list(ds.data_vars.values())[0]
            return da.load()

    # ------------------------------------------------------------------

    def _load_year(self, year: int) -> torch.Tensor:
        channels = []

        # ---- Surface vars ----
        for var, src in zip(self.input_geo_var_surf, self.input_geo_var_surf_src):
            da = self._load_variable(year, var, src)
            da = self._aggregate_temporal(da)
            data = da.isel(valid_time=self.time_steps).values
            channels.append(np.nan_to_num(data, nan=0.0))

        # ---- Pressure vars ----
        for var, src in zip(self.input_geo_var_press, self.input_geo_var_press_src):
            da = self._load_variable(year, var, src)
            da = self._aggregate_temporal(da)

            level_dim = 'level' if 'level' in da.dims else 'pressure_level'

            data = da.isel(
                valid_time=self.time_steps,
                **{level_dim: self.pressure_levels}
            ).values

            data = np.nan_to_num(data, nan=0.0)

            # (T, L, H, W) → (T×L, H, W)
            data = data.reshape(-1, *data.shape[-2:])
            channels.append(data)

        stacked = np.concatenate(channels, axis=0)
        return torch.from_numpy(stacked).float()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_index)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        year, _ = self.data_index[idx]

        if year not in self._year_cache:
            self._year_cache[year] = self._load_year(year)

        x = self._year_cache[year]
        if self.transform:
            x = self.transform(x)

        y = self.targets.get(year, np.nan)
        return x, torch.tensor([y], dtype=torch.float32)

    # ------------------------------------------------------------------

    def get_channel_info(self) -> Dict:
        num_surf = len(self.input_geo_var_surf)
        num_press = len(self.input_geo_var_press) * len(self.pressure_levels)
        num_time = len(self.time_steps)

        total = (num_surf + num_press) * num_time

        names = []
        for t in self.time_steps:
            for v in self.input_geo_var_surf:
                names.append(f"{v}_t{t}")
            for v in self.input_geo_var_press:
                for l in self.pressure_levels:
                    names.append(f"{v}_p{l}_t{t}")

        return {
            "num_channels": total,
            "channel_names": names,
        }

    # ------------------------------------------------------------------

    def get_metadata(self, idx: int) -> Dict:
        year, _ = self.data_index[idx]
        return {
            "year": year,
            "temporal_aggregation": self.temporal_aggregation,
        }