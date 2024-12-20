# stdlib imports --------------------------------------------------- #
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime
import json
import functools
import logging
import logging.handlers
import os
import pathlib
import sys
import time
import types
import typing
import uuid
import zoneinfo
from typing import Any, Generator, Iterable, Literal

# 3rd-party imports necessary for processing ----------------------- #
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import sklearn
import pynwb
import tqdm
import upath
import zarr

import numba_psth

logger = logging.getLogger(__name__)

CO_COMPUTATION_ID = os.environ.get("CO_COMPUTATION_ID")
AWS_BATCH_JOB_ID = os.environ.get("AWS_BATCH_JOB_ID")

def is_pipeline():
    return bool(AWS_BATCH_JOB_ID)

# logging ----------------------------------------------------------- #
class PSTFormatter(logging.Formatter):

    def converter(self, timestamp):
        # may require 'tzdata' package
        dt = datetime.datetime.fromtimestamp(timestamp, tz=zoneinfo.ZoneInfo("UTC"))
        return dt.astimezone(zoneinfo.ZoneInfo("US/Pacific"))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            t = dt.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s

def setup_logging(
    level: int | str = logging.INFO, filepath: str | None = None
) -> logging.handlers.QueueListener | None:
    """
    Setup logging that works for local, capsule and pipeline environments.

    - with no input arguments, log messages at INFO level and above are printed to stdout
    - in Code Ocean capsules, stdout is captured in an 'output' file automatically
    - in pipelines, stdout from each capsule instance is also captured in a central 'output' file
      - for easier reading, this function saves log files from each capsule instance individually to logs/<AWS_BATCH_JOB_ID>.log
    - in local environments or capsules, file logging can be enabled by setting the `filepath` argument

    Note: logger is not currently safe for multiprocessing/threading (ignore WIP below)

    Note: if file logging is enabled in a multiprocessing/multithreading context, a `queue` should be set to True
    to correctly handle logs from multiple processes/threads. In this mode, a QueueListener is returned.
    When processes/threads shutdown, `QueueListener().stop()` must be called to ensure all logs are captured correctly.
    The `queue_logging()` context manager is provided to handle this within a process/thread:

        ```python
        def worker_process():
            with queue_logging():
                logger.info('Process started')
                # do work here
                logger.info('Process finished')

        processes = []
        for _ in range(5):
            process = multiprocessing.Process(target=worker_process)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
        logger.info('All processes finished')
        ```

    """
    if is_pipeline():
        assert AWS_BATCH_JOB_ID is not None
        co_prefix = f"{AWS_BATCH_JOB_ID.split('-')[0]}."
    else:
        co_prefix = ""

    fmt = f"%(asctime)s | %(levelname)s | {co_prefix}%(name)s.%(funcName)s | %(message)s"
    
    formatter = PSTFormatter( # use Seattle time
        fmt=fmt,
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )
    
    handlers: list[logging.Handler] = []
    
    # Create a console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    handlers.append(console_handler)
    
    if is_pipeline() and not filepath:
        filepath = f"/results/logs/{AWS_BATCH_JOB_ID}.log"
        # note: filename must be unique if we want to collect logs at end of pipeline
        
    if filepath:
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=filepath,
            maxBytes=1024 * 1024 * 10,
        )
        handlers.append(file_handler)

    # Apply formatting to the console handler and attach to root logger
    for handler in handlers:
        handler.setFormatter(formatter)
    # Configure the root logger
    logging.basicConfig(level=level, handlers=handlers)


# data access ------------------------------------------------------- #    
def get_session_table() -> pl.DataFrame:
    return pl.read_parquet(get_datacube_dir() / 'session_table.parquet')
    
def get_df(component: str) -> pl.DataFrame:
    path = get_datacube_dir() / 'consolidated' / f'{component}.parquet'
    return (
        pl.read_parquet(path)
        .with_columns(
            pl.col('session_id').str.split('_').list.slice(0, 2).list.join('_')
        )
    )

@functools.cache
def get_nwb_paths() -> tuple[pathlib.Path, ...]:
    return tuple(get_data_root().rglob('*.nwb'))

def get_nwb(session_id_or_path: str | pathlib.Path, raise_on_missing: bool = True, raise_on_bad_file: bool = True) -> pynwb.NWBFile:
    if isinstance(session_id_or_path, (pathlib.Path, upath.UPath)):
        nwb_path = session_id_or_path
    else:
        if not isinstance(session_id_or_path, str):
            raise TypeError(f"Input should be a session ID (str) or path to an NWB file (str/Path), got: {session_id_or_path!r}")
        if pathlib.Path(session_id_or_path).exists():
            nwb_path = session_id_or_path
        elif session_id_or_path.endswith(".nwb") and any(p.name == session_id_or_path for p in get_nwb_paths()):
            nwb_path = next(p for p in get_nwb_paths() if p.name == session_id_or_path)
        else:
            try:
                nwb_path = next(p for p in get_nwb_paths() if p.stem == session_id_or_path)
            except StopIteration:
                msg = f"Could not find NWB file for {session_id_or_path!r}"
                if not raise_on_missing:
                    logger.error(msg)
                    return
                else:
                    raise FileNotFoundError(f"{msg}. Available files: {[p.name for p in get_nwb_paths()]}") from None
    logger.info(f"Reading {nwb_path}")
    try:
        nwb = pynwb.NWBHDF5IO(nwb_path).read()
    except RecursionError:
        msg = f"{nwb_path.name} cannot be read due to RecursionError (hdf5 may still be accessible)"
        if not raise_on_bad_file:
            logger.error(msg)
            return
        else:
            raise RecursionError(msg)
    else:
        return nwb
        
def _get_spike_times_single_nwb(nwb_path: str | pathlib.Path, unit_ids: str | Iterable[str], use_pynwb: bool = True) -> dict[str, npt.NDArray[np.float64]]:    
    if isinstance(unit_ids, str):
        unit_ids = (unit_ids,)
    unit_ids = tuple(unit_ids)
    if isinstance(nwb_path, str):
        nwb_path = pathlib.Path(nwb_path)
    if not nwb_path.exists():
        raise FileNotFoundError(nwb_path)
    logging.debug(f"Fetching spike times for {len(unit_ids)} units from {nwb_path}")
    if use_pynwb:
        t0 = time.time()
        nwb = pynwb.NWBHDF5IO(nwb_path, 'r').read()
        logger.debug(f"Opened NWB in {time.time() - t0:.2f}s")
        t0 = time.time()
        nwb_unit_ids = nwb.units.unit_id[:]
        logger.debug(f"Got unit IDs in NWB in {time.time() - t0:.2f}s")
        t0 = time.time()
        nwb_unit_idx = [np.where(nwb_unit_ids == unit_id)[0][0] for unit_id in unit_ids]
        logger.debug(f"Got unit indices in NWB in {time.time() - t0:.2f}s")
        t0 = time.time()
        spike_times = nwb.units.get_unit_spike_times(nwb_unit_idx)
        logger.debug(f"Got spike times from NWB in {time.time() - t0:.2f}s")
        return dict(zip(unit_ids, spike_times))
    else:
        t0 = time.time()
        units = h5py.File(nwb_path.as_posix(), 'r')['units']
        nwb_unit_ids = units['unit_id'].asstr()[()]
        nwb_unit_idx = [np.where(nwb_unit_ids == unit_id)[0][0] for unit_id in unit_ids]
        unit_id_to_spike_times = {}
        spike_times_index = units["spike_times_index"]
        for unit_id, unit_idx in zip(unit_ids, nwb_unit_idx):
            if unit_idx == 0:
                start_idx = 0
            else:
                start_idx = spike_times_index[unit_idx - 1].item()
            end_idx = spike_times_index[unit_idx].item()
            assert start_idx < end_idx, f"{start_idx=} >= {end_idx=}"
            spike_times = units["spike_times"][start_idx:end_idx]
            unit_id_to_spike_times[unit_id] = spike_times
        logger.debug(f"Got spike times from hdf5 directly in {time.time() - t0:.2f}s")
        return unit_id_to_spike_times

def get_spike_times(unit_ids: str | Iterable[str]) -> dict[str, npt.NDArray[np.float64]]:
    """"""
    if isinstance(unit_ids, str):
        unit_ids = (unit_ids,)
    unit_ids = sorted(unit_ids)
    session_to_unit_ids = {}
    for unit_id in unit_ids:
        session_id = '_'.join(unit_id.split('_')[:2])
        session_to_unit_ids.setdefault(session_id, []).append(unit_id)
    logging.debug(f"Fetching spike times ({len(unit_ids)} units) from {len(session_to_unit_ids)} session(s)")
    
    future_to_nwb_session_id = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for session_id in session_to_unit_ids:
            nwb_path = get_datacube_dir() / 'nwb' / f"{session_id}.nwb"
            future = executor.submit(
                _get_spike_times_single_nwb,
                nwb_path=nwb_path,
                unit_ids=session_to_unit_ids[session_id],
                use_pynwb=False,
            )
            future_to_nwb_session_id[future] = session_id
        unit_id_to_spike_times = {}
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_nwb_session_id), total=len(tuple(future_to_nwb_session_id)), unit='sessions', desc=f"Getting spike times ({len(unit_ids)} units)"):
            unit_id_to_spike_times.update(future.result())
    assert len(unit_id_to_spike_times) == len(unit_ids)
    return unit_id_to_spike_times

# analysis ----------------------------------------------------------- #
def make_psth(
    spike_times: npt.NDArray[np.floating],
    start_times: npt.NDArray[np.floating],
    baseline_dur: float = 0.1,
    response_dur: float = 1.0,
    bin_size: float = 0.001,
    conv_kernel_size: float = 0.01,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if conv_kernel_size < bin_size:
        raise ValueError(f"PSTH {conv_kernel_size=} must be greater than or equal to {bin_size=}")
    spike_times = np.array(spike_times, dtype=np.float64)
    if spike_times.ndim != 1:
        raise ValueError(f"Expected spike_times to be a 1-d array: got ({spike_times.shape})")
    return numba_psth.makePSTH(spike_times, np.array(start_times, dtype=np.float64), float(baseline_dur), float(response_dur), float(bin_size), float(conv_kernel_size))

def unit_id_to_session_id(unit_id: str) -> str:
    return unit_id.rpartition("_")[0]

def combine_exprs(exprs: Iterable[pl.expr]) -> pl.expr:
    return pl.Expr.and_(*exprs)

@dataclasses.dataclass(frozen=True, repr=False) # repr=False to avoid printing psths; use slots=True on 3.10+ for lower memory usage
class UnitResponse:
    """Lightweight immutable bundle of PSTH + metadata for a single unit. Make millions of these."""
    psth: Iterable[float]
    n_trials: int
    unit_id: str
    @property
    def session_id(self) -> str:
        return self.unit_id.rpartition("_")[0]

@dataclasses.dataclass(repr=False)
class Condition:
    """Bundle of parameters that define a condition, and placeholder for unit responses under that condition. Make tens of these."""
    trials_filter: pl.expr | Iterable[pl.expr]
    session_table_filter: pl.expr | Iterable[pl.expr]
    units_filter: pl.expr | Iterable[pl.expr]
    area: str
    context: str
    stim: str
    # fields that are filled in after data processing:
    unit_responses: tuple[UnitResponse, ...] = None
    bins: npt.NDArray[np.float64] = None
    result: Any = None
    
    def __hash__(self) -> int:
        return (
            hash(combine_exprs(self.units_filter).meta.serialize(format='json'))
            ^ hash(combine_exprs(self.trials_filter).meta.serialize(format='json'))
            ^ hash(combine_exprs(self.session_table_filter).meta.serialize(format='json'))
        )
    
    def __lt__(self, other: Condition) -> bool:
        return hash(self) < hash(other)
    
    def __eq__(self, other: Condition) -> bool:
        return hash(self) == hash(other)
    
    @property
    def psths(self) -> npt.NDArray[np.float64]:
        """units x bins"""
        if self.unit_responses is None:
            return AttributeError("Spike times haven't been fetched yet")
        a = np.array([r.psth for r in self.unit_responses])
        assert a.ndim == 2
        assert a.shape[0] == len(self.unit_responses)
        assert a.shape[1] == len(self.bins)
        return a
    @property
    def unit_ids(self) -> npt.NDArray[str]:
        return np.array([r.unit_id for r in self.unit_responses])

def get_scaled_concatenated_psths(
    conditions: Iterable[Condition],
    z_score_axis: Literal[0, 1] = 0,
    pre_subtract_baseline: bool = True,
) -> dict[str, tuple[tuple[Condition, ...], npt.NDArray[np.float64]]]:
    areas = sorted({c.area for c in conditions})
    area_to_conditions: dict[str, list[Condition]] = {
        area: sorted(c for c in conditions if c.area == area) 
        for area in areas
    }
    def scale(a):
        if pre_subtract_baseline:
            a = (a.T - np.mean(a, axis=1)).T
        if z_score_axis == 0:
            return (a - np.mean(a, axis=0)) / np.std(a, axis=0)
        elif z_score_axis == 1:
            return ((a.T - np.mean(a, axis=1)) / np.std(a, axis=1)).T
        else:
            raise ValueError(f"z_score_axis must be 0 (z-score calculated at each bin of PSTH, across units) or 1 (z-score calculated for each unit's PSTH in isolation)")

    info = {area: len(conditions) for area, conditions in area_to_conditions.items()}
    logger.info(f"Getting scaled psths for {{area: n_conditions}}: {info}")
    area_to_psths: dict[str, npt.NDArray[np.float64]] = {
        area: scale(np.concatenate([c.psths for c in conditions], axis=1))
        for area, conditions in area_to_conditions.items()
    }
    assert len(area_to_conditions) == len(area_to_psths)
    assert all(area_to_psths[area].shape[0] == len(condition.unit_responses) for area in areas for condition in area_to_conditions[area])
    return {area: (tuple(area_to_conditions[area]), area_to_psths[area]) for area in areas}
        
def process_conditions_by_area(
    conditions: Iterable[Condition],
    **psth_kwargs,
) -> list[Condition]:
    """For each Condition, add unit responses (with `makePSTH`) for matching trials and return the mutated instance"""
    
    conditions = tuple(conditions)
    if len(areas := {c.area for c in conditions}) > 1:
        raise ValueError(f"This function is for processing groups of Conditions with the same area: got {areas=} in {len(conditions)} Conditions")
    
    # get spike times for all units in area that pass filters for unit and session metrics
    select_units = (
        get_df('units', lazy=True)
        .filter(conditions[0].units_filter)
        .join(
            other=get_session_table(lazy=True).filter(conditions[0].session_table_filter),
            on='session_id',
            how='semi', # keeps rows in left table that have a match in right table
        )
    ).collect()
    logger.debug(f"Getting spike times for {conditions[0].area}") 
    unit_id_to_spike_times: dict[str, npt.NDArray[np.float64]] = get_spike_times(select_units['unit_id'])
    if not unit_id_to_spike_times:
        raise ValueError(f"No unit spike times returned for {conditions[0].area}: check units and session table filter expressions")
    trials = get_df('trials')
    for condition in conditions:
        # get psth for each unit in area, for trials that match the parameters of this condition 
        unit_responses = []  
        for unit_id, spike_times in tqdm.tqdm(
            unit_id_to_spike_times.items(),
            total=len(unit_id_to_spike_times),
            unit='units',
            desc=f"Getting PSTHs ({condition.area} | {condition.stim} | {condition.context})",
        ): 
            start_times = (
                trials
                .filter(
                    *condition.trials_filter,
                    pl.col('session_id').str.starts_with(unit_id_to_session_id(unit_id))
                )
            )['stim_start_time']
            psth, bins = make_psth(spike_times=spike_times, start_times=start_times, **psth_kwargs) 
            unit_responses.append(
                UnitResponse(
                    psth=psth,
                    n_trials=len(start_times),
                    unit_id=unit_id,
                )
            )
        condition.unit_responses=tuple(unit_responses)
        condition.bins=bins
    return conditions

def get_unit_responses(
    conditions: Iterable[Condition],
    parallel: bool = True,
    **psth_kwargs,
) -> list[Condition]:
    """Dispatch conditions grouped by CCF area for processing together. 1 process per area, multiple conditions per area. Mutates each Condition in place."""
    conditions_by_area = [[c for c in conditions if c.area == area] for area in {c.area for c in conditions}]
    conditions_with_responses = []
    if not parallel:
        for conditions in conditions_by_area:
            conditions_with_responses.extend(process_conditions_by_area(conditions))
    else:
        futures = []
        with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
            for conditions in conditions_by_area:
                futures.append(executor.submit(process_conditions_by_area, conditions=conditions, **psth_kwargs))
            for future in concurrent.futures.as_completed(futures):
                conditions_with_responses.extend(future.result())
    return conditions_with_responses

# paths ----------------------------------------------------------- #
@functools.cache
def get_datacube_dir() -> pathlib.Path:
    for p in sorted(get_data_root().iterdir(), reverse=True): # in case we have multiple assets attached, the latest will be used
        if p.is_dir() and p.name.startswith('dynamicrouting_datacube'):
            path = p
            break
    else:
        for p in get_data_root().iterdir():
            if any(pattern in p.name for pattern in ('session_table', 'nwb', 'consolidated', )):
                path = get_data_root()
                break
        else:
            raise FileNotFoundError(f"Cannot determine datacube dir: {list(get_data_root().iterdir())=}")
    logger.info(f"Using files in {path}")
    return path

@functools.cache
def get_data_root(as_str: bool = False) -> pathlib.Path:
    expected_paths = ('/data', '/tmp/data', )
    for p in expected_paths:
        if (data_root := pathlib.Path(p)).exists():
            logger.info(f"Using {data_root=}")
        return data_root.as_posix() if as_str else data_root
    else:
        raise FileNotFoundError(f"data dir not present at any of {expected_paths=}")

@functools.cache
def get_nwb_paths() -> tuple[pathlib.Path, ...]:
    return tuple(get_data_root().rglob('*.nwb'))

def ensure_nonempty_results_dir() -> None:
    """A pipeline run can crash if a results folder is expected and not found or is empty 
    - ensure that a non-empty folder exists by creating a unique file"""
    if not is_pipeline():
        return
    results = pathlib.Path("/results")
    results.mkdir(exist_ok=True)
    if not list(results.iterdir()):
        path = results / uuid.uuid4().hex
        logger.info(f"Creating {path} to ensure results folder is not empty")
        path.touch()


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(
            doctest.NORMALIZE_WHITESPACE
            | doctest.ELLIPSIS
            | doctest.IGNORE_EXCEPTION_DETAIL
        ),
    )
