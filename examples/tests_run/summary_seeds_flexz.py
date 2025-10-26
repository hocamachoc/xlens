#!/usr/bin/env python3
"""Aggregate shear measurements over individual simulation seeds using FlexZBoost."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, Sequence, Tuple, List, Optional

import fitsio
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from mpi4py import MPI

from rail.estimation.algos.flexzboost import FlexZBoostEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure + aggregate from catalogs over a given seed ID range "
            "using FlexZBoost redshift slices."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--summary", action=argparse.BooleanOptionalAction, default=False
    )
    # Directory layout and naming
    parser.add_argument(
        "--pscratch",
        type=str,
        default=os.environ.get("PSCRATCH", "."),
        help="Root directory where results were written.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Layout used in path naming.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="g1",
        choices=["g1", "g2"],
        help="Which component to analyze (affects R and e used).",
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.02,
        help="True shear amplitude |g| used in sims.",
    )
    # ID range
    parser.add_argument(
        "--min-id",
        type=int,
        required=True,
        help="Minimum sim_seed (inclusive).",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        required=True,
        help="Maximum sim_seed (exclusive).",
    )
    # Measurement config
    parser.add_argument(
        "--flux-min",
        type=float,
        default=40.0,
        help="Flux cut applied to each band before selection.",
    )
    parser.add_argument(
        "--z-mins",
        type=str,
        default="0.3,0.6,0.9,1.2",
        help="Comma-separated redshift lower limits, e.g. '0.3,0.6,0.9'.",
    )
    parser.add_argument(
        "--z-width",
        type=float,
        default=0.3,
        help="Width of the redshift slice applied around each lower limit.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=0.3,
        help="Ellipticity magnitude cut upper bound.",
    )
    parser.add_argument(
        "--dg",
        type=float,
        default=0.02,
        help="Finite-difference step for selection response.",
    )
    # Geometry for density / area
    parser.add_argument(
        "--stamp-dim",
        type=int,
        default=3900,
        help="Usable image dimension (pixels) for density/area calc.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=0.2,
        help="Pixel scale (arcsec/pixel).",
    )
    # Bootstrap
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="# bootstrap resamples for m uncertainty (done on rank 0).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("FLEXZ_MODEL"),
        help=(
            "Path to the FlexZBoost trained model. Defaults to the FLEXZ_MODEL "
            "environment variable if set."
        ),
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[warn] Ignoring unknown args:", unknown_args)
    return args


def parse_zmin_list(values: str) -> Sequence[float]:
    return [float(x) for x in values.split(",")] if values else [0.3, 0.6, 0.9, 1.2]


def outdir_path(pscratch: str, layout: str, target: str, shear: float) -> str:
    shear_dir = f"shear{int(shear * 100):02d}"
    return os.path.join(pscratch, f"constant_shear_{layout}", target, shear_dir)


def cat_path(base_dir: str, sim_id: int, mode: int) -> str:
    return os.path.join(base_dir, f"cat-{sim_id:05d}-mode{mode}.fits")


def _ensure_model_path(path: str | None) -> str:
    if not path:
        raise SystemExit(
            "FlexZBoost model path must be supplied via --model-path or FLEXZ_MODEL"
        )
    if not os.path.exists(path):
        raise SystemExit(f"FlexZBoost model file not found: {path}")
    return path


def get_esq(src: Table, comp: int = 1, dg: float = 0.0) -> np.ndarray:
    e = src[f"fpfs_e{comp}"]
    de = src[f"fpfs_de{comp}_dg{comp}"]
    comp2 = int(3 - comp)
    e2 = src[f"fpfs_e{comp2}"]
    de2 = src[f"fpfs_de{comp2}_dg{comp}"]
    esq0 = e**2.0 + e2**2.0
    return esq0 + 2.0 * dg * (e * de + e2 * de2)


def get_color(src: Table, mag_zero: float, comp: int = 1, dg: float = 0.0) -> Table:
    coeff = 2.5 / np.log(10.0)
    colors = Table(length=len(src))
    for band in "grizy":
        flux_col = f"{band}_flux_gauss2"
        dflux_col = f"{band}_dflux_gauss2_dg{comp}"
        err_col = f"{flux_col}_err"
        mag_col = flux_col.replace("flux", "mag")
        mag_err_col = err_col.replace("flux", "mag")

        flux_base = src[flux_col]
        dflux = src[dflux_col]
        ferr = src[err_col]
        flux = flux_base + dflux * dg

        mag = np.full(len(flux), np.nan, dtype=float)
        mag_err = np.full(len(flux), np.nan, dtype=float)
        pos = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag[pos] = mag_zero - 2.5 * np.log10(flux[pos])
            mag_err[pos] = coeff * (ferr[pos] / flux[pos])

        colors[mag_col] = mag
        colors[mag_err_col] = mag_err
    return colors


def get_redshift(src: Table, model_path: str, comp: int = 1, dg: float = 0.0) -> np.ndarray:
    mag_zero = 30.0
    colors = get_color(src, mag_zero=mag_zero, comp=comp, dg=dg)
    estimator = FlexZBoostEstimator.make_stage(
        output_mode="return",
        zmin=0.0,
        zmax=4.0,
        nzbins=401,
        calc_summary_stats=False,
        recompute_point_estimates=False,
        calculated_point_estimates=["zmode", "zbest", "zmean", "zmedian"],
        nondetect_val=np.nan,
        mag_limits={
            "g_mag_gauss2": 25.5,
            "r_mag_gauss2": 25.5,
            "i_mag_gauss2": 25.5,
            "z_mag_gauss2": 25.5,
            "y_mag_gauss2": 25.5,
        },
        bands=[
            "g_mag_gauss2",
            "r_mag_gauss2",
            "i_mag_gauss2",
            "z_mag_gauss2",
            "y_mag_gauss2",
        ],
        err_bands=[
            "g_mag_gauss2_err",
            "r_mag_gauss2_err",
            "i_mag_gauss2_err",
            "z_mag_gauss2_err",
            "y_mag_gauss2_err",
        ],
        ref_band="i_mag_gauss2",
        qp_representation="interp",
        nprocess=1,
    )
    estimator.open_model(model=model_path)
    estimator._process_chunk(0, len(colors), colors, first=True)
    outputs = estimator._output_handle.data
    return np.ravel(outputs.ancil["zmode"])


def measure_shear_with_cut(
    src: Table,
    flux_min: float,
    model_path: str,
    emax: float = 0.3,
    zmin: float = 1.0,
    zwidth: float = 0.3,
    dg: float = 0.02,
) -> Tuple[float, float, float, float, int]:
    z0 = get_redshift(src, model_path=model_path)
    esq0 = get_esq(src)

    g_flux = src["g_flux_gauss2"]
    r_flux = src["r_flux_gauss2"]
    i_flux = src["i_flux_gauss2"]
    z_flux = src["z_flux_gauss2"]
    y_flux = src["y_flux_gauss2"]
    print(len(z0))

    mask = (
        (g_flux > flux_min)
        & (r_flux > flux_min)
        & (i_flux > flux_min)
        & (z_flux > flux_min)
        & (y_flux > flux_min)
        & (esq0 < emax * emax)
        & (z0 > zmin)
        & (z0 <= zmin + zwidth)
    )
    print(np.sum(mask))
    nn = int(np.sum(mask))
    if nn == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    wsel = src["wsel"]
    e1_all = src["fpfs_e1"]
    e2_all = src["fpfs_e2"]
    dwsel_dg1 = src["dwsel_dg1"]
    dwsel_dg2 = src["dwsel_dg2"]
    de1_dg1 = src["fpfs_de1_dg1"]
    de2_dg2 = src["fpfs_de2_dg2"]

    w0 = wsel[mask]
    e1 = np.sum(w0 * e1_all[mask])
    e2 = np.sum(w0 * e2_all[mask])

    r1 = np.sum(dwsel_dg1[mask] * e1_all[mask] + w0 * de1_dg1[mask])
    r2 = np.sum(dwsel_dg2[mask] * e2_all[mask] + w0 * de2_dg2[mask])

    def sel_term(comp: int) -> float:
        esq_p = get_esq(src, comp=comp, dg=dg)
        z_p = get_redshift(src, model_path=model_path, comp=comp, dg=dg)
        esq_m = get_esq(src, comp=comp, dg=-dg)
        z_m = get_redshift(src, model_path=model_path, comp=comp, dg=-dg)

        g_df = src[f"g_dflux_gauss2_dg{comp}"]
        r_df = src[f"r_dflux_gauss2_dg{comp}"]
        i_df = src[f"i_dflux_gauss2_dg{comp}"]
        z_df = src[f"z_dflux_gauss2_dg{comp}"]
        y_df = src[f"y_dflux_gauss2_dg{comp}"]

        gflux_p = g_flux + dg * g_df
        rflux_p = r_flux + dg * r_df
        iflux_p = i_flux + dg * i_df
        zflux_p = z_flux + dg * z_df
        yflux_p = y_flux + dg * y_df
        mask_p = (
            (gflux_p > flux_min)
            & (rflux_p > flux_min)
            & (iflux_p > flux_min)
            & (zflux_p > flux_min)
            & (yflux_p > flux_min)
            & (esq_p < emax * emax)
            & (z_p > zmin)
            & (z_p <= zmin + zwidth)
        )
        ellp = np.sum(wsel[mask_p] * src[f"fpfs_e{comp}"][mask_p])

        gflux_m = g_flux - dg * g_df
        rflux_m = r_flux - dg * r_df
        iflux_m = i_flux - dg * i_df
        zflux_m = z_flux - dg * z_df
        yflux_m = y_flux - dg * y_df
        mask_m = (
            (gflux_m > flux_min)
            & (rflux_m > flux_min)
            & (iflux_m > flux_min)
            & (zflux_m > flux_min)
            & (yflux_m > flux_min)
            & (esq_m < emax * emax)
            & (z_m > zmin)
            & (z_m <= zmin + zwidth)
        )
        ellm = np.sum(wsel[mask_m] * src[f"fpfs_e{comp}"][mask_m])
        return (ellp - ellm) / (2.0 * dg)

    r1_sel = sel_term(1)
    r2_sel = sel_term(2)
    return e1, (r1 + r1_sel), e2, (r2 + r2_sel), nn


def per_rank_work(
    ids_chunk: Iterable[int],
    base_dir: str,
    zmin_list: Sequence[float],
    flux_min: float,
    emax: float,
    zwidth: float,
    dg: float,
    target: str,
    model_path: str,
):
    ncut = len(zmin_list)
    e_pos_rows = []
    e_neg_rows = []
    r_pos_rows = []
    r_neg_rows = []

    for sim_id in ids_chunk:
        path_pos = cat_path(base_dir, sim_id, mode=40)
        path_neg = cat_path(base_dir, sim_id, mode=0)
        if not (os.path.exists(path_pos) and os.path.exists(path_neg)):
            continue

        try:
            src_pos = Table(fitsio.read(path_pos))
            src_neg = Table(fitsio.read(path_neg))
        except OSError:
            print(path_pos)
            print(path_neg)
            continue

        e_pos_row = np.zeros(ncut)
        e_neg_row = np.zeros(ncut)
        r_pos_row = np.zeros(ncut)
        r_neg_row = np.zeros(ncut)

        for idx, zmin in enumerate(zmin_list):
            e1p, R1p, e2p, R2p, _ = measure_shear_with_cut(
                src_pos,
                flux_min,
                model_path,
                emax=emax,
                zmin=zmin,
                zwidth=zwidth,
                dg=dg,
            )
            e1m, R1m, e2m, R2m, _ = measure_shear_with_cut(
                src_neg,
                flux_min,
                model_path,
                emax=emax,
                zmin=zmin,
                zwidth=zwidth,
                dg=dg,
            )

            if target == "g1":
                e_pos_row[idx] = e1p
                e_neg_row[idx] = e1m
                r_pos_row[idx] = R1p
                r_neg_row[idx] = R1m
            else:
                e_pos_row[idx] = e2p
                e_neg_row[idx] = e2m
                r_pos_row[idx] = R2p
                r_neg_row[idx] = R2m

        e_pos_rows.append(e_pos_row)
        e_neg_rows.append(e_neg_row)
        r_pos_rows.append(r_pos_row)
        r_neg_rows.append(r_neg_row)

    if not e_pos_rows:
        return (np.zeros((0, ncut)),) * 4
    return (
        np.vstack(e_pos_rows),
        np.vstack(e_neg_rows),
        np.vstack(r_pos_rows),
        np.vstack(r_neg_rows),
    )


def save_rank_partial(
    outdir: str,
    seed_index: int,
    e_pos: np.ndarray,
    e_neg: np.ndarray,
    r_pos: np.ndarray,
    r_neg: np.ndarray,
    ncut: int,
) -> str:
    partdir = os.path.join(outdir, "summary-flexz-40-00")
    os.makedirs(partdir, exist_ok=True)
    path = os.path.join(partdir, f"seed_{seed_index:05d}.npz")
    np.savez_compressed(
        path,
        E_pos=e_pos,
        E_neg=e_neg,
        R_pos=r_pos,
        R_neg=r_neg,
        ncut=np.int64(ncut),
    )
    return path


def load_and_stack_all(
    outdir: str, ncut_expected: Optional[int] = None
):
    partdir = os.path.join(outdir, "summary-flexz-40-00")
    arrays_E_pos: List[np.ndarray] = []
    arrays_E_neg: List[np.ndarray] = []
    arrays_R_pos: List[np.ndarray] = []
    arrays_R_neg: List[np.ndarray] = []
    ncut_from_file: Optional[int] = None

    for path in sorted(glob.glob(os.path.join(partdir, "*.npz"))):
        with np.load(path) as data:
            arrays_E_pos.append(data["E_pos"])
            arrays_E_neg.append(data["E_neg"])
            arrays_R_pos.append(data["R_pos"])
            arrays_R_neg.append(data["R_neg"])
            if ncut_from_file is None:
                ncut_from_file = int(data["ncut"])

    def _stack(blocks: List[np.ndarray], ncut: int) -> np.ndarray:
        valid = [blk for blk in blocks if blk.size > 0]
        if not valid:
            return np.zeros((0, ncut), dtype=np.float64)
        return np.vstack(valid)

    ncut = ncut_expected if ncut_expected is not None else (ncut_from_file or 0)
    E_pos_all = _stack(arrays_E_pos, ncut)
    E_neg_all = _stack(arrays_E_neg, ncut)
    R_pos_all = _stack(arrays_R_pos, ncut)
    R_neg_all = _stack(arrays_R_neg, ncut)
    return E_pos_all, E_neg_all, R_pos_all, R_neg_all


def bootstrap_m(
    rng: np.random.Generator,
    e_pos: np.ndarray,
    e_neg: np.ndarray,
    r_pos: np.ndarray,
    r_neg: np.ndarray,
    shear_value: float,
    nsamp: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_obj, ncut = e_pos.shape
    ms = np.zeros((nsamp, ncut))
    cs = np.zeros((nsamp, ncut))
    for idx in range(nsamp):
        choices = rng.integers(0, n_obj, size=n_obj, endpoint=False)
        denom = np.sum(r_pos[choices] + r_neg[choices], axis=0)

        num_m = np.sum(e_pos[choices] - e_neg[choices], axis=0)
        gamma = num_m / denom
        ms[idx] = gamma / shear_value - 1.0

        num_c = np.sum(e_pos[choices] + e_neg[choices], axis=0)
        cs[idx] = num_c / denom
    return ms, cs


def main() -> None:
    args = parse_args()
    model_path = _ensure_model_path(args.model_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.max_id <= args.min_id:
        raise SystemExit("--max-id must be > --min-id")

    zmin_list = parse_zmin_list(args.z_mins)
    base_dir = outdir_path(args.pscratch, args.layout, args.target, args.shear)
    ncut = len(zmin_list)

    if not args.summary:
        all_ids = np.arange(args.min_id, args.max_id, dtype=int)
        total = len(all_ids)
        base = total // size
        rem = total % size
        start = rank * base + min(rank, rem)
        stop = start + base + (1 if rank < rem else 0)
        my_ids = all_ids[start:stop]

        e_pos, e_neg, r_pos, r_neg = per_rank_work(
            my_ids,
            base_dir,
            zmin_list,
            args.flux_min,
            args.emax,
            args.z_width,
            args.dg,
            args.target,
            model_path,
        )

        index = (
            int(my_ids[0]) if len(my_ids) > 0 else (args.min_id + rank)
        )
        save_rank_partial(base_dir, index, e_pos, e_neg, r_pos, r_neg, ncut)
        comm.Barrier()
    else:
        if rank == 0:
            all_e_pos, all_e_neg, all_r_pos, all_r_neg = load_and_stack_all(
                base_dir, ncut_expected=ncut
            )

            if all_e_pos.size == 0 or all_e_neg.size == 0:
                raise SystemExit(
                    "No valid (+g/-g) pairs found in the given seed ID range."
                )

            num = np.sum(all_e_pos - all_e_neg, axis=0)
            denom = np.sum(all_r_pos + all_r_neg, axis=0)
            m = (num / denom) / args.shear - 1.0

            c = np.sum(all_e_pos + all_e_neg, axis=0) / np.sum(
                all_r_pos + all_r_neg, axis=0
            )

            area_arcmin2 = (args.stamp_dim * args.stamp_dim) * (
                args.pixel_scale / 60.0
            ) ** 2.0

            _, _, clipped_std = sigma_clipped_stats(
                all_e_pos / np.average(all_r_pos, axis=0),
                sigma=5.0,
                axis=0,
            )
            neff = (0.26 / clipped_std) ** 2.0 / area_arcmin2

            rng = np.random.default_rng(0)
            ms, cs = bootstrap_m(
                rng,
                all_e_pos,
                all_e_neg,
                all_r_pos,
                all_r_neg,
                args.shear,
                nsamp=args.bootstrap,
            )
            ord_ms = np.sort(ms, axis=0)
            lo_idx = int(0.1587 * args.bootstrap)
            hi_idx = int(0.8413 * args.bootstrap)
            sigma_m = (ord_ms[hi_idx] - ord_ms[lo_idx]) / 2.0

            ord_cs = np.sort(cs, axis=0)
            sigma_c = (ord_cs[hi_idx] - ord_cs[lo_idx]) / 2.0

            print("==============================================")
            print(f"Catalog directory: {base_dir}")
            print(f"Paired IDs (found): {all_e_pos.shape[0]}")
            print(f"ID range requested: [{args.min_id}, {args.max_id})")
            print(f"Redshift lower limits: {list(zmin_list)}")
            print(f"Redshift slice width: {args.z_width}")
            print(f"Area (arcmin^2): {area_arcmin2:.3f}")
            print("m (per redshift cut):", m)
            print("c (per redshift cut):", c)
            print("n_eff (per redshift cut):", neff)
            print("m 1-sigma (bootstrap):", sigma_m)
            print("c 1-sigma (bootstrap):", sigma_c)
            print("==============================================")


if __name__ == "__main__":
    main()
