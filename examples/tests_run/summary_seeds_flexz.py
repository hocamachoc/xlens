#!/usr/bin/env python3
"""Aggregate shear measurements over individual simulation using FlexZBoost."""
import argparse
import gc
import glob
import os
import pickle
from typing import Iterable, List, Optional, Sequence, Tuple

import fitsio
import numpy as np
import qp
from astropy.stats import sigma_clipped_stats
from numpy.lib import recfunctions as rfn

NUM_Z_GRIDS = 401
Z_MAX = 4.0
Z_GRIDS = np.linspace(0.0, Z_MAX, NUM_Z_GRIDS)
PROBS = np.array([0.025, 0.16, 0.5, 0.84, 0.975], dtype=float)


colnames = [
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_de1_dg2",
    "fpfs_e2",
    "fpfs_de2_dg1",
    "fpfs_de2_dg2",
]


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
    parser.add_argument(
        "--correction", action=argparse.BooleanOptionalAction, default=True
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
        default=30.0,  # 40
        help="Flux cut applied to each band before selection.",
    )
    parser.add_argument(
        "--z-bounds",
        type=str,
        default="0.3,0.6,0.9,1.2,1.5",
        help="Comma-separated redshift boundarys, e.g. '0.3,0.6,0.9,1.2'.",
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
        help="# bootstrap resamples for m uncertainty",
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


def parse_zbounds(values: str) -> Sequence[float]:
    return [float(x) for x in values.split(",")]


def base_path(pscratch, layout, target, shear):
    sd = f"shear{int(shear*100):02d}"
    return os.path.join(
        pscratch,
        f"constant_shear_{layout}",
        target,
        sd,
    )


def _to_native(a: np.ndarray) -> np.ndarray:
    if a.dtype.byteorder in ('=', '|'):
        return a
    return a.byteswap().newbyteorder('=')


def cat_read(base_dir: str, sim_id: int, mode: int) -> np.ndarray:
    arrs = []
    main_path = os.path.join(base_dir, f"mode{mode}", f"cat-{sim_id:05d}.fits")
    main = fitsio.read(main_path, columns=colnames)
    arrs.append(_to_native(main))

    for b in "grizy":
        fn = os.path.join(base_dir, f"mode{mode}", f"cat-{sim_id:05d}-{b}.fits")
        cols = [f"{b}_flux_gauss2", f"{b}_flux_gauss2_err",
                f"{b}_dflux_gauss2_dg1", f"{b}_dflux_gauss2_dg2"]
        barr = fitsio.read(fn, columns=cols)
        arrs.append(_to_native(barr))

    merged = rfn.merge_arrays(arrs, flatten=True, asrecarray=False, usemask=False)
    return rfn.repack_fields(merged)


def get_esq(src: np.ndarray, comp: int = 1, dg: float = 0.0) -> np.ndarray:
    e = src[f"fpfs_e{comp}"]
    de = src[f"fpfs_de{comp}_dg{comp}"]
    comp2 = int(3 - comp)
    e2 = src[f"fpfs_e{comp2}"]
    de2 = src[f"fpfs_de{comp2}_dg{comp}"]
    esq0 = e**2.0 + e2**2.0
    return esq0 + 2.0 * dg * (e * de + e2 * de2)


def get_color(
    src: np.ndarray,
    *,
    ref_band: str = "i_mag_gauss2",  # e.g. "i_mag_gauss2"
    bands: str = "grizy",            # order for adjacent colors
    mag_zero: float = 30.0,
    comp: int = 1,
    dg: float = 0.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    src : np.ndarray (structured)
        Must include fields per band:
          {b}_flux_gauss2
          {b}_dflux_gauss2_dg{comp}
          {b}_flux_gauss2_err
        for each b in `bands` (default "grizy").

    Returns
    -------
    np.ndarray, shape (N, 1 + 2*(len(bands)-1))
        Columns ordered as:
          [ref_mag,
           (g-r), err(g-r),
           (r-i), err(r-i),
           (i-z), err(i-z),
           (z-y), err(z-y)
           ]
    """
    A = 2.5 / np.log(10.0)
    n = src.shape[0]

    # 1) Per-band magnitudes and magnitude errors (from gauss2 flux + dflux*dg)
    mags  = {}
    merrs = {}
    for b in bands:
        flux_col  = f"{b}_flux_gauss2"
        dflux_col = f"{b}_dflux_gauss2_dg{comp}"
        err_col   = f"{flux_col}_err"

        # Direct field access from structured array; no extra array wrapping.
        flux_base = src[flux_col]
        dflux = src[dflux_col]
        ferr = src[err_col]

        # Perturbed flux
        flux = flux_base + dg * dflux

        # Allocate outputs (float64 math is robust; dtype follows NumPy's
        # upcasting)
        mag = np.full(n, 30.0, dtype=np.float64)
        mag_err = np.full(n, 1.0, dtype=np.float64)
        pos = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag[pos]     = mag_zero - 2.5 * np.log10(flux[pos])
            mag_err[pos] = A * (ferr[pos] / flux[pos])
        mags[b]  = mag
        merrs[b] = mag_err

    # 2) Assemble feature matrix in the same order as original input_data.T
    nb = len(bands) - 1
    feat = np.empty((n, 1 + 2 * nb), dtype=np.float64)

    # Reference magnitude: use leading letter from ref_band (e.g. "i" from
    # "i_mag_gauss2")
    ref_letter = ref_band.split("_", 1)[0]
    feat[:, 0] = mags[ref_letter]
    j = 1
    for i in range(nb):
        b1, b2 = bands[i], bands[i + 1]
        # color = mag(b1) - mag(b2)
        np.subtract(mags[b1], mags[b2], out=feat[:, j]); j += 1
        # color error = sqrt(err1^2 + err2^2)
        feat[:, j] = np.hypot(merrs[b1], merrs[b2]); j += 1
    return feat


def get_redshift(src: np.ndarray, pz_obj, comp: int = 1, dg: float = 0.0):
    mag_zero = 30.0
    colors = get_color(src, mag_zero=mag_zero, comp=comp, dg=dg)
    pdfs, _ = pz_obj.predict(
        colors,
        n_grid=NUM_Z_GRIDS,
    )
    del colors
    # Argmax per row, then map to z_grid (avoid storing pdfs beyond this)
    idx = np.argmax(pdfs, axis=1)
    zmode = np.take(Z_GRIDS, idx)
    del idx
    width95 = np.full(len(zmode), np.nan, dtype=float)
    for i, p in enumerate(pdfs):
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        zqs = np.interp(PROBS, cdf, Z_GRIDS)
        width95[i] = zqs[-1] - zqs[0]
    return zmode, width95


def measure_shear_with_cut(
    *,
    src: np.ndarray,
    pz_obj,
    zbounds: Sequence[float],
    flux_min: float = 40.0,
    emax: float = 0.3,
    dg: float = 0.02,
    target: str = "g1",
    do_correction: bool = True,
):
    esq0 = get_esq(src)
    g_flux = src["g_flux_gauss2"]
    r_flux = src["r_flux_gauss2"]
    i_flux = src["i_flux_gauss2"]
    z_flux = src["z_flux_gauss2"]
    y_flux = src["y_flux_gauss2"]
    wsel = src["wsel"]
    e1_all = src["fpfs_e1"]
    e2_all = src["fpfs_e2"]
    dwsel_dg1 = src["dwsel_dg1"]
    dwsel_dg2 = src["dwsel_dg2"]
    de1_dg1 = src["fpfs_de1_dg1"]
    de2_dg2 = src["fpfs_de2_dg2"]

    # base selection
    mask = (
        (g_flux > flux_min)
        & (r_flux > flux_min)
        & (i_flux > flux_min)
        & (z_flux > flux_min)
        & (y_flux > flux_min)
        & (esq0 < emax * emax)
    )

    zmode, width95 = get_redshift(src[mask], pz_obj=pz_obj)
    mtmp = (width95 < 2.75)
    mask[mask] &= mtmp
    zmode = zmode[mtmp]
    del mtmp, width95

    idx0 = np.digitize(zmode, zbounds, right=False)
    minlen = len(zbounds) + 1

    def sel_term(comp: int) -> np.ndarray:
        """Selection response term for component comp (1 or 2)."""

        g_df = src[f"g_dflux_gauss2_dg{comp}"]
        r_df = src[f"r_dflux_gauss2_dg{comp}"]
        i_df = src[f"i_dflux_gauss2_dg{comp}"]
        z_df = src[f"z_dflux_gauss2_dg{comp}"]
        y_df = src[f"y_dflux_gauss2_dg{comp}"]
        e_comp = src[f"fpfs_e{comp}"]

        def one_side(sign: float) -> np.ndarray:
            """Compute binned ⟨w_sel e⟩ for shear +sign*dg."""
            dg_eff = sign * dg
            esq_side = get_esq(src, comp=comp, dg=dg_eff)
            mask_side = (
                (g_flux + dg_eff * g_df > flux_min)
                & (r_flux + dg_eff * r_df > flux_min)
                & (i_flux + dg_eff * i_df > flux_min)
                & (z_flux + dg_eff * z_df > flux_min)
                & (y_flux + dg_eff * y_df > flux_min)
                & (esq_side < emax * emax)
            )

            if do_correction:
                z_side, w_side = get_redshift(
                    src[mask_side], pz_obj=pz_obj, comp=comp, dg=dg_eff
                )
            else:
                z_side, w_side = get_redshift(
                    src[mask_side], pz_obj=pz_obj, comp=comp, dg=0.0
                )

            mtmp = (w_side < 2.75)
            mask_side[mask_side] &= mtmp
            z_side = z_side[mtmp]
            del mtmp, w_side

            idx_side = np.digitize(z_side, zbounds, right=False)
            ell_side = np.bincount(
                idx_side,
                wsel[mask_side] * e_comp[mask_side],
                minlength=minlen,
            )
            del esq_side, mask_side, idx_side
            return ell_side

        ellp = one_side(+1.0)
        ellm = one_side(-1.0)
        return (ellp - ellm) / (2.0 * dg)

    if target == "g1":
        e1 = np.bincount(idx0, wsel[mask] * e1_all[mask], minlength=minlen)
        r1 = np.bincount(
            idx0,
            dwsel_dg1[mask] * e1_all[mask] + wsel[mask] * de1_dg1[mask],
            minlength=minlen,
        )
        r1_sel = sel_term(1)
        return e1, (r1 + r1_sel)
    elif target == "g2":
        e2 = np.bincount(idx0, wsel[mask] * e2_all[mask], minlength=minlen)
        r2 = np.bincount(
            idx0,
            dwsel_dg2[mask] * e2_all[mask] + wsel[mask] * de2_dg2[mask],
            minlength=minlen,
        )
        r2_sel = sel_term(2)
        return e2, (r2 + r2_sel)
    else:
        raise ValueError(f"target must be 'g1' or 'g2', got {target!r}")


def per_rank_work(
    ids_chunk: Iterable[int],
    base_dir: str,
    zbounds: Sequence[float],
    flux_min: float,
    emax: float,
    dg: float,
    target: str,
    pz_obj,
    do_correction: bool = True,
):
    e_pos_rows = []
    e_neg_rows = []
    r_pos_rows = []
    r_neg_rows = []
    for sim_id in ids_chunk:
        src_pos = cat_read(base_dir, sim_id, mode=40)
        e_pos_row, r_pos_row = measure_shear_with_cut(
            src=src_pos,
            flux_min=flux_min,
            pz_obj=pz_obj,
            emax=emax,
            zbounds=zbounds,
            dg=dg,
            target=target,
            do_correction = do_correction,
        )
        del src_pos
        gc.collect()
        src_neg = cat_read(base_dir, sim_id, mode=0)
        e_neg_row, r_neg_row = measure_shear_with_cut(
            src=src_neg,
            flux_min=flux_min,
            pz_obj=pz_obj,
            emax=emax,
            zbounds=zbounds,
            dg=dg,
            target=target,
            do_correction = do_correction,
        )
        del src_neg
        gc.collect()
        e_pos_rows.append(e_pos_row)
        e_neg_rows.append(e_neg_row)
        r_pos_rows.append(r_pos_row)
        r_neg_rows.append(r_neg_row)

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
    do_correction: bool = True,
) -> str:
    partdir = os.path.join(outdir, "summary-flexz2-40-00")
    if not do_correction:
        partdir = partdir + "-nc"
    os.makedirs(partdir, exist_ok=True)
    path = os.path.join(partdir, f"seed_{seed_index:05d}.npz")
    np.savez_compressed(
        path,
        E_pos=e_pos,
        E_neg=e_neg,
        R_pos=r_pos,
        R_neg=r_neg,
        ncut=int(ncut),
    )
    return path


def load_and_stack_all(
    outdir: str, ncut_expected: Optional[int] = None,
    do_correction: bool = True,
):
    partdir = os.path.join(outdir, "summary-flexz2-40-00")
    if not do_correction:
        partdir = partdir + "-nc"
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
    do_correction = args.correction
    if args.max_id <= args.min_id:
        raise SystemExit("--max-id must be > --min-id")
    base_dir = base_path(args.pscratch, args.layout, args.target, args.shear)
    zbounds = parse_zbounds(args.z_bounds)
    ncut = len(zbounds) + 1
    if not args.summary:
        model_path = args.model_path
        with open(model_path, "rb") as f:
            pz_obj = pickle.load(f)
            pz_obj.model.models.n_jobs = 1
        my_ids = np.arange(args.min_id, args.max_id, dtype=int)
        if len(my_ids) > 0:
            e_pos, e_neg, r_pos, r_neg = per_rank_work(
                my_ids,
                base_dir,
                zbounds,
                args.flux_min,
                args.emax,
                args.dg,
                args.target,
                pz_obj,
                do_correction=do_correction,
            )
            save_rank_partial(
                base_dir, int(my_ids[0]), e_pos, e_neg, r_pos, r_neg, ncut,
                do_correction=do_correction,
            )
    else:
        all_e_pos, all_e_neg, all_r_pos, all_r_neg = load_and_stack_all(
            base_dir, ncut_expected=ncut,
            do_correction=do_correction,
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
        print(f"Redshift boundarys: {list(zbounds)}")
        print(f"Area (arcmin^2): {area_arcmin2:.3f}")
        print("m (per redshift cut):", m)
        print("c (per redshift cut):", c)
        print("n_eff (per redshift cut):", neff)
        print("m 1-sigma (bootstrap):", sigma_m)
        print("c 1-sigma (bootstrap):", sigma_c)
        print("==============================================")

if __name__ == "__main__":
    main()
