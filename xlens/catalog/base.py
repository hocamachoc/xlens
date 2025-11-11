import numpy as np
from numpy.lib import recfunctions as rfn


NUM_Z_GRIDS = 401
Z_MAX = 4.0
Z_GRIDS = np.linspace(0.0, Z_MAX, NUM_Z_GRIDS)
PROBS = np.array([0.025, 0.16, 0.5, 0.84, 0.975], dtype=float)


def _resolve_flux_min(
    flux_min: float | dict,
    bands: str = "grizy",
) -> dict[str, float]:
    """Return per-band flux_min as dict{band: value}."""
    if isinstance(flux_min, dict):
        return {b: float(flux_min[b]) for b in bands}
    else:
        f = float(flux_min)
        return {b: f for b in bands}

def _resolve_flux_name(flux_name):
    if len(flux_name) > 0:
        if flux_name[0] != "_":
            fn = "_" + flux_name
        else:
            fn = flux_name
    else:
        fn = ""
    return fn


def get_esq(src: np.ndarray, comp: int = 1, dg: float = 0.0) -> np.ndarray:
    """Return |e|^2 evaluated at shear g_comp = dg to first order."""
    if comp not in (1, 2):
        raise ValueError(f"comp must be 1 or 2, got {comp!r}")

    e  = src[f"fpfs_e{comp}"]
    de = src[f"fpfs_de{comp}_dg{comp}"]

    comp2 = 3 - comp        # 1 ↔ 2
    e2  = src[f"fpfs_e{comp2}"]
    de2 = src[f"fpfs_de{comp2}_dg{comp}"]

    esq0 = e * e + e2 * e2
    return esq0 + 2.0 * dg * (e * de + e2 * de2)


def get_color(
    src: np.ndarray,
    *,
    bands: str = "grizy",
    ref_band: str = "i",
    mag_zero: float = 30.0,
    comp: int = 1,
    dg: float = 0.0,
    flux_name: str = "gauss2",
) -> np.ndarray:
    """
    Parameters
    ----------
    src : np.ndarray (structured)
        Must include fields per band:
          {b}_flux
          {b}_dflux_dg{comp}
          {b}_flux_err
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
    fn = _resolve_flux_name(flux_name)
    A = 2.5 / np.log(10.0)
    n = src.shape[0]

    mags: dict[str, np.ndarray] = {}
    merrs: dict[str, np.ndarray] = {}

    for b in bands:
        flux_col  = f"{b}_flux{fn}"
        dflux_col = f"{b}_dflux{fn}_dg{comp}"
        err_col   = f"{flux_col}_err"

        flux_base = src[flux_col]
        dflux     = src[dflux_col]
        ferr      = src[err_col]

        flux = flux_base + dg * dflux

        mag     = np.full(n, 30.0, dtype=np.float64)
        mag_err = np.full(n,  1.0, dtype=np.float64)

        pos = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag[pos]     = mag_zero - 2.5 * np.log10(flux[pos])
            mag_err[pos] = A * (ferr[pos] / flux[pos])

        mags[b]  = mag
        merrs[b] = mag_err

    nb = len(bands) - 1
    feat = np.empty((n, 1 + 2 * nb), dtype=np.float64)

    feat[:, 0] = mags[ref_band]

    j = 1
    for i in range(nb):
        b1, b2 = bands[i], bands[i + 1]
        # color = mag(b1) - mag(b2)
        np.subtract(mags[b1], mags[b2], out=feat[:, j])
        j += 1
        # color error = sqrt(err1^2 + err2^2)
        feat[:, j] = np.hypot(merrs[b1], merrs[b2])
        j += 1
    return feat


def get_summary_from_pdfs(pdfs: np.ndarray) -> np.ndarray:
    """Return 95% width (z_97.5 - z_2.5) per row of `pdfs`."""
    n = pdfs.shape[0]
    width95 = np.full(n, np.nan, dtype=float)
    for i, p in enumerate(pdfs):
        total = p.sum()
        if not np.isfinite(total) or total <= 0.0:
            continue
        cdf = np.cumsum(p, dtype=float)
        cdf /= cdf[-1]
        zqs = np.interp(PROBS, cdf, Z_GRIDS)
        width95[i] = zqs[-1] - zqs[0]
    return width95


def get_flexzboost(
    src: np.ndarray,
    pz_obj,
    comp: int = 1,
    dg: float = 0.0,
    mag_zero: float = 30.0,
    flux_name: str = "gauss2",
):
    colors = get_color(
        src,
        mag_zero=mag_zero,
        comp=comp,
        dg=dg,
        flux_name=flux_name,
    )
    pdfs, _ = pz_obj.predict(colors, n_grid=NUM_Z_GRIDS)
    del colors

    # Argmax per row, then map to z_grid
    idx = np.argmax(pdfs, axis=1)
    zmode = np.take(Z_GRIDS, idx)
    del idx
    width95 = get_summary_from_pdfs(pdfs)
    return zmode, width95


def measure_shear_with_cut(
    *,
    src: np.ndarray,
    pz_obj,
    zbounds: list[float],
    flux_min: float | dict = 40.0,
    emax: float = 0.3,
    z_width95_max: float = 2.75,
    dg: float = 0.02,
    target: str = "g1",
    do_correction: bool = True,
    mag_zero: float = 30.0,
    flux_name: str = "gauss2",
):
    fn = _resolve_flux_name(flux_name)
    esq0 = get_esq(src)
    g_flux = src[f"g_flux{fn}"]
    r_flux = src[f"r_flux{fn}"]
    i_flux = src[f"i_flux{fn}"]
    z_flux = src[f"z_flux{fn}"]
    y_flux = src[f"y_flux{fn}"]

    wsel    = src["wsel"]
    e1_all  = src["fpfs_e1"]
    e2_all  = src["fpfs_e2"]
    dwsel_dg1 = src["dwsel_dg1"]
    dwsel_dg2 = src["dwsel_dg2"]
    de1_dg1   = src["fpfs_de1_dg1"]
    de2_dg2   = src["fpfs_de2_dg2"]

    fm = _resolve_flux_min(flux_min, bands="grizy")
    g_flux_min = fm["g"]
    r_flux_min = fm["r"]
    i_flux_min = fm["i"]
    z_flux_min = fm["z"]
    y_flux_min = fm["y"]

    # base selection
    mask = (
        (g_flux > g_flux_min)
        & (r_flux > r_flux_min)
        & (i_flux > i_flux_min)
        & (z_flux > z_flux_min)
        & (y_flux > y_flux_min)
        & (esq0 < emax * emax)
    )

    # photo-z + width cut at base shear
    zmode, width95 = get_flexzboost(
        src[mask], pz_obj=pz_obj, mag_zero=mag_zero, flux_name=flux_name,
    )
    mtmp = (width95 < z_width95_max)
    mask[mask] &= mtmp
    zmode = zmode[mtmp]
    del mtmp, width95

    idx0 = np.digitize(zmode, zbounds, right=False)
    minlen = len(zbounds) + 1

    def sel_term(comp: int) -> np.ndarray:
        """Selection response term for component comp (1 or 2)."""
        g_df = src[f"g_dflux{fn}_dg{comp}"]
        r_df = src[f"r_dflux{fn}_dg{comp}"]
        i_df = src[f"i_dflux{fn}_dg{comp}"]
        z_df = src[f"z_dflux{fn}_dg{comp}"]
        y_df = src[f"y_dflux{fn}_dg{comp}"]
        e_comp = src[f"fpfs_e{comp}"]

        def one_side(sign: float) -> np.ndarray:
            """Compute binned ⟨w_sel e⟩ for shear +sign*dg."""
            dg_eff = sign * dg
            esq_side = get_esq(src, comp=comp, dg=dg_eff)
            mask_side = (
                (g_flux + dg_eff * g_df > g_flux_min)
                & (r_flux + dg_eff * r_df > r_flux_min)
                & (i_flux + dg_eff * i_df > i_flux_min)
                & (z_flux + dg_eff * z_df > z_flux_min)
                & (y_flux + dg_eff * y_df > y_flux_min)
                & (esq_side < emax * emax)
            )

            if do_correction:
                z_side, w_side = get_flexzboost(
                    src[mask_side],
                    pz_obj=pz_obj,
                    comp=comp,
                    dg=dg_eff,
                    mag_zero=mag_zero,
                    flux_name=flux_name,
                )
            else:
                z_side, w_side = get_flexzboost(
                    src[mask_side],
                    pz_obj=pz_obj,
                    comp=comp,
                    dg=0.0,
                    mag_zero=mag_zero,
                    flux_name=flux_name,
                )

            mtmp = (w_side < z_width95_max)
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
        return e1, r1, r1_sel
    elif target == "g2":
        e2 = np.bincount(idx0, wsel[mask] * e2_all[mask], minlength=minlen)
        r2 = np.bincount(
            idx0,
            dwsel_dg2[mask] * e2_all[mask] + wsel[mask] * de2_dg2[mask],
            minlength=minlen,
        )
        r2_sel = sel_term(2)
        return e2, r2, r2_sel
    else:
        raise ValueError(f"target must be 'g1' or 'g2', got {target!r}")
