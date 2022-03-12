"""Author: Connor Stone

Calculate galaxy structural parameters.

Functions here will take a galaxy object and compute some structural
parameter from the avaialble profiles. The parameter(s) will then be
added to the galaxy object and returned. Several calculations are
dependent on other calculations preceeding them, this is noted in
their documentation.
"""

from .Supporting_Functions import (
    app_mag_to_abs_mag,
    mag_to_L,
    L_to_mag,
    pc_to_arcsec,
    arcsec_to_pc,
    H0,
    c,
    Grav_C,
    surface_brightness,
    flux_to_mag,
    mag_to_flux,
    Get_M2L,
    inclination,
    muSB_to_ISB,
)

from .Profile_Functions import (
    Evaluate_Magnitude,
    Isophotal_Radius,
    Evaluate_Surface_Brightness,
    Effective_Radius,
    bulgefit,
    Courteau97_Model_Fit,
    Courteau97_Model_Evaluate,
    Tan_Model_Evaluate,
    Tan_Model_Fit,
    Tanh_Model_Evaluate,
    Tanh_Model_Fit,
)
from .Decorators import catch_errors, all_appR, all_bands
from .K_correction import calc_kcor
from scipy.integrate import quad, trapz
import matplotlib.pyplot as plt
import logging
import numpy as np


@catch_errors
def Calc_Apparent_Radius(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the apparent radius for a given requested radius and band.

    references: None
    """
    if eval_at_R == "RI":
        Reval = [np.inf, 0.0]
    elif eval_at_R[:2] == "Ri":
        iso = float(eval_at_R[2:])
        if iso < G["photometry"][eval_at_band]["SB"][0]:
            Reval = [0.,0.]
        else:
            Reval = Isophotal_Radius(
                G["photometry"][eval_at_band]["R"],
                G["photometry"][eval_at_band]["SB"],
                iso,
                E=G["photometry"][eval_at_band]["SB_e"],
            )
    elif eval_at_R[:2] == "Rp":
        per = float(eval_at_R[2:])
        if per <= 0 or per >= 100:
            Reval = [0., 0.]
        else:
            Reval = Effective_Radius(
                G["photometry"][eval_at_band]["R"],
                G["photometry"][eval_at_band]["totmag"],
                E=G["photometry"][eval_at_band]["totmag_e"],
                ratio=per / 100,
            )
    elif eval_at_R[:2] == "Re":
        if f'Rp50:{eval_at_band}' in G['appR']:
            Reval = [G['appR'][f"Rp50:{eval_at_band}"], G['appR'][f"E|Rp50:{eval_at_band}"]]
        else:
            Reval = Effective_Radius(
                G["photometry"][eval_at_band]["R"],
                G["photometry"][eval_at_band]["totmag"],
                E=G["photometry"][eval_at_band]["totmag_e"],
                ratio=0.5,
            )
        Reval = (float(eval_at_R[2:]) * Reval[0], Reval[1])
    else:
        raise ValueError(f"unrecognized evaluation radius: {eval_at_R}")

    G["appR"][f"{eval_at_R}:{eval_at_band}"] = Reval[0]
    G["appR"][f"E|{eval_at_R}:{eval_at_band}"] = Reval[1]
    return G

@all_appR
@catch_errors
def Calc_Physical_Radius(G, eval_at_R=None, eval_at_band=None):
    """
    Compute the physical radius for a given requested radius and band.

    references: Calc_Apparent_Radius
    """

    Rphys = arcsec_to_pc(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["distance"] * 1e6,
        thetae=G["appR"][f"E|{eval_at_R}:{eval_at_band}"],
        De=G["distance_e"] * 1e6,
    )

    G["physR"][f"{eval_at_R}:{eval_at_band}"] = Rphys[0] / 1e3
    if eval_at_R == 'RI':
        G["physR"][f"E|{eval_at_R}:{eval_at_band}"] = 0.
    else:
        G["physR"][f"E|{eval_at_R}:{eval_at_band}"] = Rphys[1] / 1e3

    return G


@all_appR
@catch_errors
def Calc_Axis_Ratio(G, eval_in_band="r", eval_at_R=None, eval_at_band=None):
    """
    Interpolate the isophote axis ratio at a requested radius and band.

    references: Calc_Apparent_Radius
    """
    if (
        G["appR"][f"{eval_at_R}:{eval_at_band}"]
        < G["photometry"][eval_in_band]["R"][-1]
    ):
        q = np.interp(
            G["appR"][f"{eval_at_R}:{eval_at_band}"],
            G["photometry"][eval_in_band]["R"],
            1 - G["photometry"][eval_in_band]["ellip"],
        )
    else:
        q = np.median(1 - G["photometry"][eval_in_band]["ellip"][-5:])
    G["q"][f"{eval_at_R}:{eval_at_band}"] = q
    G["q"][f"E|{eval_at_R}:{eval_at_band}"] = max(
        0.05,
        np.std(1 - G["photometry"][eval_in_band]["ellip"][-5:]),
    )

    return G

@catch_errors
def Calc_Inclination_Profile(G, eval_in_band="r"):
    """Compute the inclination at a requested radius and band. Accounting
    for intrinsic thickness of the disk.

    references: Calc_Axis_Ratio
    """
    G['inclination_prof']['R'] = G["photometry"][eval_in_band]["R"]
    G['inclination_prof']['inclination'] = inclination(
        1 - G["photometry"][eval_in_band]["ellip"], G["q0"]
    )
    G['inclination_prof']['inclination_e'] = (
        np.ones(len(G["photometry"][eval_in_band]["ellip"])) * 5 * np.pi / 180
    )  # fixme

    return G


@all_appR
@catch_errors
def Calc_Inclination(G, eval_in_band="r", eval_at_R=None, eval_at_band=None):
    """Compute the inclination at a requested radius and band. Accounting
    for intrinsic thickness of the disk.

    references: Calc_Axis_Ratio
    """
    G["inclination"][f"{eval_at_R}:{eval_at_band}"] = np.interp(G['appR'][f"{eval_at_R}:{eval_at_band}"], G['inclination_prof']['R'], G['inclination_prof']['inclination'])
    G["inclination"][f"E|{eval_at_R}:{eval_at_band}"] = 5 * np.pi / 180  # fixme

    return G


@catch_errors
def Calc_C97_Velocity_Fit(G):
    x = Courteau97_Model_Fit(
        G["rotation curve"]["R"],
        G["rotation curve"]["V"],
        G["rotation curve"]["V_e"],
        fixed_origin=True,
    )

    G["rc_model"]["C97:param order"] = ["r_t", "v_c", "beta", "gamma", "v0", "x0"]
    G["rc_model"]["C97:r_t"] = x[0]
    G["rc_model"]["C97:v_c"] = x[1]
    G["rc_model"]["C97:beta"] = x[2]
    G["rc_model"]["C97:gamma"] = x[3]
    G["rc_model"]["C97:v0"] = x[4]
    G["rc_model"]["C97:x0"] = x[5]

    return G


@catch_errors
def Calc_Tan_Velocity_Fit(G):

    x = Tan_Model_Fit(
        G["rotation curve"]["R"],
        G["rotation curve"]["V"],
        G["rotation curve"]["V_e"],
        fixed_origin=True,
    )

    G["rc_model"]["Tan:param order"] = ["r_t", "v_c", "v0", "x0"]
    G["rc_model"]["Tan:r_t"] = x[0]
    G["rc_model"]["Tan:v_c"] = x[1]
    G["rc_model"]["Tan:v0"] = x[2]
    G["rc_model"]["Tan:x0"] = x[3]

    return G

@catch_errors
def Calc_Tanh_Velocity_Fit(G):
    
    x = Tanh_Model_Fit(
        G["rotation curve"]["R"],
        G["rotation curve"]["V"],
        G["rotation curve"]["V_e"],
        fixed_origin=True,
    )

    G["rc_model"]["Tanh:param order"] = ["r_t", "v_c", "v0", "x0"]
    G["rc_model"]["Tanh:r_t"] = x[0]
    G["rc_model"]["Tanh:v_c"] = x[1]
    G["rc_model"]["Tanh:v0"] = x[2]
    G["rc_model"]["Tanh:x0"] = x[3]

    return G


@all_appR
@catch_errors
def Calc_Velocity(G, eval_at_R=None, eval_at_band=None, eval_with_model="C97"):
    """
    Compute the rotation velocity at a requested radius and band.

    references: Calc_C97_Velocity_Fit, Calc_Apparent_Radius
    """
    x = list(
        G["rc_model"][f"{eval_with_model}:{p}"]
        for p in G["rc_model"][f"{eval_with_model}:param order"]
    )
    if eval_with_model == "C97":
        model = Courteau97_Model_Evaluate
    elif eval_with_model == "Tan":
        model = Tan_Model_Evaluate
    elif eval_with_model == "Tanh":
        model = Tanh_Model_Evaluate
    else:
        raise ValueError(f"unrecognized model type {eval_with_model}")
    if eval_at_R == 'RI':
        # If evaluating at infinity, try to evaluate at the limit, but max out at 1.5 times the largest actual measured radius
        Vobs = min(model(x[:-2] + [0, 0], np.inf), np.max(np.abs(G["rotation curve"]['V']))*1.5)
    else:
        Vobs = model(x[:-2] + [0, 0], G["appR"][f"{eval_at_R}:{eval_at_band}"])
    incl_corr = np.sin(G["inclination"][f"{eval_at_R}:{eval_at_band}"])
    Vcorr = abs(Vobs) / incl_corr
    Vcorr_E = G["rotation curve"]["V_e"][
        np.argmin(
            np.abs(
                np.abs(G["rotation curve"]["R"])
                - G["appR"][f"{eval_at_R}:{eval_at_band}"]
            )
        )
    ]

    G["V"][f"{eval_with_model}:{eval_at_R}:{eval_at_band}"] = Vcorr
    G["V"][f"E|{eval_with_model}:{eval_at_R}:{eval_at_band}"] = Vcorr_E

    return G


@catch_errors
def Calc_Colour_Profile(G, eval_in_colour1="g", eval_in_colour2="r"):
    """
    Compute a profile giving colour density as a function of radius

    references: None
    """
    maxR = min(
        G["photometry"][eval_in_colour1]["R"][-1],
        G["photometry"][eval_in_colour2]["R"][-1],
    )
    subprof1 = G["photometry"][eval_in_colour1]["R"] <= maxR
    subprof2 = G["photometry"][eval_in_colour2]["R"] <= maxR
    subR = [
        min(
            G["photometry"][eval_in_colour1]["R"][0],
            G["photometry"][eval_in_colour2]["R"][0],
        )
    ]
    i1 = 0
    i2 = 0
    while subR[-1] < (maxR - 1e-3):
        if np.isclose(subR[-1], G["photometry"][eval_in_colour1]["R"][i1]) or G[
            "photometry"
        ][eval_in_colour1]["R"][i1] < (subR[-1] - 1e-3):
            i1 += 1
        else:
            subR.append(G["photometry"][eval_in_colour1]["R"][i1])
        if np.isclose(subR[-1], G["photometry"][eval_in_colour2]["R"][i2]) or G[
            "photometry"
        ][eval_in_colour2]["R"][i2] < (subR[-1] - 1e-3):
            i2 += 1
        else:
            subR.append(G["photometry"][eval_in_colour2]["R"][i2])
    subR = np.array(subR)

    G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"] = {}
    G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["R"] = subR
    G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["col"] = np.interp(
        subR,
        G["photometry"][eval_in_colour1]["R"][subprof1],
        G["photometry"][eval_in_colour1]["SB"][subprof1],
    ) - np.interp(
        subR,
        G["photometry"][eval_in_colour2]["R"][subprof2],
        G["photometry"][eval_in_colour2]["SB"][subprof2],
    )
    G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["col_e"] = np.sqrt(
        np.interp(
            subR,
            G["photometry"][eval_in_colour1]["R"][subprof1],
            G["photometry"][eval_in_colour1]["SB_e"][subprof1],
        )
        ** 2
        + np.interp(
            subR,
            G["photometry"][eval_in_colour2]["R"][subprof2],
            G["photometry"][eval_in_colour2]["SB_e"][subprof2],
        )
        ** 2
    )
    G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["totcol"] = np.interp(
        subR,
        G["photometry"][eval_in_colour1]["R"][subprof1],
        G["photometry"][eval_in_colour1]["totmag"][subprof1],
    ) - np.interp(
        subR,
        G["photometry"][eval_in_colour2]["R"][subprof2],
        G["photometry"][eval_in_colour2]["totmag"][subprof2],
    )
    G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["totcol_e"] = np.interp(
        subR,
        G["photometry"][eval_in_colour1]["R"][subprof1],
        G["photometry"][eval_in_colour1]["totmag_e"][subprof1],
    ) - np.interp(
        subR,
        G["photometry"][eval_in_colour2]["R"][subprof2],
        G["photometry"][eval_in_colour2]["totmag_e"][subprof2],
    )

    return G

@all_bands
@all_appR
@catch_errors
def Calc_Apparent_Magnitude(G, eval_in_band=None, eval_at_R=None, eval_at_band=None):
    """
    Compute the total apparent magnitude within a requested radius and at a requested band.

    references: Calc_Apparent_Radius
    """

    # if extrapolating to infinity, do evaluations at R23.5 and
    # extrapolate at the end
    if eval_at_R == "RI":
        eval_at_R = "Rp80"
        to_inf = True
    else:
        to_inf = False

    sbR = G["photometry"][eval_in_band]["R"]
    sbSB = G["photometry"][eval_in_band]["SB"]

    Mag = Evaluate_Magnitude(
        sbR,
        G["photometry"][eval_in_band]["totmag"],
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        E=G["photometry"][eval_in_band]["totmag_e"],
    )

    # add extrapolation if integrating to infinity
    if to_inf:
        infCHOOSE = sbR > G['appR'][f"Rp60:{eval_at_band}"]
        p = np.polyfit(
            sbR[infCHOOSE],
            sbSB[infCHOOSE],
            1,
        )
        pprime = (-p[0] / 2.5, (22.5 - p[1]) / 2.5)
        prefactor = -2 * np.pi * G["q"][f"{eval_at_R}:{eval_at_band}"]
        L_inf = prefactor * (
            (10 ** (pprime[0] * G["appR"][f"{eval_at_R}:{eval_at_band}"] + pprime[1]))
            * (pprime[0] * G["appR"][f"{eval_at_R}:{eval_at_band}"] * np.log(10) - 1)
            / ((pprime[0] ** 2) * (np.log(10) ** 2))
        )
        Mag = [flux_to_mag(L_inf + mag_to_flux(Mag[0], 22.5), 22.5), Mag[1]]
        eval_at_R = "RI"

    G["appMag"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"] = Mag[0]
    G["appMag"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"] = Mag[1]

    return G

@all_bands
@all_appR
@catch_errors
def Calc_Absolute_Magnitude(G, eval_in_band=None, eval_at_R=None, eval_at_band=None):
    """
    Compute the total absolute magnitude within a requested radius and at a requested band.

    references: Calc_Apparent_Magnitude (at multiple bands for K-correction)
    """
    
    Mag = app_mag_to_abs_mag(
        G["appMag"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"],
        G["distance"] * 1e6,
        G["appMag"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"],
        G["distance_e"] * 1e6,
    )

    G["absMag"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"] = Mag[0]
    G["absMag"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"] = Mag[1]

    return G


@all_bands
@all_appR
@catch_errors
def Calc_Luminosity(G, eval_in_band=None, eval_at_R=None, eval_at_band=None):
    """
    Compute the total luminosity within a requested radius and at a requested band.

    references: Calc_Absolute_Magnitude
    """

    Lum = mag_to_L(
        G["absMag"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"],
        eval_in_band,
        mage=G["absMag"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"],
    )

    G["L"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"] = Lum[0]
    G["L"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"] = Lum[1]

    return G


@all_appR
@catch_errors
def Calc_Colour_within(
    G, eval_in_colour1="g", eval_in_colour2="r", eval_at_R=None, eval_at_band=None
):
    """
    Compute the colour within a requested radius and at a requested pair of bands.

    references: Calc_Colour_Profile
    """

    if eval_at_R == 'RI':
        Col = [G['appMag'][f"{eval_in_colour1}|{eval_at_R}:{eval_at_band}"] - G['appMag'][f"{eval_in_colour2}|{eval_at_R}:{eval_at_band}"],
               np.sqrt(G['appMag'][f"{eval_in_colour1}:E|{eval_at_R}:{eval_at_band}"]**2 + G['appMag'][f"{eval_in_colour2}:E|{eval_at_R}:{eval_at_band}"]**2)]
    else:
        Col = [
            np.interp(
                G["appR"][f"{eval_at_R}:{eval_at_band}"],
                G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["R"],
                G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["totcol"],
            ),
            np.interp(
                G["appR"][f"{eval_at_R}:{eval_at_band}"],
                G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["R"],
                G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["totcol_e"],
            )
        ]
    G["Col_in"][f"{eval_in_colour1}:{eval_in_colour2}|{eval_at_R}:{eval_at_band}"] = Col[0]
    G["Col_in"][f"{eval_in_colour1}:{eval_in_colour2}:E|{eval_at_R}:{eval_at_band}"] = Col[1]

    return G


@all_appR
@catch_errors
def Calc_Colour_at(
    G, eval_in_colour1="g", eval_in_colour2="r", eval_at_R=None, eval_at_band=None
):
    """
    Compute the colour at a requested radius and at a requested pair of bands.

    references: Calc_Colour_Profile
    """
    if eval_at_R == 'RI':
        return G

    G["Col_at"][
        f"{eval_in_colour1}:{eval_in_colour2}|{eval_at_R}:{eval_at_band}"
    ] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["R"],
        G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["col"],
    )
    G["Col_at"][
        f"{eval_in_colour1}:{eval_in_colour2}:E|{eval_at_R}:{eval_at_band}"
    ] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["R"],
        G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["col_e"],
    )

    return G


@catch_errors
def Calc_Mass_to_Light_Profile(
    G, eval_in_band="g", eval_in_colour1="g", eval_in_colour2="r"
):
    """
    Compute a profile of stellar mass-to-light ratios at all radii

    references: Calc_Colour_Profile
    """

    if eval_in_band in ["g", "r", "i", "z", "H"]:
        m2l_table = "Roediger_BC03"
    elif eval_in_band in ["w1", "w2"]:
        m2l_table = "Cluver_2014"
    G["M2L_prof"][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"] = {}
    G["M2L_prof"][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"]["R"] = G[
        "Col_prof"
    ][f"{eval_in_colour1}:{eval_in_colour2}"]["R"]
    # M2L in
    #---------------------------------------------------------------------
    M2L = Get_M2L(
        G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["totcol"],
        f"{eval_in_colour1}-{eval_in_colour2}",
        eval_in_band,
        m2l_table,
        colour_err=G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["totcol_e"],
    )
    G["M2L_prof"][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"][
        "M2L_in"
    ] = M2L[0]
    G["M2L_prof"][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"][
        "M2L_in_e"
    ] = M2L[1]
    #M2L inf
    #---------------------------------------------------------------------
    M2L = Get_M2L(
        G['Col_in'][f"{eval_in_colour1}:{eval_in_colour2}|RI:r"],
        f"{eval_in_colour1}-{eval_in_colour2}",
        eval_in_band,
        m2l_table,
        colour_err = G['Col_in'][f"{eval_in_colour1}:{eval_in_colour2}:E|RI:r"], # fixme, specify band for RI
    )
    G['M2L_prof'][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"]['M2L_RI'] = M2L[0]
    G['M2L_prof'][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"]['M2L_RI_e'] = M2L[1]

    # M2L at
    #---------------------------------------------------------------------
    M2L = Get_M2L(
        G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["col"],
        f"{eval_in_colour1}-{eval_in_colour2}",
        eval_in_band,
        m2l_table,
        colour_err=G["Col_prof"][f"{eval_in_colour1}:{eval_in_colour2}"]["col_e"],
    )
    G["M2L_prof"][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"][
        "M2L_at"
    ] = M2L[0]
    G["M2L_prof"][f"{eval_in_band}|{eval_in_colour1}:{eval_in_colour2}"][
        "M2L_at_e"
    ] = M2L[1]

    return G


@all_appR
@catch_errors
def Calc_Mass_to_Light_within(
    G,
    eval_in_band=None,
    eval_at_R=None,
    eval_at_band=None,
    eval_at_colour1="g",
    eval_at_colour2="r",
):
    """
    Compute the colour based mass-to-light-ratio within a requested radius and at a requested pair of bands.

    references: Calc_Mass_to_Light_Profile
    """

    if eval_at_R == 'RI':
        G["M2L_in"][f"{eval_in_band}|Col:{eval_at_colour1}:{eval_at_colour2}|{eval_at_R}:{eval_at_band}"] = G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["M2L_RI"]
        G["M2L_in"][f"{eval_in_band}:E|Col:{eval_at_colour1}:{eval_at_colour2}|{eval_at_R}:{eval_at_band}"] = G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["M2L_RI_e"]
        return G
    
    G["M2L_in"][
        f"{eval_in_band}|Col:{eval_at_colour1}:{eval_at_colour2}|{eval_at_R}:{eval_at_band}"
    ] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["R"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["M2L_in"],
    )
    G["M2L_in"][
        f"{eval_in_band}:E|Col:{eval_at_colour1}:{eval_at_colour2}|{eval_at_R}:{eval_at_band}"
    ] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["R"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"][
            "M2L_in_e"
        ],
    )

    return G


@all_appR
@catch_errors
def Calc_Mass_to_Light_at(
    G,
    eval_in_band=None,
    eval_at_R=None,
    eval_at_band=None,
    eval_at_colour1="g",
    eval_at_colour2="r",
):
    """
    Compute the colour based mass-to-light-ratio within a requested radius and at a requested pair of bands.

    references: Calc_Mass_to_Light_Profile
    """

    if eval_at_R == 'RI':
        return G
        
    G["M2L_at"][
        f"{eval_in_band}|Col:{eval_at_colour1}:{eval_at_colour2}|{eval_at_R}:{eval_at_band}"
    ] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["R"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["M2L_at"],
    )
    G["M2L_at"][
        f"{eval_in_band}:E|Col:{eval_at_colour1}:{eval_at_colour2}|{eval_at_R}:{eval_at_band}"
    ] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"]["R"],
        G["M2L_prof"][f"{eval_in_band}|{eval_at_colour1}:{eval_at_colour2}"][
            "M2L_at_e"
        ],
    )

    return G


@catch_errors
def Calc_Concentration(G, eval_at_R1="Rp20", eval_at_R2="Rp80", eval_in_band="r"):
    """
    Compute the light concentration for a requested pair of radii and at a requested band.

    references: Calc_Apparent_Radius
    """

    Con = 5 * np.log10(
        G["appR"][f"{eval_at_R2}:{eval_in_band}"]
        / G["appR"][f"{eval_at_R1}:{eval_in_band}"]
    )
    Con_E = 5 * np.sqrt(
        G["appR"][f"E|{eval_at_R2}:{eval_in_band}"] ** 2
        + G["appR"][f"E|{eval_at_R1}:{eval_in_band}"] ** 2
    )

    G["Con"][f"{eval_at_R1}:{eval_at_R2}:{eval_in_band}"] = Con
    G["Con"][f"E|{eval_at_R1}:{eval_at_R2}:{eval_in_band}"] = Con_E

    return G


@all_bands
@catch_errors
def Calc_Sersic_Params(G, eval_in_band=None):
    """
    Compute the sersic index for a light profile of a given band.

    references: None
    """

    sbR = G["photometry"][eval_in_band]["R"]
    sbSB = G["photometry"][eval_in_band]["SB"]
    S = bulgefit(
        {
            "R": G["photometry"][eval_in_band]["R"],
            "SB": G["photometry"][eval_in_band]["SB"],
            "SB_e": G["photometry"][eval_in_band]["SB_e"],
        },
        np.median(G["photometry"][eval_in_band]["ellip"][-5:]),
    )
    for key in S:
        G["sersic"][f"{key[8:]}|{eval_in_band}"] = S[key]
        G["sersic"][f"{key[8:]}:E|{eval_in_band}"] = 0.0  # fixme error

    return G


@all_bands
@all_appR
@catch_errors
def Calc_Surface_Density_within(
    G, eval_in_band=None, eval_at_R=None, eval_at_band=None
):
    """
    Compute the light surface density within a requested radius and at a requested band.

    references: Calc_Apparent_Radius, Calc_Apparent_Magnitude
    """

    if eval_at_R == 'RI':
        return G
    A = np.pi*G["q"][f"{eval_at_R}:{eval_at_band}"]*G["appR"][f"{eval_at_R}:{eval_at_band}"] ** 2
    Ae1 = (
        G["q"][f"E|{eval_at_R}:{eval_at_band}"]
        * A
        / G["q"][f"{eval_at_R}:{eval_at_band}"]
    )
    Ae2 = (
        2
        * G["appR"][f"E|{eval_at_R}:{eval_at_band}"]
        * A
        / G["appR"][f"{eval_at_R}:{eval_at_band}"]
    )
    SB = surface_brightness(
        m=G["appMag"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"],
        A=A,
        me=G["appMag"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"],
        Ae=np.sqrt(Ae1 ** 2 + Ae2 ** 2),
    )

    G["SD_in"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"] = SB[0]
    G["SD_in"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"] = SB[1]

    return G


@all_bands
@all_appR
@catch_errors
def Calc_Surface_Density_at(G, eval_in_band=None, eval_at_R=None, eval_at_band=None):
    """
    Compute the light surface density within a requested radius and at a requested band.

    references: Calc_Apparent_Radius
    """

    if eval_at_R == 'RI':
        return G
    G["SD_at"][f"{eval_in_band}|{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["photometry"][eval_in_band]["R"],
        G["photometry"][eval_in_band]["SB"],
    )
    G["SD_at"][f"{eval_in_band}:E|{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["photometry"][eval_in_band]["R"],
        G["photometry"][eval_in_band]["SB_e"],
    )

    return G

@catch_errors
def Calc_Stellar_Mass_Profile(
    G,
    eval_in_bands=["r", "g", "z", "w1"],
    eval_in_colours1=["g", "g", "r", "w1"],
    eval_in_colours2=["r", "z", "z", "w2"],
):
    """
    Calculates a profile of total integrated stellar mass

    references: Calc_Mass_to_Light_Profile
    """

    subR = [min(G["photometry"][b]["R"][0] for b in eval_in_bands)]
    maxR = min(G["photometry"][b]["R"][-1] for b in eval_in_bands)
    ind = np.zeros(len(eval_in_bands), dtype=int)
    while subR[-1] < (maxR - 1e-3):
        options = []
        for i, b in enumerate(eval_in_bands):
            if np.isclose(subR[-1], G["photometry"][b]["R"][ind[i]]):
                ind[i] += 1
            else:
                options.append(G["photometry"][b]["R"][ind[i]])
        subR.append(min(options))
        for i, b in enumerate(eval_in_bands):
            if np.isclose(subR[-1], G["photometry"][b]["R"][ind[i]]):
                ind[i] += 1
    subR = np.array(subR)

    if "distance" in G:
        mass_estimates = np.zeros((len(subR), len(eval_in_bands)))
        for i, b, c1, c2 in zip(
            range(len(eval_in_bands)), eval_in_bands, eval_in_colours1, eval_in_colours2
        ):
            L = mag_to_L(
                app_mag_to_abs_mag(G["photometry"][b]["totmag"], G["distance"] * 1e6),
                band=b,
            )
            mass_estimates[:, i] = (
                np.interp(
                    subR,
                    G["M2L_prof"][f"{b}|{c1}:{c2}"]["R"],
                    G["M2L_prof"][f"{b}|{c1}:{c2}"]["M2L_in"],
                )
                * np.interp(subR, G["photometry"][b]["R"], L)
            )
    mass_dens_estimates = np.zeros((len(subR), len(eval_in_bands)))
    for i, b, c1, c2 in zip(
        range(len(eval_in_bands)), eval_in_bands, eval_in_colours1, eval_in_colours2
    ):
        I = muSB_to_ISB(G["photometry"][b]["SB"], band=b)
        mass_dens_estimates[:, i] = (
            np.interp(
                subR,
                G["M2L_prof"][f"{b}|{c1}:{c2}"]["R"],
                G["M2L_prof"][f"{b}|{c1}:{c2}"]["M2L_in"],
            )
            * np.interp(subR, G["photometry"][b]["R"], I)
        )

    G["Mstar_prof"]["R"] = subR
    if "distance" in G:
        G["Mstar_prof"]["Mstar"] = np.mean(mass_estimates, axis=1)
        G["Mstar_prof"]["Mstar_e"] = np.std(mass_estimates, axis=1)
        RI_estimates = list(G["M2L_prof"][f"{b}|{c1}:{c2}"]["M2L_RI"]*G['L'][f"{b}|RI:r"] for b, c1, c2 in zip(eval_in_bands, eval_in_colours1, eval_in_colours2))
        G["Mstar_prof"]['Mstar_RI'] = np.mean(RI_estimates)
        G["Mstar_prof"]['Mstar_RI_e'] = np.std(RI_estimates)
    G["Mstar_prof"]["MstarDens"] = np.mean(mass_dens_estimates, axis=1)
    G["Mstar_prof"]["MstarDens_e"] = np.std(mass_dens_estimates, axis=1)

    return G


@all_appR
@catch_errors
def Calc_Stellar_Mass(G, eval_at_R=None, eval_at_band=None):

    if eval_at_R == 'RI':
        G["Mstar"][f"{eval_at_R}:{eval_at_band}"] = G["Mstar_prof"]['Mstar_RI']
        G["Mstar"][f"E|{eval_at_R}:{eval_at_band}"] = G["Mstar_prof"]['Mstar_RI_e']
        return G
    
    G["Mstar"][f"{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["Mstar_prof"]["R"],
        G["Mstar_prof"]["Mstar"],
    )
    G["Mstar"][f"E|{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["Mstar_prof"]["R"],
        G["Mstar_prof"]["Mstar_e"],
    )

    return G


@catch_errors
def Calc_Stellar_Mass_Density_Radius(G, eval_at_R="Rd1", eval_at_tracer="*"):
    """
    calculates the radius at which a certain stellar mass density is reached

    references: Calc_Stellar_Mass_Profile
    """

    R = Isophotal_Radius(
        G["Mstar_prof"]["R"],
        -np.log10(G["Mstar_prof"]["MstarDens"]),
        -np.log10(float(eval_at_R[2:])),
        E=G["Mstar_prof"]["MstarDens_e"] / (np.log(10) * G["Mstar_prof"]["MstarDens"]),
    )

    G["appR"][f"{eval_at_R}:{eval_at_tracer}"] = R[0]
    G["appR"][f"E|{eval_at_R}:{eval_at_tracer}"] = R[1]

    return G


@catch_errors
def Calc_Dynamical_Mass_Profile(G, eval_in_band="r", eval_with_model="C97"):
    """
    calculates a profile of dynamical mass values using newtonian M = RV^2 / G

    references: Calc_Stellar_Mass_Profile, Calc_C97_Velocity_Fit, Calc_Tan_Velocity_Fit
    """

    R = G["Mstar_prof"]["R"]
    x = list(
        G["rc_model"][f"{eval_with_model}:{p}"]
        for p in G["rc_model"][f"{eval_with_model}:param order"]
    )
    if eval_with_model == "C97":
        model = Courteau97_Model_Evaluate
    elif eval_with_model == "Tan":
        model = Tan_Model_Evaluate
    else:
        raise ValueError(f"unrecognized model type {eval_with_model}")
    Vobs = model(x[:-2] + [0, 0], R)
    incl = np.interp(
        R,
        G['inclination_prof']["R"],
        G['inclination_prof']["inclination"],
    )
    incl_e = np.interp(
        R,
        G['inclination_prof']["R"],
        G['inclination_prof']["inclination_e"],
    )
    Vcorr = abs(Vobs) / np.sin(incl)
    Vcorr_E = np.interp(
        R, np.abs(G["rotation curve"]["R"]), G["rotation curve"]["V_e"]
    ) / np.sin(incl)

    physR = np.array(list(list(arcsec_to_pc(rr, G["distance"] * 1e6, De=G["distance_e"] * 1e6)) for rr in R)) 
    G["Mdyn_prof"]["R"] = R
    G["Mdyn_prof"]["M"] = physR[:,0] * ((Vcorr*1e3) ** 2) / Grav_C
    G["Mdyn_prof"]["M_e"] = np.sqrt(
        (2 * Vcorr_E * G["Mdyn_prof"]["M"] / Vcorr) ** 2
        + (physR[:,1] * G["Mdyn_prof"]["M"] / physR[:,0]) ** 2
        + (2 * incl_e * G["Mdyn_prof"]["M"] / np.tan(incl))
    )

    return G


@all_appR
@catch_errors
def Calc_Dynamical_Mass(G, eval_at_R=None, eval_at_band=None):
    """
    computes the dynamical mass at a specific radius

    references: Calc_Dynamical_Mass_Profile
    """

    if eval_at_R == 'RI':
        return G
    
    G["Mdyn"][f"{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["Mdyn_prof"]["R"],
        G["Mdyn_prof"]["M"],
    )
    G["Mdyn"][f"E|{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["Mdyn_prof"]["R"],
        G["Mdyn_prof"]["M_e"],
    )
    
    return G


@catch_errors
def Calc_Angular_Momentum_Profile(G, eval_in_band="r", eval_with_model="C97"):
    """

    references: Calc_C97_Velocity_Fit, Calc_Tan_Velocity_Fit
    """

    x = list(
        G["rc_model"][f"{eval_with_model}:{p}"]
        for p in G["rc_model"][f"{eval_with_model}:param order"]
    )
    if eval_with_model == "C97":
        model = Courteau97_Model_Evaluate
    elif eval_with_model == "Tan":
        model = Tan_Model_Evaluate
    else:
        raise ValueError(f"unrecognized model type {eval_with_model}")

    dJ = lambda r: (
        arcsec_to_pc(r, G["distance"] * 1e6)
        * abs(
            model(x[:-2] + [0, 0], r)
            / (
                np.sin(
                    np.interp(
                        r,
                        G['inclination_prof']["R"],
                        G['inclination_prof']["inclination"],
                    )
                )
            )
        )
        ** 3
    ) / (1e3 * Grav_C / 1e6)
    dJerr = lambda r: (
        3
        * arcsec_to_pc(r, G["distance"] * 1e6)
        * abs(
            model(x[:-2] + [0, 0], r)
            / (
                np.sin(
                    np.interp(
                        r,
                        G['inclination_prof']["R"],
                        G['inclination_prof']["inclination"],
                    )
                )
            )
        )
        ** 2
    ) / (1e3 * Grav_C / 1e6)

    J = []
    Jerr = []
    for R in G["Mstar_prof"]["R"]:
        J.append(quad(dJ, 0, R)[0])
        V = model(x[:-2] + [0, 0], R)
        Jerr1 = quad(dJerr, 0, R)[0] * np.interp(
            R, G["rotation curve"]["R"], G["rotation curve"]["V_e"]
        )
        pR = arcsec_to_pc(R, G["distance"] * 1e6, G["distance_e"] * 1e6)
        Jerr2 = pR[1] * pR[0] * V ** 3 * 1e3 * pR[0] / (Grav_C / 1e6)
        Jerr.append(np.sqrt(Jerr1 ** 2 + Jerr2 ** 2))

    G["AngMom_prof"]["R"] = G["Mstar_prof"]["R"]
    G["AngMom_prof"]["J"] = np.array(J)
    G["AngMom_prof"]["J_e"] = np.array(Jerr)  # fixme check

    return G


@all_appR
@catch_errors
def Calc_Angular_Momentum(G, eval_at_R=None, eval_at_band=None):


    if eval_at_R == 'RI':
        return G
    
    G["AM"][f"{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["AngMom_prof"]["R"],
        G["AngMom_prof"]["J"],
    )
    G["AM"][f"E|{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["AngMom_prof"]["R"],
        G["AngMom_prof"]["J_e"],
    )

    return G


@catch_errors
def Calc_Stellar_Angular_Momentum_Profile(
    G,
    eval_in_band="r",
    eval_with_model="C97",
):

    x = list(
        G["rc_model"][f"{eval_with_model}:{p}"]
        for p in G["rc_model"][f"{eval_with_model}:param order"]
    )
    if eval_with_model == "C97":
        model = Courteau97_Model_Evaluate
    elif eval_with_model == "Tan":
        model = Tan_Model_Evaluate
    else:
        raise ValueError(f"unrecognized model type {eval_with_model}")

    dJs = (
        lambda r: 2
        * np.pi
        * arcsec_to_pc(r, G["distance"] * 1e6) ** 2
        * np.interp(r, G["Mstar_prof"]["R"], G["Mstar_prof"]["MstarDens"])
        * abs(
            model(x[:-2] + [0, 0], r)
            / (
                np.sin(
                    np.interp(
                        r,
                        G['inclination_prof']["R"],
                        G['inclination_prof']["inclination"],
                    )
                )
            )
        )
    )

    dJserr_Ms = (
        lambda r: 2
        * np.pi
        * arcsec_to_pc(r, G["distance"] * 1e6) ** 2
        * abs(
            model(x[:-2] + [0, 0], r)
            / (
                np.sin(
                    np.interp(
                        r,
                        G['inclination_prof']["R"],
                        G['inclination_prof']["inclination"],
                    )
                )
            )
        )
    )
    dJserr_V = (
        lambda r: 2
        * np.pi
        * arcsec_to_pc(r, G["distance"] * 1e6) ** 2
        * np.interp(r, G["Mstar_prof"]["R"], G["Mstar_prof"]["MstarDens"])
    )

    Js = []
    Jserr = []
    for R in G["Mstar_prof"]["R"]:
        Js.append(quad(dJs, 0, R)[0])
        Jserr1 = quad(dJserr_Ms, 0, R)[0] * np.interp(
            R, G["Mstar_prof"]["R"], G["Mstar_prof"]["MstarDens_e"]
        )
        Jserr2 = quad(dJserr_V, 0, R)[0] * np.interp(
            R, G["rotation curve"]["R"], G["rotation curve"]["V_e"]
        )
        V = model(x[:-2] + [0, 0], R)
        pR = arcsec_to_pc(R, G["distance"] * 1e6, G["distance_e"] * 1e6)
        Jserr3 = (
            pR[1]
            * np.interp(R, G["Mstar_prof"]["R"], G["Mstar_prof"]["MstarDens"])
            * V
            * (1e3 * pR[0]) ** 2
            / (Grav_C / 1e6)
        )
        Jserr.append(np.sqrt(Jserr1 ** 2 + Jserr2 ** 2 + Jserr3 ** 2))

    G["AngMomStar_prof"]["R"] = G["Mstar_prof"]["R"]
    G["AngMomStar_prof"]["Js"] = np.array(Js)
    G["AngMomStar_prof"]["Js_e"] = np.array(Jserr)  # fixme check

    return G


@all_appR
@catch_errors
def Calc_Stellar_Angular_Momentum(
    G,
    eval_at_R=None,
    eval_at_band=None,
):

    if eval_at_R == 'RI':
        return G
    
    G["AMstar"][f"{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["AngMomStar_prof"]["R"],
        G["AngMomStar_prof"]["Js"],
    )
    G["AMstar"][f"E|{eval_at_R}:{eval_at_band}"] = np.interp(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["AngMomStar_prof"]["R"],
        G["AngMomStar_prof"]["Js_e"],
    )

    return G
