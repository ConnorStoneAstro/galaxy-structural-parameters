import numpy as np
from .Profile_Functions import Isophotal_Radius
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def Apply_Cosmological_Dimming(G):

    G["cosmological dimming corr"] = -2.5 * np.log10((1 + G["zhel"]) ** 3)

    for b in G["SB"]:
        G["SB"][b]["sb"] += G["cosmological dimming corr"]

    return G


def Apply_Extinction_Correction(G):

    for b in G["SB"]:
        G["SB"][b]["sb"] -= G["SB"][b]["extinction"]

    return G


def Apply_Profile_Truncation(G):

    for b in G["SB"]:
        CHOOSE = np.logical_and(G["SB"][b]["sb"] < 90, G["SB"][b]["sb E"] < 0.3)
        truncR = G["SB"][b]["R"][-1] * 1.01
        Rstart = Isophotal_Radius(G["SB"][b]["R"][CHOOSE], G["SB"][b]["sb"][CHOOSE], G["SB"][b]["sb"][CHOOSE][0] + 5)[0]

        def linear_floor(x, R, SB):
            return np.mean(np.abs(SB - np.clip(x[0]*R + x[1],a_min = None, a_max = x[2])))

        outer_CHOOSE = np.logical_and(CHOOSE, G["SB"][b]["R"] > Rstart)
        x0 = list(np.polyfit(G["SB"][b]["R"][outer_CHOOSE], G["SB"][b]["sb"][outer_CHOOSE], 1))+[28]
        res = minimize(linear_floor, x0 = x0, args = (G["SB"][b]["R"][outer_CHOOSE], G["SB"][b]["sb"][outer_CHOOSE]), method = 'Nelder-Mead')
        if res.success:
            truncR = (res.x[2] - res.x[1]) / res.x[0]
        else:
            for i in range(np.argmin(np.abs(G["SB"][b]["R"] - Rstart)), len(G["SB"][b]["R"])):
                if G["SB"][b]["sb"][i] > 90:
                    truncR = G["SB"][b]["R"][i]
        CHOOSE[G["SB"][b]["R"] > truncR] = False
        
        G["SB"][b]["R"] = G["SB"][b]["R"][CHOOSE]
        G["SB"][b]["sb"] = G["SB"][b]["sb"][CHOOSE]
        G["SB"][b]["sb E"] = G["SB"][b]["sb E"][CHOOSE]
        try:
            G["SB"][b]["m"] = G["SB"][b]["m"][CHOOSE]
            G["SB"][b]["m E"] = G["SB"][b]["m E"][CHOOSE]
        except KeyError:
            pass
        try:
            G["SB"][b]["q"] = G["SB"][b]["q"][CHOOSE]
            G["SB"][b]["q E"] = G["SB"][b]["q E"][CHOOSE]
        except KeyError:
            pass

    return G
