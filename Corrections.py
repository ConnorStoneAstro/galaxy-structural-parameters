import numpy as np


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
        truncR = G["SB"][b]["R"][-1] * 1.01
        for i in range(int(len(G["SB"][b]["R"]) / 4), len(G["SB"][b]["R"])):
            if G["SB"][b]["sb"][i] > 90:
                truncR = G["SB"][b]["R"][i]

        CHOOSE = np.logical_and(G["SB"][b]["sb"] < 90, G["SB"][b]["sb E"] < 0.3)
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
