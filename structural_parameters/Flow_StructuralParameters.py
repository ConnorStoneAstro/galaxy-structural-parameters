import sys
import os
from .StructuralParameters import (
    Calc_Apparent_Radius,
    Calc_Physical_Radius,
    Calc_Axis_Ratio,
    Calc_Inclination,
    Calc_C97_Velocity_Fit,
    Calc_Tan_Velocity_Fit,
    Calc_Velocity,
    Calc_Apparent_Magnitude,
    Calc_Absolute_Magnitude,
    Calc_Luminosity,
    Calc_Colour,
    Calc_Mass_to_Light,
    Calc_Surface_Density,
    Calc_Sersic_Params,
    Calc_Concentration,
)
from .Corrections import (
    Apply_Cosmological_Dimming,
    Apply_Extinction_Correction,
    Apply_Profile_Truncation,
    Apply_Redshift_Velocity_Correction,
    Decide_Bypass,
)
from .Diagnostic_Plots import (
    Plot_Photometry,
    Plot_Radii,
    Plot_Velocity,
)
from functools import partial
import numpy as np
sys.path.append(os.environ["PROGRAMMING"])
from FlowChart import flow

# Structural Parameters Flowchart
######################################################################
Structural_Parameters = flow.Chart("structural parameters", logfile = 'structural_parameters.log')
Structural_Parameters.linear_mode(True)

# Apply corrections to photometry
######################################################################
Photometry_Corrections = flow.Chart("photometry corrections")
Photometry_Corrections.linear_mode(True)
Photometry_Corrections.add_process_node(
    "cosmological dimming corr", Apply_Cosmological_Dimming
)
Photometry_Corrections.add_process_node(
    "extinction correction", Apply_Extinction_Correction
)
Photometry_Corrections.add_process_node(
    "profile truncation", Apply_Profile_Truncation
)
Photometry_Corrections.linear_mode(False)
Structural_Parameters.add_node(Photometry_Corrections)
Structural_Parameters.add_process_node('plot photometry', Plot_Photometry)

# Individual Structural Parameters
######################################################################
eval_radii = list(f"Ri{rr:g}" for rr in np.arange(22, 26.5, 0.5)) + list(
    f"Re{rr:g}" for rr in np.arange(20, 90, 10)
)
eval_bands = ["g", "r", "z"]

# Apparent Radius
Apparent_Radius = flow.Chart("apparent radius")
Apparent_Radius.linear_mode(True)
for er in eval_radii:
    for eb in eval_bands:
        Apparent_Radius.add_process_node(
            f"apparent radius:{er}:{eb}",
            partial(Calc_Apparent_Radius, eval_at_R=er, eval_at_band=eb),
        )
Apparent_Radius.linear_mode(False)
Structural_Parameters.add_node(Apparent_Radius)
Structural_Parameters.add_process_node('plot radii', Plot_Radii)

# Axis Ratio
Axis_Ratio = flow.Chart("axis ratio")
Axis_Ratio.linear_mode(True)
for er in eval_radii:
    for eb in eval_bands:
        Axis_Ratio.add_process_node(
            f"axis ratio:{er}:{eb}",
            partial(Calc_Axis_Ratio, eval_at_R=er, eval_at_band=eb),
        )
Axis_Ratio.linear_mode(False)
Structural_Parameters.add_node(Axis_Ratio)

# Inclination
Inclination = flow.Chart("inclination")
Inclination.linear_mode(True)
for er in eval_radii:
    for eb in eval_bands:
        Inclination.add_process_node(
            f"inclination:{er}:{eb}",
            partial(Calc_Inclination, eval_at_R=er, eval_at_band=eb),
        )
Inclination.linear_mode(False)
Structural_Parameters.add_node(Inclination)

# Concentration
Concentration = flow.Chart("concentration")
Concentration.linear_mode(True)
for er1, er2 in [("Re20", "Re80"), ("Re50", "Re80"), ("Ri23.5", "Ri26")]:
    for eb in eval_bands:
        Concentration.add_process_node(
            f"concentration:{er1}:{er2}:{eb}",
            partial(
                Calc_Concentration, eval_at_R1=er1, eval_at_R2=er2, eval_at_band=eb
            ),
        )
Concentration.linear_mode(False)
Structural_Parameters.add_node(Concentration)

# Apparent Magnitude
Apparent_Magnitude = flow.Chart("apparent magnitude")
Apparent_Magnitude.linear_mode(True)
for er in eval_radii + ["Rinf"]:
    for eb in eval_bands:
        Apparent_Magnitude.add_process_node(
            f"apparent magnitude:{er}:{eb}",
            partial(Calc_Apparent_Magnitude, eval_at_R=er, eval_at_band=eb),
        )
Apparent_Magnitude.linear_mode(False)
Structural_Parameters.add_node(Apparent_Magnitude)

# Colour
Colour = flow.Chart("colour")
Colour.linear_mode(True)
for er in eval_radii:
    for eb1, eb2 in [("z", "r"), ("z", "g"), ("r", "g")]:
        Colour.add_process_node(
            f"colour:{er}:{eb1}:{eb2}",
            partial(Calc_Colour, eval_at_R=er, eval_at_band1=eb1, eval_at_band2=eb2),
        )
Colour.linear_mode(False)
Structural_Parameters.add_node(Colour)

# Mass to Light
Mass_to_Light = flow.Chart("mass to light")
Mass_to_Light.linear_mode(True)
for er in eval_radii:
    for eb1, eb2 in [("z", "r"), ("z", "g"), ("r", "g")]:
        Mass_to_Light.add_process_node(
            f"mass to light:{er}:{eb1}:{eb2}",
            partial(
                Calc_Mass_to_Light, eval_at_R=er, eval_at_band1=eb1, eval_at_band2=eb2
            ),
        )
Mass_to_Light.linear_mode(False)
Structural_Parameters.add_node(Mass_to_Light)

# Surface Density
Surface_Density = flow.Chart("surface density")
Surface_Density.linear_mode(True)
for er in eval_radii:
    for eb in eval_bands:
        Surface_Density.add_process_node(
            f"surface density:{er}:{eb}",
            partial(Calc_Surface_Density, eval_at_R=er, eval_at_band=eb),
        )
Surface_Density.linear_mode(False)
Structural_Parameters.add_node(Surface_Density)

# Sersic Params
Sersic_Params = flow.Chart("sersic params")
Sersic_Params.linear_mode(True)
for eb in eval_bands:
    Sersic_Params.add_process_node(
        f"sersic params:{eb}",
        partial(Calc_Sersic_Params, eval_at_band=eb),
    )
Sersic_Params.linear_mode(False)
Structural_Parameters.add_node(Sersic_Params)

# Distance Dependent Quantities
######################################################################

# Physical Radius
Physical_Radius = flow.Chart("physical radius")
Physical_Radius.linear_mode(True)
for er in eval_radii:
    for eb in eval_bands:
        Physical_Radius.add_process_node(
            f"physical radius:{er}:{eb}",
            partial(Calc_Physical_Radius, eval_at_R=er, eval_at_band=eb),
        )
Physical_Radius.linear_mode(False)
Structural_Parameters.add_node(Physical_Radius)

# Absolute Magnitude
Absolute_Magnitude = flow.Chart("absolute magnitude")
Absolute_Magnitude.linear_mode(True)
for er in eval_radii + ["Rinf"]:
    for eb in eval_bands:
        Absolute_Magnitude.add_process_node(
            f"absolute magnitude:{er}:{eb}",
            partial(Calc_Absolute_Magnitude, eval_at_R=er, eval_at_band=eb),
        )
Absolute_Magnitude.linear_mode(False)
Structural_Parameters.add_node(Absolute_Magnitude)

# Luminosity
Luminosity = flow.Chart("luminosity")
Luminosity.linear_mode(True)
for er in eval_radii + ["Rinf"]:
    for eb in eval_bands:
        Luminosity.add_process_node(
            f"luminosity:{er}:{eb}",
            partial(Calc_Luminosity, eval_at_R=er, eval_at_band=eb),
        )
Luminosity.linear_mode(False)
Structural_Parameters.add_node(Luminosity)

# Velocity Dependent Quantities
######################################################################

# Apply corrections to Velocity
Velocity_Corrections = flow.Chart("velocity corrections")
Velocity_Corrections.linear_mode(True)
Velocity_Corrections.add_process_node(
    "redshift velocity corr", Apply_Redshift_Velocity_Correction
)
Velocity_Corrections.linear_mode(False)
Structural_Parameters.add_node(Velocity_Corrections)

# Fit Velocity
Velocity_Fit = flow.Chart("velocity fits")
Velocity_Fit.linear_mode(True)
Velocity_Fit.add_process_node('fit C97 model', Calc_C97_Velocity_Fit)
Velocity_Fit.add_process_node('fit Tan model', Calc_Tan_Velocity_Fit)
Velocity_Fit.linear_mode(False)
Structural_Parameters.add_node(Velocity_Fit)
Structural_Parameters.add_process_node('plot velocity', Plot_Velocity)

# Velocity
Velocity = flow.Chart("velocity")
Velocity.linear_mode(True)
for er in eval_radii:
    for eb in eval_bands:
        Velocity.add_process_node(
            f"velocity:{er}:{eb}", partial(Calc_Velocity, eval_at_R=er, eval_at_band=eb)
        )
Velocity.linear_mode(False)
Structural_Parameters.add_node(Velocity)

# Close flowchart construction
######################################################################
Structural_Parameters.linear_mode(False)

# Pipeline Decisions
######################################################################
# velocity bypass
Structural_Parameters.add_decision_node('bypass velocity', partial(Decide_Bypass, check_key = "RC"))
Structural_Parameters.insert_node('bypass velocity', "velocity corrections")
Structural_Parameters.link_nodes('bypass velocity', Velocity.forward.name)

# Absolute quantities bypass
Structural_Parameters.add_decision_node('bypass distance dependent', partial(Decide_Bypass, check_key = "D"))
Structural_Parameters.insert_node('bypass distance dependent', "physical radius")
Structural_Parameters.link_nodes('bypass distance dependent', Luminosity.forward.name)

######################################################################
# Structural_Parameters.draw("plots/Structural_Parameters_Flowchart.png")
