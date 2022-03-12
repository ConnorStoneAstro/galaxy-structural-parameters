import sys
import os
from .StructuralParameters import *
from .Corrections import (
    Apply_Cosmological_Dimming,
    Apply_Extinction_Correction,
    Apply_Profile_Truncation,
    Apply_Redshift_Velocity_Correction,
    Apply_Inclination_Correction,
    Decide_Bypass,
)
from .Diagnostic_Plots import (
    Plot_Photometry,
    Plot_Radii,
    Plot_Velocity,
)
from .Supporting_Functions import allradii
from functools import partial
from collections import defaultdict
import numpy as np
sys.path.append(os.environ["PROGRAMMING"])
from FlowChart import flow

def to_defaultdict(G):
    if isinstance(G, defaultdict):
        return G
    newG = defaultdict(dict)
    newG.update(G)
    return newG
    
def Build_Inclination_Correction_Prep(eval_after_R = None, eval_after_band = None):
    # Structural Parameters Flowchart
    ######################################################################
    Incl_Prep = flow.Chart("inclination preparation", logfile = 'structural_parameters.log')
    Incl_Prep.linear_mode(True)

    # Convert to defaultdict
    ######################################################################
    Incl_Prep.add_process_node("prime galaxy dict", to_defaultdict)
    
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
    Incl_Prep.add_node(Photometry_Corrections)

    # Individual Structural Parameters
    ######################################################################
    # Apparent Radius
    Incl_Prep.add_process_node(
        "reference radius",
        partial(Calc_Apparent_Radius, eval_at_R=eval_after_R, eval_at_band=eval_after_band),
    )

    # Inclination
    Incl_Prep.add_process_node("inclination", partial(Calc_Inclination_Profile, eval_in_band = eval_after_band))

    # Close flowchart construction
    ######################################################################
    Incl_Prep.linear_mode(False)

    return Incl_Prep
    
def Build_Structural_Parameters_Flowchart(
        primary_band = 'r',
        eval_at_radii = allradii,
        eval_at_bands = ['r'],
        colours = [("f", "n"), ("g", "r"), ("g", "z"), ("r", "z"), ("w1", "w2")],
        concentrations = [("Rp20", "Rp80"), ("Ri22", "Ri26"), ("Ri23.5", "Ri26")],
        stellarmass_bands = [('r', 'g', 'z', 'w1'), ('g', 'g', 'r', 'w1'), ('r', 'z', 'z', 'w2')],
        eval_at_density_radii = ["Rd500", "Rd100", "Rd50", "Rd10", "Rd5", "Rd1"],
        incl_corr_specification = None):
    
    # Structural Parameters Flowchart
    ######################################################################
    Structural_Parameters = flow.Chart("structural parameters", logfile = 'structural_parameters.log')
    Structural_Parameters.linear_mode(True)

    # Convert to defaultdict
    ######################################################################
    Structural_Parameters.add_process_node("prime galaxy dict", to_defaultdict)
    
    # Inclination
    Structural_Parameters.add_process_node("inclination profile", partial(Calc_Inclination_Profile, eval_in_band = primary_band))
    
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
    Photometry_Corrections.add_process_node(
        "inclination correction", partial(Apply_Inclination_Correction, specification = incl_corr_specification)
    )
    Photometry_Corrections.linear_mode(False)
    Structural_Parameters.add_node(Photometry_Corrections)
    Structural_Parameters.add_process_node('plot photometry', Plot_Photometry)

    # Individual Structural Parameters
    ######################################################################
    # Apparent Radius
    Apparent_Radius = flow.Chart("apparent radius")
    Apparent_Radius.linear_mode(True)
    for er in eval_at_radii:
        for eb in eval_at_bands:
            Apparent_Radius.add_process_node(
                f"apparent radius:{er}:{eb}",
                partial(Calc_Apparent_Radius, eval_at_R=er, eval_at_band=eb),
            )
    Apparent_Radius.linear_mode(False)
    Structural_Parameters.add_node(Apparent_Radius)
    Structural_Parameters.add_process_node('plot radii', Plot_Radii)

    # Colour Profiles
    Colour_Profiles = flow.Chart("colour profiles")
    Colour_Profiles.linear_mode(True)
    for eb1, eb2 in colours:
        Colour_Profiles.add_process_node(
            f"colour profile:{eb1}:{eb2}",
            partial(Calc_Colour_Profile, eval_in_colour1=eb1, eval_in_colour2=eb2),
        )
    Colour_Profiles.linear_mode(False)
    Structural_Parameters.add_node(Colour_Profiles)

    # Mass to Light profiles
    Mass_to_Light = flow.Chart("mass to light")
    Mass_to_Light.linear_mode(True)
    for b, c1, c2 in zip(*stellarmass_bands):
        Mass_to_Light.add_process_node(
            f"mass to light profile:{b}:{c1}:{c2}",
            partial(
                Calc_Mass_to_Light_Profile, eval_in_band = b, eval_in_colour1=c1, eval_in_colour2=c2
            ),
        )
    Mass_to_Light.linear_mode(False)
    Structural_Parameters.add_node(Mass_to_Light)

    # Stellar Mass profile
    Structural_Parameters.add_process_node("stellar mass profile", partial(
        Calc_Stellar_Mass_Profile,
        eval_in_bands = stellarmass_bands[0],
        eval_in_colours1 = stellarmass_bands[1],
        eval_in_colours2 = stellarmass_bands[2],
    ))

    # Stellar Mass Density Radius
    StellarDensity_Radius = flow.Chart("stellar density radius")
    StellarDensity_Radius.linear_mode(True)
    for er in eval_at_density_radii:
        StellarDensity_Radius.add_process_node(
            f"apparent density radius:{er}",
            partial(Calc_Stellar_Mass_Density_Radius, eval_at_R=er),
        )
    StellarDensity_Radius.linear_mode(False)
    Structural_Parameters.add_node(StellarDensity_Radius)

    # Colours
    Colour = flow.Chart("colour")
    Colour.linear_mode(True)
    for eb1, eb2 in colours:
        Colour.add_process_node(
            f"colour in:{eb1}:{eb2}",
            partial(Calc_Colour_within, eval_in_colour1=eb1, eval_in_colour2=eb2),
        )
        Colour.add_process_node(
            f"colour at:{eb1}:{eb2}",
            partial(Calc_Colour_at, eval_in_colour1=eb1, eval_in_colour2=eb2),
        )
    Colour.linear_mode(False)
    Structural_Parameters.add_node(Colour)
    
    # Axis Ratio
    Structural_Parameters.add_process_node("axis ratio", partial(Calc_Axis_Ratio, eval_in_band = primary_band))

    # Inclination
    Structural_Parameters.add_process_node("inclination", partial(Calc_Inclination, eval_in_band = primary_band))

    # Concentration
    Concentration = flow.Chart("concentration")
    Concentration.linear_mode(True)
    for er1, er2 in concentrations:
        for eb in eval_at_bands:
            Concentration.add_process_node(
                f"concentration:{er1}:{er2}:{eb}",
                partial(
                    Calc_Concentration, eval_at_R1=er1, eval_at_R2=er2, eval_in_band=eb
                ),
            )
    Concentration.linear_mode(False)
    Structural_Parameters.add_node(Concentration)

    # Apparent Magnitude
    Structural_Parameters.add_process_node("apparent magnitude", Calc_Apparent_Magnitude)

    # Surface Density
    Structural_Parameters.add_process_node("surface density in", Calc_Surface_Density_within)
    Structural_Parameters.add_process_node("surface density at", Calc_Surface_Density_at)

    # Sersic Params
    Structural_Parameters.add_process_node("sersic params", Calc_Sersic_Params)

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
    Velocity_Fit.add_process_node('fit Tanh model', Calc_Tanh_Velocity_Fit)
    Velocity_Fit.linear_mode(False)
    Structural_Parameters.add_node(Velocity_Fit)
    Structural_Parameters.add_process_node('plot velocity', Plot_Velocity)

    # Velocity
    Velocity = flow.Chart("velocity")
    Velocity.linear_mode(True)
    Velocity.add_process_node(f"velocity:C97", partial(Calc_Velocity, eval_with_model='C97'))
    Velocity.add_process_node(f"velocity:Tan", partial(Calc_Velocity, eval_with_model='Tan'))
    Velocity.add_process_node(f"velocity:Tanh", partial(Calc_Velocity, eval_with_model='Tanh'))
    Velocity.linear_mode(False)
    Structural_Parameters.add_node(Velocity)

    # Distance Dependent Quantities
    ######################################################################

    # Physical Radius
    Structural_Parameters.add_process_node("physical radius", Calc_Physical_Radius)

    # Absolute Magnitude
    Structural_Parameters.add_process_node("absolute magnitude", Calc_Absolute_Magnitude)

    # Stellar Mass
    Structural_Parameters.add_process_node("stellar mass", Calc_Stellar_Mass)    
    
    # Dynamical Mass
    Structural_Parameters.add_process_node("dynamical mass profile", partial(Calc_Dynamical_Mass_Profile, eval_in_band = primary_band))
    Structural_Parameters.add_process_node("dynamical mass", Calc_Dynamical_Mass)

    # # Angular Momentum
    # Structural_Parameters.add_process_node("angular momentum profile", partial(Calc_Angular_Momentum_Profile, eval_in_band = primary_band))
    # Structural_Parameters.add_process_node("angular momentum", Calc_Angular_Momentum)

    # # Stellar Angular Momentum
    # Structural_Parameters.add_process_node("stellar angular momentum profile", partial(Calc_Stellar_Angular_Momentum_Profile, eval_in_band = primary_band))
    # Structural_Parameters.add_process_node("stellar angular momentum", Calc_Stellar_Angular_Momentum)
    
    # Luminosity
    Structural_Parameters.add_process_node("luminosity", Calc_Luminosity)

    # Close flowchart construction
    ######################################################################
    Structural_Parameters.linear_mode(False)

    # # Pipeline Decisions
    # ######################################################################
    # # velocity bypass
    # Structural_Parameters.add_decision_node('bypass velocity', partial(Decide_Bypass, check_key = "rotation curve"))
    # Structural_Parameters.insert_node('bypass velocity', "velocity corrections")
    # Structural_Parameters.link_nodes('bypass velocity', Velocity.forward.name)

    # # Absolute quantities bypass
    # Structural_Parameters.add_decision_node('bypass distance dependent', partial(Decide_Bypass, check_key = "distance"))
    # Structural_Parameters.insert_node('bypass distance dependent', "physical radius")
    # Structural_Parameters.link_nodes('bypass distance dependent', Structural_Parameters.nodes['luminosity'].forward.name)

    ######################################################################
    # Structural_Parameters.draw("plots/Structural_Parameters_Flowchart.png")
    return Structural_Parameters

