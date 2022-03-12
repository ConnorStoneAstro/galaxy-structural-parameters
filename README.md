# Structural Parameters Pipeline

This code was used to generate the structural parameters tables in
Stone et al. 2022. To facilitate widespread use of the PROBES
compendium we have released the code so anyone can easily compute
their own structural parameters in a manner consistent with the other
parameters. One can also incorporate this code into their own analysis
routines to run Bayesian analysis and propogate uncertainties through
the pipeline. Further readme's and documentation describe specific
parts of the structural parameters pipeline, here we will go over
basic usage.

The pipeline code is very modular and so can flexibly be incorporated
in anyone's code. At the base level are individual structual parameter
functions which take in a dictionary of data about a galaxy and update
the dictionary with their particular structural parameter. These
functions can of course the extracted and used individually, however
the code is of course more powerful if used for the full pipeline
capabilities. Included with the code is a copy of the flowchart
generating package "flow" which is used to construct analysis
pipelines. A pipeline builder is included which will use the flow code
to construct a flowchart to compute the desired quantities. Example
code is given below for interacting with the pipeline builder.

```python
from structural_parameters import Build_Structural_Parameters_Flowchart, flow

Galaxies = read_galaxies(path_to_datatables, has = ['g', 'r', 'z', 'w1', 'w2'])

print('prepping galaxies')
for g in Galaxies:
    Galaxies[g]['plot'] = {'all':True, 'saveto': './'}
    Galaxies[g]['q0'] = 0.1

Calc_Structural_Parameters = Build_Structural_Parameters_Flowchart()

Calc_Structural_Parameters.draw("pipeline.png")

params_for_one_galaxy = Calc_Structural_Parameters(Galaxies['ESO479-G001'])

PIPE_params = flow.Pipe("parallel structural parameters", Calc_Structural_Parameters, safe_mode = True)
AllParams = PIPE_params(Galaxies.values())
```

So, what is happening in the above code? It's as easy as
ABCDEF. First, we import the pipeline builder and the flowchart
code. Second, we read in the galaxies, this "read_galaxies" function
is included in with the PROBES compendium data tables and simply reads
in all the profiles and galaxy data into dictionaries. Third, We run
the pipeline builder with no arguments to get the full pipeline for
all parameters, this is a very large pipeline and will compute
hundreds of parameters per galaxy, to run faster one can select
specific functions that they want (just beware that some structural
parameters depend on previous ones). Fourth, we draw the pipeline,
just to get a visualization of what's going to happen. Fifth, we try
out the pipeline on one galaxy, if all you need is one galaxy then
this is as far as you need to go. Yes, the pipeline can be run just
like a function with an input of a galaxy dictionary. Sixth, if we
want to compute parameters for all the galaxies then we wrap it in a
flow Pipe object. The pipe is just a wrapper which will take a list of
galaxy dictionaries and pass them in parallel to the
"Calc_Structural_Parameters" flowchart. The "AllParams" result is what
we get after running all the galaxies through the analysis pipeline,
it should be a list of dictionaries with lots of structural parameters
in them! Note that if the code crashes for a single galaxy, it will
simply output "None" in that spot of the list.

If you wish to take advantage of the inclination corrections as used
in the paper, then you can compute them and run the analysis pipeline
as follows.

```python
from structural_parameters import Build_Structural_Parameters_Flowchart, flow
from structural_parameters import Build_Inclination_Correction_Prep, Fit_Inclination_Correction

Galaxies = read_galaxies(path_to_datatables, has = ['g', 'r', 'z', 'w1', 'w2'])

print('prepping galaxies')
for g in Galaxies:
    Galaxies[g]['plot'] = {'all':True, 'saveto': './'}
    Galaxies[g]['q0'] = 0.1

Prep_For_Inclination_Corrections = Build_Inclination_Correction_Prep(eval_after_R = 'Rp50', eval_after_band = 'r')
PIPE_prep = flow.Pipe("prep for inclination corrections", Prep_For_Inclination_Corrections, safe_mode = True)
Prepped_galaxies = PIPE_prep(Galaxies.values())
specifications = Fit_Inclination_Correction(Prepped_galaxies, eval_after_R = 'Rp50', eval_after_band = 'r')

Calc_Structural_Parameters = Build_Structural_Parameters_Flowchart(incl_corr_specification = specifications)

Calc_Structural_Parameters.draw("pipeline.png")

params_for_one_galaxy = Calc_Structural_Parameters(Galaxies['ESO479-G001'])

PIPE_params = flow.Pipe("parallel structural parameters", Calc_Structural_Parameters, safe_mode = True)
AllParams = PIPE_params(Galaxies.values())
```

This is more or less the same as before, except in the middle we
compute the parameters for the inclination corrections. Some
structural parameters are needed before we can compute the inclination
correction, instead of running the whole analysis pipeline we run a
smaller version which only computes what is needed. Then we pass the
"Prepped_galaxies" (which is just the galaxies plus a few structural
parameters) to the fitting function. The inclination correction
fitting is more or less a least squares fit with a few cuts.