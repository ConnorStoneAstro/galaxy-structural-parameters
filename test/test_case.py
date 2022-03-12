import numpy as np
from context import Build_Structural_Parameters_Flowchart
import pandas as pd

Test_Galaxy = {'name': 'ESO479-G001',
               'distance': 68.51, # Mpc
               'distance_e': 4.8, # Mpc
               'Vcmb': 4645, # km/s
               'Vcmb E': 14, # km/s
               'Vhel': 4846.145762, # km/s
               'Vhel E': 2.997925, # km/s
               'zhel': 0.01616,
               'zhel E': 0.00001,
               'RA J2000': 36.368083, # deg
               'DEC J2000': -25.637917, # deg
               'q0': 0.05,
               'plot': {'all': True},
               'extinction': {'g': 0.056, 'r': 0.039, 'z': 0.021},
               'photometry': {'f': {}, 'n': {}, 'g': {}, 'r': {}, 'z': {}, 'w1': {}, 'w2': {}},
               'rotation curve': {},
}

T = pd.read_csv(f'ESO479-G001_rc.prof', skiprows = 1)
for k in T.keys():
    Test_Galaxy['rotation curve'][k] = np.array(T[k])

for b in Test_Galaxy['photometry']:
    T = pd.read_csv(f'ESO479-G001_{b}.prof', skiprows = 1)
    for k in T.keys():
        Test_Galaxy['photometry'][b][k] = np.array(T[k])

Structural_Parameters = Build_Structural_Parameters_Flowchart()
Structural_Parameters.draw('pipeline.png')
with open('galaxy_parameters.txt', 'w') as f:
    f.write(str(Structural_Parameters(Test_Galaxy)))
