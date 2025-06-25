import numpy as np
import functions as fn
import os
import sys
from tqdm.auto import tqdm
if os.environ.get("PHASE_SPACE", "0") == "1":
    import params_fixed as par
else:
    import params as par

is_phase_space = os.environ.get("PHASE_SPACE", "0") == "1"

def run_integrator():
    """
    carico i dati iniziali

    definisco le variabili

    while finché non trovo l'ultima sezione

    prima cosa da fare: controllo sezione di poincare

    poi aggiornamento q e p

    poi aumento il tempo 

    """

    