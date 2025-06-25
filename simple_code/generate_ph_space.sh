#!/bin/bash

RADIUS=0.1
PARTICLES_PER_JOB=50

POINCARE_MODE=all

# -------------------------

export PHASE_SPACE=1
python generate_init_cond.py ${RADIUS} ${PARTICLES_PER_JOB} phase
python integrator.py ${POINCARE_MODE}
python action_angle.py ${POINCARE_MODE}
#python plotter.py
#python count.py ${POINCARE_MODE}


echo "Phase space generation complete."