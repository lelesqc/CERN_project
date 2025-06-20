#!/bin/bash

RADIUS=0.05
PARTICLES_PER_JOB=25

# -------------------------

export PHASE_SPACE=1
python generate_init_cond.py ${RADIUS} ${PARTICLES_PER_JOB} phase
python integrator.py all
python action_angle.py
python plotter.py

echo "Phase space generation complete."