#!/bin/bash

GRID_LIM=0.05
PARTICLES=25

MODE="phasespace"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${GRID_LIM} ${PARTICLES}
python integrator.py ${MODE}
python action_angle.py ${MODE}
python plotter.py ${MODE}

echo "Completed."