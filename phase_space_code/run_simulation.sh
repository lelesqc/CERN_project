#!/bin/bash

GRID_LIM=12.9
PARTICLES=50

MODE="tune"  # Options: "tune", "phasespace"

DATA_FILE="../code/integrator/evolved_qp_last.npz"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${GRID_LIM} ${PARTICLES}
python integrator.py ${MODE}
python action_angle.py ${MODE}

if [ "$MODE" = "tune" ]; then
    python tune.py
fi

python plotter.py ${MODE}

echo "Completed."