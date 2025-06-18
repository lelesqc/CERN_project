#!/bin/bash

TOTAL_JOBS=1
RADIUS=10
PARTICLES_PER_JOB=2000

# -------------------------

echo "Starting batch generation of ${TOTAL_JOBS} initial condition files..."

for (( i=1; i<=${TOTAL_JOBS}; i++ ))
do
  echo "--- Running job with SEED=${i} ---" 
  python generate_init_cond.py ${RADIUS} ${PARTICLES_PER_JOB} ${i}
  python integrator.py
  python action_angle.py
  python plotter.py

  echo "Job with SEED=${i} finished."
  echo ""
done

echo "Batch generation complete."