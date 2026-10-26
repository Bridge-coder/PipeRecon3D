#!/bin/bash
# docs/13 Z1: the lamp model + coverage term (--alpha-w) on the other data sets, one after another (queue.sh skips a
# run that already has summary_r2.json). Each run = that data set's best model with only these options added.
bash "$(dirname "${BASH_SOURCE[0]}")/queue.sh" \
  "concrete_lab run_g0al_a01 --anchor --points-every 2 --lamp --lamp-smooth-mm 20 --alpha-w 0.1" \
  "sewer_pipe1 run_vhf_l_a01 --anchor --surface surface_vhf.npz --lamp --lamp-smooth-mm 20 --alpha-w 0.1" \
  "inpipe_vision run_vhf_a01 --anchor --surface surface_vhf.npz --lamp --lamp-ambient --lamp-smooth-mm 20 --alpha-w 0.1"
