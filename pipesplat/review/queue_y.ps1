# docs/11: the training runs of this round, one after another (queue.sh skips a run that already has summary_r2.json)
$q = "/mnt/" + $PSScriptRoot.Substring(0, 1).ToLower() + ($PSScriptRoot.Substring(2) -replace '\\', '/') + "/queue.sh"   # this folder as WSL sees it
wsl -d pipe3d -u root -- bash $q `
  "sewer_pipe1 run_vhf --anchor --surface surface_vhf.npz" `
  "sewer_pipe1 run_g0a --anchor" `
  "concrete_2 run_3dgs --baseline 3dgs" `
  "concrete_2 run_mcmc --baseline mcmc" `
  "concrete_2 run_2dgs --baseline 2dgs" `
  "inpipe_vision run_3dgs --baseline 3dgs" `
  "inpipe_vision run_mcmc --baseline mcmc" `
  "inpipe_vision run_2dgs --baseline 2dgs" `
  "sewer_pipe1 run_3dgs --baseline 3dgs" `
  "sewer_pipe1 run_mcmc --baseline mcmc" `
  "sewer_pipe1 run_2dgs --baseline 2dgs" `
  "concrete_lab run_3dgs --baseline 3dgs --points-every 2" `
  "concrete_lab run_mcmc --baseline mcmc --points-every 2" `
  "concrete_lab run_2dgs --baseline 2dgs --points-every 2" `
  "concrete_2 run_g0a_opaque --anchor --opaque" `
  "concrete_2 run_g0a_pose --anchor --pose-opt"
