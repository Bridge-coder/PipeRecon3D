# docs/11: the rest of this round after 3DGS-MCMC diverged on concrete_2 with gsplat's defaults (its remaining runs dropped)
$q = "/mnt/" + $PSScriptRoot.Substring(0, 1).ToLower() + ($PSScriptRoot.Substring(2) -replace '\\', '/') + "/queue.sh"   # this folder as WSL sees it
wsl -d pipe3d -u root -- bash $q `
  "inpipe_vision run_3dgs --baseline 3dgs" `
  "inpipe_vision run_2dgs --baseline 2dgs" `
  "concrete_2 run_g0a_opaque --anchor --opaque" `
  "sewer_pipe1 run_3dgs --baseline 3dgs" `
  "sewer_pipe1 run_2dgs --baseline 2dgs" `
  "concrete_lab run_3dgs --baseline 3dgs --points-every 2" `
  "concrete_lab run_2dgs --baseline 2dgs --points-every 2" `
  "concrete_2 run_g0a_pose --anchor --pose-opt"
