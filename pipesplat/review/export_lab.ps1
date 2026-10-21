# docs/10: independent check on the lab concrete pipe A (not used for any tuning): export scene + surface (Windows)
Set-Location C:\Users\PC\Desktop\PipeRecon3D
python -I pipesplat/export_g0.py vision D:/pipe3d_new/concrete_lab "D:/pipe3d_new/mix15_3lm/data/300mm_mix_15meters_ON_3lm_3u_CLOSED_1_stop_at_14meters.mp4" D:/pipe3d_gs/g0/concrete_lab --lamp lampmap/results/concrete_lab/lamp_map.npy
python -I pipesplat/export_surface.py vision D:/pipe3d_new/concrete_lab D:/pipe3d_gs/g0/concrete_lab
