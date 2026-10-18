"""Write a known radial distortion into the image list made by openMVG_main_SfMInit_ImageListing.

Usage: set_distortion.py SFM_DATA_JSON K1

The listing program takes the camera matrix on its command line (-k) but always starts the distortion at
zero. The list must have been made with camera model 2 ("pinhole radial 1"): x_d = x_u * (1 + K1 * r_u^2)
in normalised image coordinates, negative K1 = barrel distortion.
"""
import json
import sys

path, k1 = sys.argv[1], float(sys.argv[2])
sfm = json.load(open(path, encoding="utf-8"))
for entry in sfm["intrinsics"]:
    data = entry["value"]["ptr_wrapper"]["data"]
    if "disto_k1" not in data:
        sys.exit(f"{path}: camera model {entry['value'].get('polymorphic_name')} has no disto_k1; list the images with -c 2")
    print(f"intrinsic {entry['key']}: focal {data['focal_length']:g}, principal point {data['principal_point']}, "
          f"k1 {data['disto_k1'][0]:g} -> {k1:g}")
    data["disto_k1"] = [k1]
json.dump(sfm, open(path, "w", encoding="utf-8"), indent=1)
