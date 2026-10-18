"""Common time base for all streams of an extracted InPipe-SLAM bag (written by f1_time.py, used by F2-F5).

Common time = the RealSense clock, which the realsense driver already keeps on host (epoch) time.
time_offsets.json holds, per stream, how to turn its rows into common time (ns):
  {"kind": "stamp", "offset_ns": a, "drift": b, "t0_ns": t0}  ->  t = stamp + a + b * (stamp - t0)
  {"kind": "log",   "lag_ns": c}                                ->  t = log_time - c
"""
import csv
import json
import os

import numpy as np


def load_offsets(path):
    with open(path) as f:
        return json.load(f)


def to_common(rule, stamp_ns=None, log_ns=None):
    if rule["kind"] == "stamp":
        s = np.asarray(stamp_ns, np.int64)
        return s + np.int64(round(rule["offset_ns"])) + np.round(rule["drift"] * (s - rule["t0_ns"])).astype(np.int64)
    return np.asarray(log_ns, np.int64) - np.int64(round(rule["lag_ns"]))


def stream_times(ex_dir, offsets, stream):
    """Common times of every row of a stream's CSV (lidar, imu_lidar, imu_camera, color, usbcam, depth, encoders)."""
    files = {"lidar": "lidar_index.csv", "imu_lidar": "imu_lidar.csv", "imu_camera": "imu_camera.csv",
             "color": "images/color_index.csv", "usbcam": "images/usbcam_index.csv",
             "depth": "images/depth_index.csv", "encoders": "encoders.csv"}
    with open(os.path.join(ex_dir, files[stream])) as f:
        rows = list(csv.DictReader(f))
    log = np.array([int(r["log_time_ns"]) for r in rows], np.int64)
    stamp = np.array([int(r["stamp_ns"]) for r in rows], np.int64) if "stamp_ns" in rows[0] else None
    return to_common(offsets[stream], stamp, log), rows
