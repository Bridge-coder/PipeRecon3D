"""Extract an InPipe-SLAM ROS 2 bag (MCAP) into plain files, reading it straight out of the Zenodo zip.

The bag of sequence 172011 is 39 GB uncompressed (10 GB as zip), mostly raw images. Instead of unpacking
it, the zip member is decompressed as a stream and every message is written out once:

  OUT/bag_info.json          topics, message counts, time span, camera_info of every camera, tf_static,
                             PointCloud2 field layout
  OUT/lidar/NNNNNN.npy       one structured array per /lidar_points scan (all fields kept as recorded)
  OUT/lidar_index.csv        idx, log_time_ns, stamp_ns, frame_id, n_points
  OUT/imu_lidar.csv, imu_camera.csv, encoders.csv, cmd_vel.csv, tf.csv, trajectory.csv
  OUT/images/<cam>/NNNNNN.png + OUT/images/<cam>_index.csv   (color, usbcam, depth; infra1/2 with --infra)

Times: log_time_ns is when the recorder received the message, stamp_ns is the header stamp set by the
driver. Both are kept; which one to trust for synchronisation is decided later.

usage: python -I extract_bag.py BAG.zip|BAG.mcap OUT [--infra] [--max-seconds S]
"""
import csv
import io
import json
import os
import sys
import time
import zipfile
import zlib

import cv2
import numpy as np
from mcap.exceptions import McapError
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

args = sys.argv[1:]
infra = "--infra" in args
if infra:
    args.remove("--infra")
max_seconds = None
if "--max-seconds" in args:
    i = args.index("--max-seconds")
    max_seconds = float(args[i + 1])
    del args[i:i + 2]
SRC, OUT = args

IMAGE_TOPICS = {
    "/camera/camera/color/image_raw": "color",
    "/camera/usbcam/image_raw": "usbcam",
    "/camera/camera/depth/image_rect_raw": "depth",
}
if infra:
    IMAGE_TOPICS["/camera/camera/infra1/image_rect_raw"] = "infra1"
    IMAGE_TOPICS["/camera/camera/infra2/image_rect_raw"] = "infra2"
PC2_TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


class NoSeek(io.RawIOBase):
    """Hide seek() so that mcap reads the stream front to back instead of jumping to the summary."""

    def __init__(self, f):
        self.f = f

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, b):
        data = self.f.read(len(b))
        b[:len(data)] = data
        return len(data)


class RawInflate(io.RawIOBase):
    """Inflate a deflate stream starting at a byte offset of a file (no zip directory needed)."""

    def __init__(self, path, offset):
        self.f = open(path, "rb")
        self.f.seek(offset)
        self.z = zlib.decompressobj(-15)
        self.buf = b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.z.eof:
                return 0
            raw = self.f.read(1 << 20)
            if not raw:
                raise EOFError("compressed data ended early")
            self.buf = self.z.decompress(raw)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def open_source(path):
    if path.lower().endswith(".zip"):
        try:
            z = zipfile.ZipFile(path)
            name = next(n for n in z.namelist() if n.endswith(".mcap"))
            return io.BufferedReader(NoSeek(z.open(name)), buffer_size=1 << 22)
        except zipfile.BadZipFile:
            # zip directory (at the end of the file) not there yet, e.g. download still running:
            # find the .mcap member's local header near the start and inflate from there
            with open(path, "rb") as f:
                head = f.read(1 << 20)
            i = head.find(b"PK\x03\x04")
            while i >= 0:
                nlen, xlen = int.from_bytes(head[i + 26:i + 28], "little"), int.from_bytes(head[i + 28:i + 30], "little")
                name = head[i + 30:i + 30 + nlen].decode("utf-8", "replace")
                if name.endswith(".mcap"):
                    print(f"zip directory unreadable; inflating member {name} from offset {i}", flush=True)
                    return io.BufferedReader(RawInflate(path, i + 30 + nlen + xlen), buffer_size=1 << 22)
                i = head.find(b"PK\x03\x04", i + 4)
            raise
    return io.BufferedReader(NoSeek(open(path, "rb")), buffer_size=1 << 22)


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def image_to_array(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    if enc in ("rgb8", "bgr8"):
        a = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return a[:, :, ::-1] if enc == "rgb8" else a
    if enc in ("rgba8", "bgra8"):
        a = buf.reshape(h, step)[:, :w * 4].reshape(h, w, 4)[:, :, :3]
        return a[:, :, ::-1] if enc == "rgba8" else a
    if enc in ("mono8", "8uc1"):
        return buf.reshape(h, step)[:, :w]
    if enc in ("mono16", "16uc1"):
        dt = ">u2" if msg.is_bigendian else "<u2"
        return np.frombuffer(msg.data, dt).reshape(h, step // 2)[:, :w].astype(np.uint16)
    if enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        return cv2.cvtColor(buf.reshape(h, step)[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if enc in ("uyvy",):
        return cv2.cvtColor(buf.reshape(h, step)[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"image encoding {msg.encoding} not handled")


def pc2_to_array(msg):
    names, formats, offsets = [], [], []
    for f in msg.fields:
        dt = PC2_TYPES[f.datatype]
        if msg.is_bigendian:
            dt = dt.replace("<", ">")
        names.append(f.name)
        formats.append(dt if f.count == 1 else (dt, f.count))
        offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    n = msg.width * msg.height
    return np.frombuffer(msg.data, dtype, count=n).copy()


os.makedirs(os.path.join(OUT, "lidar"), exist_ok=True)
for cam in IMAGE_TOPICS.values():
    os.makedirs(os.path.join(OUT, "images", cam), exist_ok=True)

files, writers = {}, {}


def writer(name, header):
    if name not in writers:
        files[name] = open(os.path.join(OUT, name), "w", newline="")
        writers[name] = csv.writer(files[name])
        writers[name].writerow(header)
    return writers[name]


info = {"source": os.path.abspath(SRC), "topics": {}, "camera_info": {}, "tf_static": [], "pointcloud_fields": None}
counts, img_idx = {}, {}
lidar_i = 0
t_first = None
t_wall = time.time()
t_report = 0
reader = NonSeekingReader(open_source(SRC), decoder_factories=[DecoderFactory()])
stopped_early = None
try:
    for schema, channel, message, msg in reader.iter_decoded_messages(log_time_order=False):
        topic, t = channel.topic, message.log_time
        if t_first is None:
            t_first = t
        if max_seconds is not None and (t - t_first) / 1e9 > max_seconds:
            stopped_early = f"--max-seconds {max_seconds}"
            break
        if topic not in info["topics"]:
            info["topics"][topic] = {"type": schema.name if schema else None, "first_log_time_ns": t}
        counts[topic] = counts.get(topic, 0) + 1
        info["topics"][topic]["last_log_time_ns"] = t

        if topic == "/lidar_points":
            a = pc2_to_array(msg)
            if info["pointcloud_fields"] is None:
                info["pointcloud_fields"] = {
                    "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count}
                               for f in msg.fields],
                    "point_step": msg.point_step, "height": msg.height, "width": msg.width,
                    "is_dense": bool(msg.is_dense), "frame_id": msg.header.frame_id}
            np.save(os.path.join(OUT, "lidar", f"{lidar_i:06d}.npy"), a)
            writer("lidar_index.csv", ["idx", "log_time_ns", "stamp_ns", "frame_id", "n_points"]).writerow(
                [lidar_i, t, stamp_ns(msg.header), msg.header.frame_id, len(a)])
            lidar_i += 1
        elif topic in IMAGE_TOPICS:
            cam = IMAGE_TOPICS[topic]
            k = img_idx.get(cam, 0)
            cv2.imwrite(os.path.join(OUT, "images", cam, f"{k:06d}.png"), image_to_array(msg),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
            writer(f"images/{cam}_index.csv", ["idx", "log_time_ns", "stamp_ns", "frame_id", "encoding",
                                               "width", "height"]).writerow(
                [k, t, stamp_ns(msg.header), msg.header.frame_id, msg.encoding, msg.width, msg.height])
            img_idx[cam] = k + 1
        elif topic in ("/lidar_imu", "/camera/camera/imu"):
            name = "imu_lidar.csv" if topic == "/lidar_imu" else "imu_camera.csv"
            o, g, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            writer(name, ["log_time_ns", "stamp_ns", "frame_id", "qx", "qy", "qz", "qw",
                          "gx", "gy", "gz", "ax", "ay", "az"]).writerow(
                [t, stamp_ns(msg.header), msg.header.frame_id, o.x, o.y, o.z, o.w, g.x, g.y, g.z, a.x, a.y, a.z])
        elif topic == "/encoders":
            data = list(msg.data)
            if "encoders_layout" not in info:
                info["encoders_layout"] = {"dim": [{"label": d.label, "size": d.size, "stride": d.stride}
                                                   for d in msg.layout.dim],
                                           "data_offset": msg.layout.data_offset, "n_values": len(data)}
            writer("encoders.csv", ["log_time_ns"] + [f"v{j}" for j in range(len(data))]).writerow([t] + data)
        elif topic == "/cmd_vel":
            writer("cmd_vel.csv", ["log_time_ns", "vx", "vy", "vz", "wx", "wy", "wz"]).writerow(
                [t, msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z])
        elif topic in ("/tf", "/tf_static"):
            for tr in msg.transforms:
                p, q = tr.transform.translation, tr.transform.rotation
                row = [t, stamp_ns(tr.header), tr.header.frame_id, tr.child_frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                if topic == "/tf_static":
                    info["tf_static"].append(dict(zip(["log_time_ns", "stamp_ns", "parent", "child", "x", "y", "z",
                                                       "qx", "qy", "qz", "qw"], row)))
                else:
                    writer("tf.csv", ["log_time_ns", "stamp_ns", "parent", "child", "x", "y", "z",
                                      "qx", "qy", "qz", "qw"]).writerow(row)
        elif topic == "/trajectory":
            # Path messages grow over time; keep the newest pose of each message (a time series)
            if msg.poses:
                ps = msg.poses[-1]
                p, q = ps.pose.position, ps.pose.orientation
                writer("trajectory.csv", ["log_time_ns", "stamp_ns", "frame_id", "n_poses", "x", "y", "z",
                                          "qx", "qy", "qz", "qw"]).writerow(
                    [t, stamp_ns(ps.header), msg.header.frame_id, len(msg.poses), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        elif topic.endswith("/camera_info"):
            cam = topic.split("/")[-2]
            if cam not in info["camera_info"]:
                info["camera_info"][cam] = {
                    "topic": topic, "frame_id": msg.header.frame_id, "width": msg.width, "height": msg.height,
                    "distortion_model": msg.distortion_model, "d": list(msg.d), "k": list(msg.k),
                    "r": list(msg.r), "p": list(msg.p)}

        if (t - t_first) / 1e9 >= t_report:
            print(f"bag time {(t - t_first) / 1e9:6.1f} s, wall {time.time() - t_wall:6.0f} s, "
                  f"lidar {lidar_i}, images {img_idx}", flush=True)
            t_report += 10
except (zlib.error, EOFError, zipfile.BadZipFile, McapError) as e:
    stopped_early = f"{type(e).__name__}: {e}"

for f in files.values():
    f.close()
for topic, c in counts.items():
    info["topics"][topic]["count"] = c
info["stopped_early"] = stopped_early
with open(os.path.join(OUT, "bag_info.json"), "w") as f:
    json.dump(info, f, indent=1)
print(f"done in {time.time() - t_wall:.0f} s; lidar scans {lidar_i}, images {img_idx}; stopped early: {stopped_early}")
