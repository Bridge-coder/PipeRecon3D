"""PLY reading for OpenMVS output (copied from 运动估计三维重建/scripts/evaluate.py, unchanged)."""
import numpy as np


def read_ply_vertices(path):
    """Vertex table of a binary little-endian PLY file as a numpy structured array."""
    types = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
             "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no end_header in " + path)
            lines.append(line.decode("ascii", "replace").strip())
            if lines[-1] == "end_header":
                break
        if not any("binary_little_endian" in ln for ln in lines[:3]):
            raise ValueError("only binary little-endian PLY is supported")
        n, fields, lists, in_vertex = 0, [], [], False
        for ln in lines:
            p = ln.split()
            if p[:1] == ["element"]:
                in_vertex = p[1] == "vertex"
                n = int(p[2]) if in_vertex else n
            elif p[:1] == ["property"] and in_vertex:
                if p[1] == "list":        # e.g. OpenMVS stores the views of every point: count + items
                    lists.append((np.dtype(types[p[2]]).itemsize, np.dtype(types[p[3]]).itemsize))
                elif lists:
                    raise ValueError("scalar property after a list property is not supported")
                else:
                    fields.append((p[2], "<" + types[p[1]]))
        dtype = np.dtype(fields)
        if not lists:
            return np.fromfile(f, dtype=dtype, count=n)
        # variable-length records: walk through them once to find where each one starts
        data = f.read()
        fsz = dtype.itemsize
        offs = np.empty(n, np.int64)
        pos = 0
        for i in range(n):
            offs[i] = pos
            q = pos + fsz
            for csz, isz in lists:
                cnt = data[q] if csz == 1 else int.from_bytes(data[q:q + csz], "little")
                q += csz + cnt * isz
            pos = q
        raw = np.frombuffer(data, np.uint8)
        out = np.empty(n, dtype)
        view = out.view(np.uint8).reshape(n, fsz)
        cols = np.arange(fsz)
        for a in range(0, n, 500000):
            b = min(n, a + 500000)
            view[a:b] = raw[offs[a:b, None] + cols]
        return out
