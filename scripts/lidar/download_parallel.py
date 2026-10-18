"""Download one large file with many parallel HTTP range requests, resumable, then check its MD5.

Zenodo limits every connection to roughly 30-60 KB/s (measured 2026-10-14), while the line here
manages about 3 MB/s, so the file is fetched as fixed-size chunks by many connections at once.
Each chunk is written straight to its place in the output file; finished chunk numbers are appended
to <out>.done, so a rerun with the same arguments only fetches the chunks that are still missing.

usage: python -I download_parallel.py URL SIZE OUT MD5 [--workers 24] [--chunk-mb 32]
"""
import hashlib
import os
import queue
import sys
import threading
import time
import urllib.request

args = sys.argv[1:]
opts = {"--workers": 24, "--chunk-mb": 32}
for k in list(opts):
    if k in args:
        i = args.index(k)
        opts[k] = int(args[i + 1])
        del args[i:i + 2]
URL, SIZE, OUT, MD5 = args[0], int(args[1]), args[2], args[3].lower()
CHUNK = opts["--chunk-mb"] << 20
N = (SIZE + CHUNK - 1) // CHUNK
DONE = OUT + ".done"

lock = threading.Lock()
# chunk numbers in <out>.done only mean something for one chunk size: keep it fixed across resumes
if os.path.exists(OUT + ".chunk"):
    with open(OUT + ".chunk") as f:
        if int(f.read()) != CHUNK:
            sys.exit(f"{OUT} was started with a different --chunk-mb; use the same value to resume")
else:
    with open(OUT + ".chunk", "w") as f:
        f.write(str(CHUNK))
done = set()
if os.path.exists(DONE):
    with open(DONE) as f:
        done = {int(x) for x in f.read().split()}
if not os.path.exists(OUT):
    with open(OUT, "wb") as f:
        f.truncate(SIZE)
elif os.path.getsize(OUT) != SIZE:
    sys.exit(f"{OUT} exists with size {os.path.getsize(OUT)}, expected {SIZE}")

todo = queue.Queue()
for i in range(N):
    if i not in done:
        todo.put(i)
got = [0]  # bytes received in this session


def fetch(i):
    # Zenodo also closes each connection after a few minutes (~8 MB at its per-connection rate), so a
    # broken request continues from the last byte written instead of restarting the chunk.
    start, end = i * CHUNK, min(SIZE, (i + 1) * CHUNK) - 1
    pos = start
    failures = 0
    while pos <= end:
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(OUT, "r+b") as f:
                if r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {pos}-{end}/"):
                    raise IOError(f"unexpected response {r.status} {r.headers.get('Content-Range')}")
                f.seek(pos)
                while True:
                    buf = r.read(1 << 16)
                    if not buf:
                        break
                    f.write(buf)
                    pos += len(buf)
                    with lock:
                        got[0] += len(buf)
            failures = 0
        except Exception as e:  # network errors, 429/5xx: wait, then continue from pos
            failures += 1
            if failures > 50:
                raise RuntimeError(f"chunk {i} failed 50 times in a row") from e
            wait = min(300, 10 * failures)
            print(f"chunk {i}: {type(e).__name__}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    with lock, open(DONE, "a") as f:
        f.write(f"{i}\n")
        done.add(i)


def worker():
    while True:
        try:
            i = todo.get_nowait()
        except queue.Empty:
            return
        fetch(i)


print(f"{N} chunks of {CHUNK >> 20} MB, {len(done)} already done, {opts['--workers']} connections", flush=True)
threads = [threading.Thread(target=worker, daemon=True) for _ in range(opts["--workers"])]
for t in threads:
    t.start()
t0 = time.time()
last = 0.0
while any(t.is_alive() for t in threads):
    time.sleep(1)
    if time.time() - last >= 300:
        last = time.time()
        el = time.time() - t0
        with lock:
            have = sum(min(SIZE, (i + 1) * CHUNK) - i * CHUNK for i in done)
            rate = got[0] / max(el, 1)
        eta = (SIZE - have) / rate / 60 if rate > 0 else float("nan")
        print(f"{time.strftime('%H:%M')}  {have / 1e9:.2f}/{SIZE / 1e9:.2f} GB in finished chunks, "
              f"{rate / 1e6:.2f} MB/s, ETA {eta:.0f} min", flush=True)

if len(done) != N:
    sys.exit(f"only {len(done)} of {N} chunks done; run again to resume")
print("all chunks done, checking MD5 ...", flush=True)
h = hashlib.md5()
with open(OUT, "rb") as f:
    for buf in iter(lambda: f.read(1 << 24), b""):
        h.update(buf)
print(f"md5 {h.hexdigest()} expected {MD5}: {'DOWNLOAD_OK' if h.hexdigest() == MD5 else 'MD5_MISMATCH'}", flush=True)
