#!/usr/bin/env python
"""Log machine load every N seconds while a run goes: CPU % (whole machine), cores busy > 50 %, busiest process,
RAM used, GPU util / memory / temperature / power. One CSV line per sample.
Usage: sample_load.py OUT.csv [--every 30]"""
import argparse
import os
import subprocess
import time
from datetime import datetime


def cpu_times():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu"):
            f = line.split()
            v = list(map(int, f[1:]))
            out[f[0]] = (sum(v), v[3] + v[4])  # total, idle + iowait
    return out


def top_proc():
    try:
        r = subprocess.run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu", "--no-headers"], capture_output=True, text=True)
        p = r.stdout.splitlines()[0].split(None, 1)
        return p[1].strip(), float(p[0])
    except Exception:
        return "", 0.0


def gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return [s.strip() for s in r.stdout.strip().split(",")]
    except Exception:
        return ["", "", "", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--every", type=float, default=30)
    a = ap.parse_args()
    new = not os.path.exists(a.out)
    f = open(a.out, "a", buffering=1)
    if new:
        f.write("time,cpu_pct,cores_busy,cores,load1,top_proc,top_proc_pct,ram_used_gb,gpu_util,gpu_mem_mb,gpu_temp_c,gpu_power_w\n")
    prev = cpu_times()
    while True:
        time.sleep(a.every)
        cur = cpu_times()
        def busy(k):
            dt = cur[k][0] - prev[k][0]
            return 100 * (1 - (cur[k][1] - prev[k][1]) / dt) if dt else 0.0
        cores = [k for k in cur if k != "cpu"]
        nbusy = sum(busy(k) > 50 for k in cores)
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        ram = (mem["MemTotal"] - mem["MemAvailable"]) / 2 ** 20
        name, pct = top_proc()
        f.write(",".join(map(str, [datetime.now().strftime("%H:%M:%S"), round(busy("cpu"), 1), nbusy, len(cores),
                                   os.getloadavg()[0], name, pct, round(ram, 2), *gpu()])) + "\n")
        prev = cur


if __name__ == "__main__":
    main()
