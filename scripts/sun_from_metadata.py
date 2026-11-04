#!/usr/bin/env python
"""True sun position for a phone video from its own metadata (GPS + capture time), NOAA solar-position formulas.

Reads com.apple.quicktime.location.ISO6709 and com.apple.quicktime.creationdate (capture start, local time with
offset; creation_time is the later file-write time and is not used) and the duration with ffprobe, and gives the
sun's elevation (with refraction) and compass azimuth (deg clockwise from true north) at the start, middle and end.
The video has no compass heading, so only the elevation can be used in the scene frame directly.

Usage: sun_from_metadata.py VIDEO --out sun_metadata.json
"""
import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


def solar_position(t, lat, lon):
    """t: aware datetime; lat/lon in degrees (east positive). Returns (elevation_deg incl. refraction, azimuth_deg)."""
    u = t.astimezone(timezone.utc)
    jd = u.timestamp() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    L = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + math.sin(2 * Mr) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * Mr) * 0.000289)
    om = math.radians(125.04 - 1934.136 * jc)
    lam = math.radians(L + C - 0.00569 - 0.00478 * math.sin(om))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(om))
    decl = math.asin(math.sin(eps) * math.sin(lam))
    y = math.tan(eps / 2) ** 2
    Lr = math.radians(L)
    eqt = 4 * math.degrees(y * math.sin(2 * Lr) - 2 * e * math.sin(Mr) + 4 * e * y * math.sin(Mr) * math.cos(2 * Lr)
                           - 0.5 * y * y * math.sin(4 * Lr) - 1.25 * e * e * math.sin(2 * Mr))
    minutes = u.hour * 60 + u.minute + u.second / 60 + u.microsecond / 6e7
    tst = (minutes + eqt + 4 * lon) % 1440
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = max(-1.0, min(1.0, math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)))
    zen = math.acos(cz)
    el = 90 - math.degrees(zen)
    if el > 85:
        refr = 0.0
    elif el > 5:
        te = math.tan(math.radians(el))
        refr = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
    elif el > -0.575:
        refr = 1735 + el * (-518.2 + el * (103.4 + el * (-12.79 + el * 0.711)))
    else:
        refr = -20.772 / math.tan(math.radians(el))
    el += refr / 3600
    ca = max(-1.0, min(1.0, (math.sin(la) * cz - math.sin(decl)) / (math.cos(la) * math.sin(zen))))
    az = math.degrees(math.acos(ca))
    az = (az + 180) % 360 if ha > 0 else (540 - az) % 360
    return el, az


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(a.video)],
                                      check=True, capture_output=True, text=True).stdout)["format"]
    tags = probe.get("tags", {})
    loc = tags.get("com.apple.quicktime.location.ISO6709")
    when = tags.get("com.apple.quicktime.creationdate")
    if not loc or not when:
        raise SystemExit("ERROR: no com.apple.quicktime GPS / creationdate tags in this video")
    lat, lon = (float(v) for v in re.match(r"([+-][\d.]+)([+-][\d.]+)", loc).groups())
    t0 = datetime.strptime(when, "%Y-%m-%dT%H:%M:%S%z")
    dur = float(probe["duration"])
    res = dict(video=str(a.video), lat=lat, lon=lon, accuracy_m=float(tags.get("com.apple.quicktime.location.accuracy.horizontal", "nan")),
               capture_start=t0.isoformat(), duration_s=round(dur, 2), heading="not in metadata")
    for name, dt in (("start", 0.0), ("mid", dur / 2), ("end", dur)):
        el, az = solar_position(t0 + timedelta(seconds=dt), lat, lon)
        res[name] = dict(elevation_deg=round(el, 3), azimuth_deg=round(az, 3))
    res["elevation_deg"] = res["mid"]["elevation_deg"]
    res["azimuth_deg"] = res["mid"]["azimuth_deg"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"GPS {lat:+.4f} {lon:+.4f}, capture {t0.isoformat()} for {dur:.1f} s: sun elevation "
          f"{res['start']['elevation_deg']:.2f} -> {res['end']['elevation_deg']:.2f} deg, azimuth "
          f"{res['start']['azimuth_deg']:.2f} -> {res['end']['azimuth_deg']:.2f} deg; mid {res['elevation_deg']:.2f} / "
          f"{res['azimuth_deg']:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
