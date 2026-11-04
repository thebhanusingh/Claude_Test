"""Turn web_orient.py output into splat-transform actions.
splat-transform reads .ply into PlayCanvas engine space E = diag(-1,-1,1) * P (180 deg about Z), and its
-r x,y,z equals scipy Rotation.from_euler('xyz', [x,y,z], degrees=True) applied in E (checked on a 4-point test PLY).
Produces: -t <-c_E> -r <euler> -B <box in final frame>, so the scene's up ends on +Y with its centre at the origin.
Usage: web_st_args.py IN.ply ORIENT.json [--pct 1 99] [--pad 0.1]"""
import argparse, json, numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation as Rot
ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("orient")
ap.add_argument("--pct", type=float, nargs=2, default=[1, 99]); ap.add_argument("--pad", type=float, default=0.1)
a = ap.parse_args(); o = json.load(open(a.orient)); F = np.diag([-1.0, -1.0, 1.0])
u = F @ np.array(o["up"]); c = F @ np.array(o["centre"])
y = np.array([0, 1.0, 0]); vx = np.cross(u, y); s = np.linalg.norm(vx); co = u @ y
K = np.array([[0, -vx[2], vx[1]], [vx[2], 0, -vx[0]], [-vx[1], vx[0], 0]])
R = np.eye(3) + K + K @ K * ((1 - co) / (s * s))
eul = Rot.from_matrix(R).as_euler("xyz", degrees=True)
v = PlyData.read(a.ply)["vertex"]; xyz = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
Q = ((xyz @ F.T) - c) @ R.T; Qo = Q[op > 0.5]
lo, hi = np.percentile(Qo, a.pct[0], 0), np.percentile(Qo, a.pct[1], 0); pad = a.pad * (hi - lo)
box = np.concatenate([lo - pad, hi + pad]); inside = ((Q >= box[:3]) & (Q <= box[3:])).all(1).sum()
f = lambda arr: ",".join(f"{x:.4f}" for x in arr)
print(json.dumps(dict(args=f"-t {f(-c)} -r {f(eul)} -B {f(box)}", up_after=(R @ u).round(4).tolist(),
                      box=box.round(3).tolist(), inside=int(inside), total=len(Q))))
