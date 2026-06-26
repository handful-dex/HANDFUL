"""
export_tcp_data.py
──────────────────
Exports block-in-TCP-frame positions to JSON for the web visualizer.
Also pre-computes KDE grids (XY, XZ, YZ) so the browser doesn't have to.

Usage:
    python export_tcp_data.py \
        --data  intermediate_states/xArm7-v1-pick-all__train__9__1771746645/grasp_to_push_img.pt \
        --out   static/data/tcp_positions.json
"""

import argparse
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
import os


FOLDERS = [
    "intermediate_states/fingers_0_1_2_3_active_1_palm_True_seed_10",
    "intermediate_states/fingers_0_1_2_3_active_2_palm_True_seed_9",
    "intermediate_states/fingers_0_2_1_3_active_2_palm_False_seed_9",
    "intermediate_states/fingers_1_0_2_3_active_1_palm_True_seed_9",
    "intermediate_states/fingers_1_2_0_3_active_2_palm_True_seed_9",
    "intermediate_states/fingers_2_0_1_3_active_1_palm_True_seed_9",
    "intermediate_states/fingers_3_0_1_2_active_2_palm_False_seed_9",
    "intermediate_states/fingers_3_1_0_2_active_2_palm_False_seed_9",
    "intermediate_states/fingers_3_2_0_1_active_2_palm_False_seed_9",
    "intermediate_states/xArm7-v1-pick-all__train__9__1771746645",

]

DATA_FILE = "grasp_to_push.pt"
OUT_FILE  = "tcp_positions.json"

def transform_to_tcp_frame(block_pose, tcp_pose):
    block_pos = block_pose[:3]
    tcp_pos   = tcp_pose[:3]
    tcp_quat  = tcp_pose[3:]          # [w, x, y, z]
    tcp_rot   = R.from_quat([tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]])
    return tcp_rot.inv().apply(block_pos - tcp_pos)


def palm_bbox():
    X_min, X_max = -0.0375, 0.0425
    Y_min, Y_max = -0.0655, 0.0595
    Z_min        = -0.12
    Z_max        = Z_min + 0.046
    corners = [
        [X_max, Y_max, Z_min], [X_min, Y_max, Z_min],
        [X_min, Y_min, Z_min], [X_max, Y_min, Z_min],
        [X_max, Y_max, Z_max], [X_min, Y_max, Z_max],
        [X_min, Y_min, Z_max], [X_max, Y_min, Z_max],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return corners, edges


def finger_boxes():
    X_min     = -0.0395
    Z_mid     = -0.12 + 0.030
    Z_h       = 0.014
    flen      = 0.15              # updated
    fhw       = 0.0145            # updated
    defs = [
        ('Index',  0.043,    '#4A90E2'),
        ('Middle', -0.00245, '#4A90E2'),
        ('Ring',   -0.0479,  '#4A90E2'),
    ]
    out = []
    for name, yc, color in defs:
        x0, x1 = X_min - flen, X_min
        y0, y1 = yc - fhw, yc + fhw
        z0, z1 = Z_mid - Z_h, Z_mid + Z_h
        out.append({'name': name, 'color': color,
                    'x': [x0, x1], 'y': [y0, y1], 'z': [z0, z1]})
    # Thumb — updated x range and y extent
    Y_max = 0.0615
    out.append({'name': 'Thumb', 'color': '#4A90E2',
                'x': [0.0135, 0.0425],
                'y': [Y_max, Y_max + 0.110],
                'z': [Z_mid - Z_h, Z_mid + Z_h]})

    return out


# ── KDE grid helper ───────────────────────────────────────────────────────────

def kde_grid(a, b, n=120, pad=0.30):
    """Returns (xi, yi, zz) as plain lists for JSON serialisation."""
    pa = (a.max() - a.min()) * pad
    pb = (b.max() - b.min()) * pad
    xi = np.linspace(a.min() - pa, a.max() + pa, n)
    yi = np.linspace(b.min() - pb, b.max() + pb, n)
    xx, yy = np.meshgrid(xi, yi)
    kernel = gaussian_kde(np.vstack([a, b]))
    zz = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(n, n)
    return xi.tolist(), yi.tolist(), zz.tolist()


def run_one(data_path, out_path):
    all_episodes = torch.load(data_path)

    # Group by finger config
    has_fi = any('active_finger_indices' in ep[-1] for ep in all_episodes)
    groups = {}
    if has_fi:
        for ep in all_episodes:
            fi  = ep[-1].get('active_finger_indices', None)
            key = tuple(sorted(fi.cpu().tolist() if isinstance(fi, torch.Tensor) else fi)) \
                  if fi is not None else ('unknown',)
            groups.setdefault(key, []).append(ep)
    else:
        groups = {('all',): all_episodes}

    payload = {'groups': [], 'palm': {}, 'fingers': []}

    palm_c, palm_e = palm_bbox()
    payload['palm'] = {'corners': palm_c, 'edges': palm_e}
    payload['fingers'] = finger_boxes()

    for key, eps in sorted(groups.items()):
        pts = []
        for ep in eps:
            s = ep[-1]
            if 'cube' in s and 'tcp_pose' in s:
                cw = s['cube'].cpu().numpy()     if isinstance(s['cube'],     torch.Tensor) else np.array(s['cube'])
                tw = s['tcp_pose'].cpu().numpy() if isinstance(s['tcp_pose'], torch.Tensor) else np.array(s['tcp_pose'])
                pts.append(transform_to_tcp_frame(cw, tw).tolist())
        if not pts:
            continue

        pts_np = np.array(pts)
        label  = f"Fingers {list(key)}" if key != ('all',) else "All Grasps"

        xi_xy, yi_xy, zz_xy = kde_grid(pts_np[:,0], pts_np[:,1])
        xi_xz, yi_xz, zz_xz = kde_grid(pts_np[:,0], pts_np[:,2])
        xi_yz, yi_yz, zz_yz = kde_grid(pts_np[:,1], pts_np[:,2])

        payload['groups'].append({
            'label': label,
            'n':     len(pts),
            'pts':   pts,
            'kde': {
                'xy': {'xi': xi_xy, 'yi': yi_xy, 'z': zz_xy},
                'xz': {'xi': xi_xz, 'yi': yi_xz, 'z': zz_xz},
                'yz': {'xi': xi_yz, 'yi': yi_yz, 'z': zz_yz},
            }
        })
        print(f"Exported '{label}' — {len(pts)} points")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    print(f"\nSaved → {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--out',  default=None)
    args = parser.parse_args()

    if args.data and args.out:
        run_one(args.data, args.out)
    else:
        for folder in FOLDERS:
            print(f"\n── {folder}")
            out_folder = os.path.relpath(folder, "intermediate_states")
            out_folder = os.path.join("intermediate_states_tcp", out_folder)
            run_one(
                os.path.join(folder, DATA_FILE),
                os.path.join(out_folder, OUT_FILE),
            )


if __name__ == '__main__':
    main()