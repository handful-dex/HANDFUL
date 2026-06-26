import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    palette="pastel",
    rc={
        "font.family": "serif",
        "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua", "Georgia"],
    },
)
BASE_FS  = 18
TITLE_FS = BASE_FS + 4
LABEL_FS = BASE_FS
TICK_FS  = BASE_FS - 4
ANNOT_FS = BASE_FS - 6

C_PALM    = "#E35E52"
C_MCP     = "#4A90E2"
C_ORIGIN  = "#E35E52"
CMAP_HEAT = "viridis"

SPINE_COLOR = "#aaaaaa"
SPINE_LW    = 1.2


# ──────────────────────────────────────────────────────────────────────────────
# COORDINATE / GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def transform_to_tcp_frame(block_pose, tcp_pose):
    block_pos = block_pose[:3]
    tcp_pos   = tcp_pose[:3]
    tcp_quat  = tcp_pose[3:]   # [w, x, y, z]
    tcp_rot   = R.from_quat([tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]])
    return tcp_rot.inv().apply(block_pos - tcp_pos)


def get_palm_bounding_box_corners_in_tcp_frame():
    X_min, X_max = -0.0375, 0.0425
    Y_min, Y_max = -0.0655, 0.0595
    Z_min_tcp    = -0.12
    Z_max_tcp    = Z_min_tcp + 0.0460

    corners = np.array([
        [X_max, Y_max, Z_min_tcp], [X_min, Y_max, Z_min_tcp],
        [X_min, Y_min, Z_min_tcp], [X_max, Y_min, Z_min_tcp],
        [X_max, Y_max, Z_max_tcp], [X_min, Y_max, Z_max_tcp],
        [X_min, Y_min, Z_max_tcp], [X_max, Y_min, Z_max_tcp],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    return corners, edges


def get_finger_schematic():
    X_min = -0.0395
    Z_mid = -0.12 + 0.030
    Z_h   = 0.014

    finger_len = 0.15
    finger_hw  = 0.0145

    finger_defs = [
        ('Index',  0.043,    '#4A90E2'),
        ('Middle', -0.00245, '#4A90E2'),
        ('Ring',   -0.0479,  '#4A90E2'),
    ]

    box_edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]

    fingers = []
    for name, yc, color in finger_defs:
        x0, x1 = X_min - finger_len, X_min
        y0, y1 = yc - finger_hw, yc + finger_hw
        z0, z1 = Z_mid - Z_h, Z_mid + Z_h
        c = np.array([[x1,y1,z0],[x0,y1,z0],[x0,y0,z0],[x1,y0,z0],
                       [x1,y1,z1],[x0,y1,z1],[x0,y0,z1],[x1,y0,z1]])
        fingers.append((name, c, color))

    # Thumb
    Y_max = 0.0615
    tx0, tx1 = 0.0135, 0.0425
    ty0, ty1 = Y_max, Y_max + 0.110
    z0, z1   = Z_mid - Z_h, Z_mid + Z_h
    tc = np.array([[tx1,ty1,z0],[tx0,ty1,z0],[tx0,ty0,z0],[tx1,ty0,z0],
                    [tx1,ty1,z1],[tx0,ty1,z1],[tx0,ty0,z1],[tx1,ty0,z1]])
    fingers.append(('Thumb', tc, '#4A90E2'))
    
    return fingers, box_edges


def plot_fingers_3d(ax3d, fingers, edges):
    for name, c, color in fingers:
        for i, j in edges:
            ax3d.plot([c[i,0],c[j,0]], [c[i,1],c[j,1]], [c[i,2],c[j,2]],
                      color=color, linestyle='--', linewidth=1.2, alpha=0.75)


def plot_fingers_2d(ax, fingers, edges, proj_axes):
    a, b = proj_axes
    for name, c, color in fingers:
        xs = [c[i, a] for i in range(8)]
        ys = [c[i, b] for i in range(8)]
        rect_x = [min(xs), max(xs), max(xs), min(xs), min(xs)]
        rect_y = [min(ys), min(ys), max(ys), max(ys), min(ys)]
        ax.plot(rect_x, rect_y, color=color, linestyle='--',
                linewidth=1.5, alpha=0.80)


def plot_bounding_box(ax, corners, edges, style='r--'):
    for i, j in edges:
        p1, p2 = corners[i], corners[j]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                color=style[0], linestyle=style[1:], linewidth=1.5, alpha=0.7)


def group_episodes_by_grasp_type(all_episodes):
    has_fi = any('active_finger_indices' in ep[-1] for ep in all_episodes)
    if not has_fi:
        return None
    groups = {}
    for ep in all_episodes:
        fi = ep[-1].get('active_finger_indices', None)
        key = tuple(sorted(fi.cpu().tolist() if isinstance(fi, torch.Tensor) else fi)) \
              if fi is not None else ('unknown',)
        groups.setdefault(key, []).append(ep)
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# AXIS STYLE HELPER  (mirrors curriculum_vs_baseline.py)
# ──────────────────────────────────────────────────────────────────────────────

def _style_2d_ax(ax):
    """
    Uniform spine + tick treatment for all 2-D panels.
    Keeps a clean rectangular box (no despine), matches curriculum style.
    """
    # All four spines visible, uniform weight and colour
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
        spine.set_color(SPINE_COLOR)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS,
        length=4,
        width=SPINE_LW,
        color=SPINE_COLOR,
        pad=4,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linewidth=0.6, alpha=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# 2-D KDE HEATMAP PANEL
# ──────────────────────────────────────────────────────────────────────────────

def _kde_heatmap(ax, x, y, palm_corners, fingers, proj_axes, xlabel, ylabel, title):
    a, b = proj_axes

    pad_x = (x.max() - x.min()) * 0.30
    pad_y = (y.max() - y.min()) * 0.30
    xi = np.linspace(x.min() - pad_x, x.max() + pad_x, 200)
    yi = np.linspace(y.min() - pad_y, y.max() + pad_y, 200)
    xx, yy = np.meshgrid(xi, yi)
    kernel = gaussian_kde(np.vstack([x, y]))
    zz = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    lvls = np.linspace(zz.min(), zz.max(), 14)[1:]
    cf = ax.contourf(xx, yy, zz, levels=lvls, cmap=CMAP_HEAT, alpha=0.88)
    ax.contour(xx, yy, zz, levels=lvls[::3], colors='white', linewidths=0.5, alpha=0.35)

    # Colorbar — extra pad so its label doesn't crowd the axis tick labels
    cbar = plt.colorbar(cf, ax=ax, fraction=0.030, pad=0.08)
    cbar.set_label("Density", fontsize=ANNOT_FS, labelpad=8)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(SPINE_LW)
    cbar.outline.set_edgecolor(SPINE_COLOR)

    # Palm bbox face per projection
    if proj_axes == (0, 1):
        face = [(0,1),(1,2),(2,3),(3,0)]
    elif proj_axes == (0, 2):
        face = [(0,4),(4,5),(5,1),(1,0)]
    else:  # (1, 2) YZ
        face = [(0,4),(4,7),(7,3),(3,0)]

    first = True
    for i, j in face:
        ax.plot([palm_corners[i,a], palm_corners[j,a]],
                [palm_corners[i,b], palm_corners[j,b]],
                color=C_PALM, linestyle='--', linewidth=2.0, alpha=0.85,
                label='Palm' if first else None)
        first = False

    _, finger_edges = get_finger_schematic()
    plot_fingers_2d(ax, fingers, finger_edges, proj_axes)

    ax.scatter([0], [0], s=160, marker='+', color=C_ORIGIN,
               linewidths=2.5, zorder=5, label='TCP origin')

    ax.set_xlabel(xlabel, fontsize=LABEL_FS, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=10)

    # Title anchored at a fixed offset so all panels sit at the same height
    ax.set_title(title, fontsize=TITLE_FS, pad=14)

    ax.set_aspect('equal')
    _style_2d_ax(ax)

    ax.legend(fontsize=ANNOT_FS, frameon=True, facecolor='white',
              framealpha=0.9, loc='upper right', borderpad=0.6,
              handlelength=1.4, labelspacing=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D PANEL
# ──────────────────────────────────────────────────────────────────────────────

def _3d_scatter_panel(ax3d, pts, palm_corners, fingers, palm_edges, title):
    all_c = np.concatenate([pts, palm_corners])
    mr = np.max(np.abs(all_c)) * 1.1
    g = 60

    xi, yi = np.linspace(-mr, mr, g), np.linspace(-mr, mr, g)
    xx_f, yy_f = np.meshgrid(xi, yi)
    zz_f = gaussian_kde(pts[:,[0,1]].T)(
        np.vstack([xx_f.ravel(), yy_f.ravel()])).reshape(xx_f.shape)
    lvls_f = np.linspace(zz_f.max() * 0.15, zz_f.max(), 10)
    ax3d.contourf(xx_f, yy_f, zz_f, levels=lvls_f, zdir='z', offset=-mr,
                  cmap=CMAP_HEAT, alpha=0.8)

    xi2, zi2 = np.linspace(-mr, mr, g), np.linspace(-mr, mr, g)
    xx_w, zz_w = np.meshgrid(xi2, zi2)
    dd_w = gaussian_kde(pts[:,[0,2]].T)(
        np.vstack([xx_w.ravel(), zz_w.ravel()])).reshape(xx_w.shape)
    lvls_w = np.linspace(dd_w.max() * 0.15, dd_w.max(), 10)
    ax3d.contourf(xx_w, dd_w, zz_w, levels=lvls_w, zdir='y', offset=mr,
                  cmap=CMAP_HEAT, alpha=0.8)

    yi3, zi3 = np.linspace(-mr, mr, g), np.linspace(-mr, mr, g)
    yy_l, zz_l = np.meshgrid(yi3, zi3)
    dd_l = gaussian_kde(pts[:,[1,2]].T)(
        np.vstack([yy_l.ravel(), zz_l.ravel()])).reshape(yy_l.shape)
    lvls_l = np.linspace(dd_l.max() * 0.15, dd_l.max(), 10)
    ax3d.contourf(dd_l, yy_l, zz_l, levels=lvls_l, zdir='x', offset=-mr,
                  cmap=CMAP_HEAT, alpha=0.8)

    ax3d.scatter(pts[:,0], pts[:,1], pts[:,2],
                 c=np.arange(len(pts)), cmap='viridis', s=50, alpha=0.6, depthshade=True)

    plot_bounding_box(ax3d, palm_corners, palm_edges, style='r--')
    plot_fingers_3d(ax3d, fingers, get_finger_schematic()[1])
    ax3d.scatter([0], [0], [0], c='red', s=200, marker='x', linewidths=3, label='TCP')

    ax3d.set_xlabel('X fwd', fontsize=TICK_FS, labelpad=2)
    ax3d.set_ylabel('Y lat', fontsize=TICK_FS, labelpad=2)
    ax3d.set_zlabel('Z up',  fontsize=TICK_FS, labelpad=2)
    ax3d.set_title(title, fontsize=TITLE_FS, pad=6)
    ax3d.tick_params(labelsize=TICK_FS - 2)
    ax3d.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax3d.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax3d.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax3d.set_xlim([-mr, mr]); ax3d.set_ylim([-mr, mr]); ax3d.set_zlim([-mr, mr])
    ax3d.set_box_aspect([1,1,1])
    ax3d.legend(fontsize=ANNOT_FS)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def visualize_block_positions_in_tcp_frame(all_episodes, saved_states_dir):
    grasp_groups = group_episodes_by_grasp_type(all_episodes)
    groups = grasp_groups if grasp_groups else {'all': all_episodes}

    n_groups = len(groups)
    palm_corners, palm_edges = get_palm_bounding_box_corners_in_tcp_frame()
    fingers, finger_edges    = get_finger_schematic()

    # ── Figure 1: 3-D scatter (one subplot per grasp type) ───────────────────
    fig_3d = plt.figure(figsize=(9 * n_groups, 9))
    fig_3d.suptitle("Block Position in TCP Frame — 3-D View",
                    fontsize=TITLE_FS + 4, fontweight='bold', y=1.01)

    # ── Figure 2: 2-D KDE panels (3 columns per grasp type) ──────────────────
    fig_2d, axes = plt.subplots(
        1, n_groups * 3,
        figsize=(9 * n_groups * 3, 8),
        gridspec_kw={"wspace": 0.25},
    )
    if n_groups * 3 == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    fig_2d.suptitle("Block Position in TCP Frame at Grasp — 2-D KDE Projections",
                    fontsize=TITLE_FS + 4, fontweight='bold', y=1.03)

    for g_idx, (cfg, eps) in enumerate(sorted(groups.items())):
        pts = []
        for ep in eps:
            s = ep[-1]
            if 'cube' in s and 'tcp_pose' in s:
                cw = s['cube'].cpu().numpy()     if isinstance(s['cube'],     torch.Tensor) else s['cube']
                tw = s['tcp_pose'].cpu().numpy() if isinstance(s['tcp_pose'], torch.Tensor) else s['tcp_pose']
                pts.append(transform_to_tcp_frame(cw, tw))
        if not pts:
            continue

        pts = np.array(pts)
        lbl = f"Fingers {list(cfg)}" if cfg != 'all' else "All Grasps"

        # 3-D panel
        ax3d = fig_3d.add_subplot(1, n_groups, g_idx + 1, projection='3d')
        _3d_scatter_panel(ax3d, pts, palm_corners, fingers, palm_edges,
                          title=f'{lbl}\n3-D View ({len(pts)} grasps)')

        # 2-D panels
        base = g_idx * 3
        _kde_heatmap(axes[base],   pts[:,0], pts[:,1], palm_corners, fingers,
                     proj_axes=(0,1),
                     xlabel='TCP X — forward (m)', ylabel='TCP Y — lateral (m)',
                     title=f'{lbl}  ·  Top (XY)')

        _kde_heatmap(axes[base+1], pts[:,0], pts[:,2], palm_corners, fingers,
                     proj_axes=(0,2),
                     xlabel='TCP X — forward (m)', ylabel='TCP Z — vertical (m)',
                     title=f'{lbl}  ·  Side (XZ)')

        _kde_heatmap(axes[base+2], pts[:,1], pts[:,2], palm_corners, fingers,
                     proj_axes=(1,2),
                     xlabel='TCP Y — lateral (m)', ylabel='TCP Z — vertical (m)',
                     title=f'{lbl}  ·  Front (YZ)')

        print(f"\n=== Block Position ({lbl}, n={len(pts)}) ===")
        for i, ax_n in enumerate("XYZ"):
            print(f"  {ax_n}: mean={pts[:,i].mean():.4f}  std={pts[:,i].std():.4f}")

    # Align xlabels so they all sit at the same vertical position regardless
    # of how set_aspect() resizes individual axes
    fig_2d.align_xlabels(axes)

    fig_3d.tight_layout()
    fig_2d.tight_layout()

    stem_3d = f"{saved_states_dir}/tcp_relative_positions_3d"
    stem_2d = f"{saved_states_dir}/tcp_relative_positions_2d"

    for stem, fig in [(stem_3d, fig_3d), (stem_2d, fig_2d)]:
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{stem}.pdf", format='pdf', bbox_inches='tight')
        print(f"Saved → {stem}.png / .pdf")

    plt.close('all')


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "intermediate_states/fingers_0_1_2_3_active_1_palm_True_seed_10"
    saved_states = torch.load(f"{DATA_DIR}/grasp_to_push_img.pt")
    visualize_block_positions_in_tcp_frame(saved_states, DATA_DIR)