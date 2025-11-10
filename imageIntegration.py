import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os

# --- Configuration ---
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

beam_images = {
    0:   {"centre": (717, 548), "exposure": None},
    50:  {"centre": (672, 536), "exposure": None},
    100: {"centre": (736, 544), "exposure": None},
    175: {"centre": (743, 553), "exposure": None},
    225: {"centre": (729, 551), "exposure": None},
    325: {"centre": (699, 547), "exposure": None},
    375: {"centre": (705, 552), "exposure": None},
    425: {"centre": (699, 519), "exposure": None},
}

default_exposure = 1e-6  # s
allNormal = True
base_path = "Ag_Spec_Matt/"

def to3string(dist: int):
    return str(dist).zfill(3)

def process_image(distance, centre=None, exposure=None, normalise=False):
    """Process a single beam image and return all derived quantities."""
    path = f"{base_path}{to3string(distance)}_0.bmp"
    img = plt.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    ny, nx = img.shape

    # exposure handling
    if exposure is None:
        exposure = default_exposure
    img = img / exposure
    if normalise:
        img = img / img.max()

    # beam centre
    if centre is not None:
        cx, cy = centre
    else:
        cy, cx = center_of_mass(img)

    # coordinate grid
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    X, Y = np.meshgrid(x, y)
    corners = np.array([
        [0 - cx,     0 - cy],
        [nx-1 - cx,  0 - cy],
        [0 - cx,     ny-1 - cy],
        [nx-1 - cx,  ny-1 - cy],
    ])
    r_max = np.sqrt((corners**2).sum(axis=1)).max()

    # polar grid high-res
    nr = int(np.ceil(r_max))
    nt = int(np.ceil(2 * np.pi * r_max))
    nt = min(nt, 6000)
    r = np.linspace(0, r_max, nr)
    theta = np.linspace(-np.pi, np.pi, nt)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")
    x_p = r_grid * np.cos(theta_grid) + cx
    y_p = r_grid * np.sin(theta_grid) + cy
    polar_img = map_coordinates(img, [y_p, x_p], order=1)

    # integration

    I_r_unnorm = np.trapezoid(polar_img, theta, axis=1)
    P_total = np.trapezoid(I_r_unnorm * r, r)
    profile_x = r
    profile_y = I_r_unnorm / I_r_unnorm.max()
    profile_label = "I(r)"
    polar_extent = (theta.min(), theta.max(), r.min(), r.max())
    polar_xlabel = "θ (radians)"
    polar_ylabel = "r (pixels)"

    return img, polar_img, profile_x, profile_y, P_total, (cx, cy), polar_extent, polar_xlabel, polar_ylabel, profile_label

# --- Process all images ---
results = {}
for d, info in beam_images.items():
    img, polar_img, x_prof, y_prof, P, centre, polar_extent, polar_xlabel, polar_ylabel, profile_label = process_image(
        d,
        centre=info.get("centre"),
        exposure=info.get("exposure"),
        normalise=allNormal,
    )
    results[d] = {
        "img": img,
        "polar_img": polar_img,
        "x_prof": x_prof,
        "y_prof": y_prof,
        "P_total": P,
        "centre": centre,
        "polar_extent": polar_extent,
        "polar_xlabel": polar_xlabel,
        "polar_ylabel": polar_ylabel,
        "profile_label": profile_label,
    }

# --- Combined subplot grid ---
n = len(results)
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12, 3.2 * n))
# Create a custom GridSpec layout
gs = gridspec.GridSpec(
    nrows=n, ncols=3, figure=fig,
    width_ratios=[1, 1, 1],   # same width for each column
    wspace=0.25,               # small global horizontal space
    hspace=0.15               # same vertical space as before
)

axs = np.empty((n, 3), dtype=object)
for i in range(n):
    axs[i, 0] = fig.add_subplot(gs[i, 0])   # left column
    axs[i, 1] = fig.add_subplot(gs[i, 1])   # middle column
    axs[i, 2] = fig.add_subplot(gs[i, 2])   # right column

if n == 1:
    axs = np.expand_dims(axs, 0)

#fig.suptitle(f"Beam analysis — integration: {integration_order.replace('_', ' ')}", fontsize=14)

for i, (d, data) in enumerate(results.items()):
    img = data["img"]
    polar_img = data["polar_img"]
    cx, cy = data["centre"]
    x_prof, y_prof = data["x_prof"], data["y_prof"]
    polar_extent = data["polar_extent"]
    polar_xlabel = data["polar_xlabel"]
    polar_ylabel = data["polar_ylabel"]
    profile_label = data["profile_label"]

    # --- Column 1: Original image ---
    axs[i, 0].imshow(img, cmap="inferno", origin="lower")
    axs[i, 0].plot(cx, cy, "bo", markersize=3, alpha=0.6)
    axs[i, 0].set_ylabel("y (pixels)")
    if i == 0:
        axs[i, 0].set_title("Original")
    if i < n - 1:
        axs[i, 0].set_xlabel("")
        axs[i, 0].set_xticklabels([])
    else:
        axs[i, 0].set_xlabel("x (pixels)")

    # --- Column 2: Polar-transformed image ---
    axs[i, 1].imshow(polar_img, extent=polar_extent, aspect="auto", cmap="inferno", origin="lower")
    axs[i, 1].set_ylabel(polar_ylabel)
    if i == 0:
        axs[i, 1].set_title("Polar coordinates")
    if i < n - 1:
        axs[i, 1].set_xlabel("")
        axs[i, 1].set_xticklabels([])
    else:
        axs[i, 1].set_xlabel(polar_xlabel)

    # --- Column 3: Integrated profile ---
    axs[i, 2].plot(x_prof, y_prof)
    axs[i, 2].set_ylabel("Norm P/r")
    if i == 0:
        axs[i, 2].set_title(f"{profile_label} profile")
    if i < n - 1:
        axs[i, 2].set_xlabel("")
        axs[i, 2].set_xticklabels([])
    else:
        axs[i, 2].set_xlabel(profile_label.split("(")[1][:-1])

    # Label each row on the left with its distance
    axs[i, 0].text(-0.25, 0.5, f"{d} mm", transform=axs[i, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=10)

for i in range(n):
    pos = axs[i, 0].get_position()
    # move right edge closer to middle column by 2% of figure width
    axs[i, 0].set_position([
        pos.x0 + 0.06,  # shift right by 3% of total figure width
        pos.y0,
        pos.width,
        pos.height
    ])

#plt.savefig("integratedImages_Old", dpi=300, bbox_inches='tight')

plt.show()



# --- Overlay of all profiles ---
plt.figure(figsize=(8, 5))
for d, data in results.items():
    plt.plot(data["x_prof"], data["y_prof"], label=f"{d} mm")
plt.xlabel(data["profile_label"].split("(")[1][:-1] + " (pixels or radians)")
plt.ylabel("Norm P/r (a.u.)")
#plt.title(f"All {data['profile_label']} profiles — {integration_order.replace('_', ' ')} integration")
plt.legend()
plt.tight_layout()
#plt.savefig("all_integratedImages_Old", dpi=300, bbox_inches='tight')
plt.show()

# --- Total power vs distance ---
plt.figure(figsize=(6, 4))
distances = list(beam_images.keys())
powers = [results[d]["P_total"] for d in distances]
plt.plot(distances, powers, "o-", color="tab:red")
plt.xlabel("Distance (mm)")
plt.ylabel("Relative total power (a.u.)")
#plt.title(f"Total integrated power vs distance\n(integration: {integration_order.replace('_', ' ')})")
plt.tight_layout()
#plt.savefig("powers_Old", dpi=300, bbox_inches='tight')
plt.show()



#name = "Ag_Spec_Matt/375_0.bmp"

##img1 = mpimg.imread(name)
##plt.imshow(img1)

#newname = name[13:-3]+"png"

#plt.savefig(newname, dpi=300, bbox_inches='tight')