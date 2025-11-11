import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os

# --- Configuration ---
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())
focus_distance = None

beam_images = {
    0:   {"centre": (748, 532), "exposure": None},
    25:   {"centre": (747, 523), "exposure": None},
    50:  {"centre": (751, 588), "exposure": None},
    75:  {"centre": (751, 549), "exposure": None},    
    100: {"centre": (751, 541), "exposure": None},
    125:   {"centre": (750, 556), "exposure": None},
    150:  {"centre": (750, 522), "exposure": None},
    175:  {"centre": (748, 534), "exposure": None}, 
    200: {"centre": (748, 493), "exposure": None},
    225:   {"centre": (748, 459), "exposure": None},
    250:  {"centre": (748, 517), "exposure": None},
    275:  {"centre": (749, 546), "exposure": None}, 
    300: {"centre": (751, 534), "exposure": None},
    325:   {"centre": (753, 521), "exposure": None},
    350:  {"centre": (756, 465), "exposure": None},
    375:  {"centre": (761, 501), "exposure": None}, 
    400: {"centre": (768, 477), "exposure": None},
    425: {"centre": (778, 519), "exposure": None},
    450: {"centre": (790, 585), "exposure": None},
    475: {"centre": (805, 525), "exposure": None},
}

default_exposure = 12.097e-3  # s
allNormal = True
base_path = "Beam_Images_New/"

def to3string(dist: int):
    return str(dist).zfill(3)

def process_image(distance, centre=None, exposure=None, normalise=False):
    """Process a single beam image and return all derived quantities."""
    path = f"{base_path}{to3string(distance)}_0_12_097ms.bmp"
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
    profile_label = "P/r"
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

# --- Decide what to plot ---
if focus_distance is not None:
    # Only plot the chosen distance
    distances_to_plot = [focus_distance]
else:
    # Plot all
    distances_to_plot = list(results.keys())

n = len(distances_to_plot)
fig, axs = plt.subplots(n, 3, figsize=(12, 3.2 * n))

if n == 1:
    axs = np.expand_dims(axs, 0)

#fig.suptitle(f"Beam analysis — integration: {integration_order.replace('_', ' ')}", fontsize=14)

for i, d in enumerate(distances_to_plot):
    data = results[d]
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
        axs[i, 2].set_xlabel("P/r (W·px⁻¹)")

    # Label each row on the left with its distance
    axs[i, 0].text(-0.25, 0.5, f"{d} mm", transform=axs[i, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=10)

plt.subplots_adjust(hspace=0.1, wspace=0.25)

#plt.savefig("integratedImages_Old", dpi=300, bbox_inches='tight')

plt.show()


# --- Conditional summary plots ---
if focus_distance is None:
    # --- Overlay of all profiles ---
    plt.figure(figsize=(8, 5))
    for d, data in results.items():
        plt.plot(data["x_prof"], data["y_prof"], label=f"{d} mm")
    plt.xlabel(data["profile_label"] + " (pixels)")
    plt.ylabel("Norm P/r (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Total power vs distance ---
    plt.figure(figsize=(6, 4))
    distances = list(beam_images.keys())
    powers = [results[d]["P_total"] for d in distances]
    plt.plot(distances, powers, "o-", color="tab:red")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Relative total power (a.u.)")
    plt.tight_layout()
    plt.show()

else:
    print(f"Displayed only {focus_distance} mm (single-image mode).")




#name = "Ag_Spec_Matt/375_0.bmp"

##img1 = mpimg.imread(name)
##plt.imshow(img1)

#newname = name[13:-3]+"png"

#plt.savefig(newname, dpi=300, bbox_inches='tight')