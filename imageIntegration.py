import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os

# --- Configuration ---
os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

# one dict per image: centre + exposure
beam_images = {
    0:   {"centre": (717, 548), "exposure": None},
    50:  {"centre": (672, 536), "exposure": None},
    100: {"centre": (736, 544), "exposure": None},
    175: {"centre": (743, 553), "exposure": None},
    225: {"centre": (729, 551), "exposure": None},
    325: {"centre": (699, 547), "exposure": None},
    425: {"centre": (699, 519), "exposure": None},
}

default_exposure = 1e-9   # s
allNormal = False         # keep False when exposure is meaningful
base_path = "Ag_Spec_Matt/"

def to3string(dist: int):
    return str(dist).zfill(3)

def process_image(distance, centre=None, exposure=None, normalise=False):
    path = f"{base_path}{to3string(distance)}_0.bmp"
    print(f"\nProcessing: {path}")

    img = plt.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)

    # image size is 1080 (y) × 1440 (x)
    ny, nx = img.shape   # ny=1080, nx=1440

    # exposure
    if exposure is None:
        exposure = default_exposure
        print(f"No exposure time specified — using default {exposure:.4f} s")
    else:
        print(f"Exposure time: {exposure:.4f} s")

    # convert to something ∝ power
    img = img / exposure

    if normalise:
        img = img / img.max()
        print(" -> Image normalised before integration")

    # centre
    if centre is not None:
        cx, cy = centre
        print(f"Using manual centre: (x={cx}, y={cy})")
    else:
        cy, cx = center_of_mass(img)
        print(f"Detected centre of mass: (x={cx:.2f}, y={cy:.2f})")

    # build Cartesian grid centred on beam
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    X, Y = np.meshgrid(x, y)

    # compute r_max as distance from centre to farthest corner
    corners = np.array([
        [0 - cx,     0 - cy],
        [nx-1 - cx,  0 - cy],
        [0 - cx,     ny-1 - cy],
        [nx-1 - cx,  ny-1 - cy],
    ])
    r_max = np.sqrt((corners**2).sum(axis=1)).max()   # ≈ 900 for your case

    # high-res polar grid
    nr = int(np.ceil(r_max))          # ~1 pixel radial resolution
    nt = int(np.ceil(2 * np.pi * r_max))   # ~1 pixel around outer ring
    nt = min(nt, 6000)  # safety cap

    r = np.linspace(0, r_max, nr)
    theta = np.linspace(-np.pi, np.pi, nt)

    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    x_p = r_grid * np.cos(theta_grid) + cx
    y_p = r_grid * np.sin(theta_grid) + cy

    # interpolate to polar
    polar_img = map_coordinates(img, [y_p, x_p], order=1)

    # integrate over theta to get power vs radius
    I_r_unnorm = np.trapezoid(polar_img, theta, axis=1)

    # total power (relative)
    P_total = 2 * np.pi * np.trapezoid(I_r_unnorm * r, r)

    # normalised profile for plotting
    I_r = I_r_unnorm / I_r_unnorm.max()

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Distance = {distance} mm", fontsize=14)

    axs[0].imshow(img, cmap='inferno', origin='lower')
    axs[0].plot(cx, cy, 'bo', markersize=3, alpha=0.6,
                label=('Manual centre' if centre else 'Detected centre'))
    axs[0].legend()
    axs[0].set_title("Original beam image")
    axs[0].set_xlabel("x (pixels)")
    axs[0].set_ylabel("y (pixels)")

    axs[1].imshow(
        polar_img,
        extent=(theta.min(), theta.max(), r.min(), r.max()),
        aspect='auto',
        cmap='inferno',
        origin='lower'
    )
    axs[1].set_title("Beam image in polar coordinates (dense)")
    axs[1].set_xlabel("θ (radians)")
    axs[1].set_ylabel("r (pixels)")

    axs[2].plot(r, I_r)
    axs[2].set_title("Radial power density profile")
    axs[2].set_xlabel("r (pixels)")
    axs[2].set_ylabel("Normalised power density (a.u.)")

    plt.tight_layout()
    plt.show()

    return r, I_r, P_total, (cx, cy), exposure

# run through all
results = {}
for d, info in beam_images.items():
    r, I_r, P, centre_used, exp_used = process_image(
        d,
        centre=info.get("centre"),
        exposure=info.get("exposure"),
        normalise=allNormal
    )
    results[d] = {
        "r": r,
        "I_r": I_r,
        "P_total": P,
        "centre": centre_used,
        "exposure": exp_used,
    }



#name = "Ag_Spec_Matt/375_0.bmp"

##img1 = mpimg.imread(name)
##plt.imshow(img1)

#newname = name[13:-3]+"png"

#plt.savefig(newname, dpi=300, bbox_inches='tight')