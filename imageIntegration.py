import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os
from scipy.integrate import cumulative_trapezoid as cumtrapz
from matplotlib import rcParams

h = 6.62607015e-34  # Planck's constant (J·s)
c = 2.99792458e8    # speed of light (m/s)
wavelength = 328.1629601e-9 #wavelength of light
gamma_nat = 1.4e8
I_sat = (np.pi * h * c * gamma_nat)/(3 * wavelength**3)

# --- Configuration ---
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

fontsz = 16
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'


focus_distance = None # Only show a certain distance

save_all_plots = True
#save_all_plots = False

pixel_size = 3.45e-6 #m
pixel_area = pixel_size**2 #3.45 x 3.45 micrometers squared
photon_energy = h * c / wavelength

p_total = 1.205e-6 #W #(0.460-0.023)*1e-6
print("TOTAL MEASURED POWER = " + str(p_total))

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
allNormal = False
base_path = "Beam_Images_New/"

def to3string(dist: int):
	"""Converts integers to 3 digit strings, i.e. 25 -> 025"""
	return str(dist).zfill(3)

def round_sig(x, sig=3):
	"""Round a number to a given number of significant figures."""
	if x == 0:
		return 0
	return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

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
	img = img / (exposure * 255) #gives unscaled intensity values to each pixel

	sf = 1

	#print(np.sum(img))

	if normalise:
		img = img / img.max()
	# beam centre
	if centre is not None:
		cx, cy = centre
	else:
		cy, cx = center_of_mass(img)

	# coordinate grid
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
	r = np.linspace(0, r_max, nr) #convert pixel lengths to real lengths in m
	r_m = r * pixel_size
	theta = np.linspace(-np.pi, np.pi, nt) #theta in radians
	r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")
	x_p = r_grid * np.cos(theta_grid) + cx
	y_p = r_grid * np.sin(theta_grid) + cy
	polar_img = map_coordinates(img, [y_p, x_p], order=1)

	# integration
	P_r_unnorm = np.trapezoid(polar_img, theta, axis=1)#gives the power per unit radial length
	P_total = np.trapezoid(P_r_unnorm * r, r)#Integrates over r to get the power, need to apply a scale factor so it equals the total measured power
	P_encircled = cumtrapz(P_r_unnorm * r, r, initial=0)
	#print(distance, P_total)
	scale_factor = p_total / (P_total)
	#Total measured power is known

	profile_x = r * pixel_size #radial size in m
	profile_y = (P_r_unnorm * (scale_factor / pixel_area)) / (np.pi * 2) # average intensity per radius
	r_safe = r
	r_safe[0] = r_safe[1]
	I_avg_area = P_encircled / (np.pi * r_safe**2)  # avoid divide-by-zero at r=0
	I_avg_area[0] = 0
	I_avg_area_scaled = I_avg_area * (scale_factor / pixel_area)

	I_Peak = np.max(profile_y) #peak intensity of radial average intensity distribution
	I_Ave_Peak = np.max(I_avg_area_scaled)
	profile_label = "I(r)"
	polar_extent = (theta.min(), theta.max(), r.min(), r.max())
	polar_xlabel = "θ (radians)"
	polar_ylabel = "r (pixels)"

	return img, polar_img, profile_x, profile_y, P_total, (cx, cy), polar_extent, polar_xlabel, polar_ylabel, profile_label, I_Peak, I_avg_area_scaled, I_Ave_Peak


# --- Process all images ---
results = {}
for d, info in beam_images.items():
	img, polar_img, x_prof, y_prof, P, centre, polar_extent, polar_xlabel, polar_ylabel, profile_label, I_max, I_ave_profile, I_ave_peak = process_image(
		d,
		centre=info.get("centre"),
		exposure = info.get("exposure") or default_exposure,
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
		"I_max": I_max,
		"I_Ave_profile" : I_ave_profile,
		"I_Ave_max": I_ave_peak,
	}

plot_main = not False
if plot_main:
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
		I_max = data["I_max"]
		I_ave_peak = data["I_Ave_max"]

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
		axs[i, 2].plot(x_prof * 1e3, y_prof / I_sat, label=r"$I(r)$ / $I_{sat}$", zorder = 1)
		axs[i, 2].plot(x_prof * 1e3, data["I_Ave_profile"] / I_sat, '--', label=r"$I_{avg}(r)$ / $I_{sat}$", zorder = 0)
		axs[i, 2].legend(loc="upper right", fontsize=8)

		axs[i, 2].set_ylabel(r"$I$ / $I_{sat}$")
		I_round = round_sig(I_max, 4)
		I_round2 = round_sig(I_ave_peak, 4)

		axs[i, 2].text(0.98,0.8,
			r"Peak $I(r) = $" + f"{I_round}" + " ($W~m^{-2}$)",
			ha='right', va='top',
			transform=axs[i, 2].transAxes,
			fontsize=10,)
			#bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3'))

		axs[i, 2].text(0.98,0.7,
			r"Peak $I_{avg} = $" + f"{I_round2}" + " ($W~m^{-2}$)",
			ha='right', va='top',
			transform=axs[i, 2].transAxes,
			fontsize=10,)

		if i == 0:
			axs[i, 2].set_title(f"{profile_label} profile")
		if i < n - 1:
			axs[i, 2].set_xlabel("")
			axs[i, 2].set_xticklabels([])
		else:
			axs[i, 2].set_xlabel("r (mm)")

		axs[i, 2].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

		# Label each row on the left with its distance
		axs[i, 0].text(-0.25, 0.5, f"{d} mm", transform=axs[i, 0].transAxes,
					rotation=90, va="center", ha="right", fontsize=10)

	plt.subplots_adjust(hspace=0.1, wspace=0.25)

	if focus_distance != None and save_all_plots:
		plt.savefig(to3string(focus_distance)+"_Plots", dpi=300, bbox_inches='tight')

	plt.show()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D plotting)

# --- Conditional summary plots ---
if focus_distance is None:

	# --- Create a heatmap of all I(r) profiles (radius vs distance) ---

	# Extract and align all I(r) profiles
	distances = sorted(results.keys())
	r_values = results[distances[0]]["x_prof"]

	# Interpolate all profiles to a common r grid (in case lengths differ)
	r_common = np.linspace(0, max(r_values), 500)
	I_profiles = []

	for d in distances:
		x_prof = results[d]["x_prof"]
		y_prof = results[d]["y_prof"]
		I_interp = np.interp(r_common, x_prof, y_prof)/I_sat
		I_profiles.append(I_interp)

	I_profiles = np.array(I_profiles)  # shape = (num_distances, num_r_points)

	# Transpose so that radius is vertical (y-axis)
	I_profiles = I_profiles.T  # shape = (num_r_points, num_distances)

	# Create the heatmap
	plt.figure(figsize=(10, 6))
	extent = [distances[0], distances[-1], r_common[0]*1e3, r_common[-1]*1e3]  # x=distance (mm), y=radius (mm)
	plt.imshow(
		I_profiles,
		extent=extent,
		aspect='auto',
		origin='lower',
		cmap='inferno',
		interpolation='bilinear'
	)
	cbar = plt.colorbar(label=r"$I(r)$ / $I_{sat}$")
	cbar.formatter.set_powerlimits((-3, 3))
	cbar.formatter.set_useMathText(True)
	cbar.update_ticks()
	plt.xlabel("Distance (mm)")
	plt.ylabel("Radius (mm)")
	plt.title("Radial Intensity Profiles Heatmap")
	plt.tight_layout()

	plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))

	if save_all_plots:
		plt.savefig("I_r_heatmap.png", dpi=300, bbox_inches='tight')

	plt.show()

	# --- Create a heatmap of all ⟨I⟩(r) profiles (encircled average intensity vs distance) ---

	# Interpolate all ⟨I⟩(r) profiles to the same r grid
	I_Ave_profiles = []

	for d in distances:
		x_prof = results[d]["x_prof"]
		y_ave = results[d]["I_Ave_profile"]
		I_Ave_interp = np.interp(r_common, x_prof, y_ave)/I_sat
		I_Ave_profiles.append(I_Ave_interp)

	I_Ave_profiles = np.array(I_Ave_profiles)  # shape = (num_distances, num_r_points)
	I_Ave_profiles = I_Ave_profiles.T  # transpose so radius is on y-axis

	# Create the heatmap
	plt.figure(figsize=(10, 6))
	extent = [distances[0], distances[-1], r_common[0]*1e3, r_common[-1]*1e3]  # x=distance (mm), y=radius (mm)
	plt.imshow(
		I_Ave_profiles,
		extent=extent,
		aspect='auto',
		origin='lower',
		cmap='inferno',
		interpolation='bilinear'
	)
	cbar = plt.colorbar(label=r"$I_{avg}(r)$ / $I_{sat}$")
	cbar.formatter.set_powerlimits((-3, 3))
	cbar.formatter.set_useMathText(True)
	cbar.update_ticks()
	plt.xlabel("Distance (mm)")
	plt.ylabel("Radius (mm)")
	plt.title("Encircled-Average Intensity Profiles Heatmap")
	plt.tight_layout()

	plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

	if save_all_plots:
		plt.savefig("I_Ave_r_heatmap.png", dpi=300, bbox_inches='tight')

	plt.show()

	distances = list(results.keys())
	I_max_values = np.array([results[d]["I_max"] for d in distances]) / I_sat
	I_Ave_max_values = np.array([results[d]["I_Ave_max"] for d in distances]) / I_sat

	plt.figure(figsize=(8, 5))
	plt.plot(distances, I_max_values, "o-", color="tab:red", zorder = 1, label = r"Peak $I(r)$")
	plt.plot(distances, I_Ave_max_values , "x-", color="tab:orange", zorder = 0, label = r"Peak $I_{avg}$")
	plt.xlabel("Distance from 0 point (mm)")
	plt.ylabel(r"$I_{max}$ / $I_{sat}$")
	plt.title("Peak intensity vs distance")
	plt.tight_layout()
	plt.legend(loc="lower right")

	plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

	if save_all_plots:
		plt.savefig("I_max_distance_graph", dpi=300, bbox_inches='tight')
	plt.show()

else:
	print(f"Displayed only {focus_distance} mm (single-image mode).")