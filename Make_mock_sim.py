# imports
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import os
import imageio.v2 as imageio
from tqdm import tqdm
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Generate a mock simulation set.")
parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to generate [default=100].")
parser.add_argument("--img_size", type=int, default=1000, help="Linear size of the square image (pixels) [default=1000].")
parser.add_argument("--init_ap", type=float, default=10, help="Initial aperture angle in degrees [default=10].")
parser.add_argument("--final_ap", type=float, default=160, help="Final aperture angle in degrees [default=160].")
parser.add_argument("--init_sigma", type=float, default=0.5, help="Initial Gaussian blur sigma [default=0.5].")
parser.add_argument("--final_sigma", type=float, default=30, help="Final Gaussian blur sigma [default=30].")
parser.add_argument("--dec_len", type=float, default=0.2, help="Characteristic radial decay length, in unit of image size [default=0.2].")
parser.add_argument("--inside_noise", type=float, default=0.15, help="Noise amplitude inside the cone [default=0.15].")
parser.add_argument("--outside_noise", type=float, default=0.15, help="Noise amplitude outside the cone [default=0.15].")
parser.add_argument("--save_frames", action="store_true", help="Select to save to disk an image every 10 frames.")
args = parser.parse_args()

# Settings
num_frames = args.num_frames
img_size = args.img_size
initial_centre = (img_size // 2, img_size // 2)  # (y, x)
final_centre = (img_size // 2, 20)  # towards left
initial_aperture_deg = args.init_ap
final_aperture_deg = args.final_ap
initial_sigma = args.init_sigma
final_sigma = args.final_sigma
decay_length = img_size * args.dec_len
noise_amplitude_signal = args.inside_noise
noise_amplitude_background = args.outside_noise
save_frames = args.save_frames

# Precompute coordinate grid
y, x = np.indices((img_size, img_size))
frames = []

# --- First: Build first frame to compute initial minimum cone value ---
# Frame 0 interpolation
t0 = 0
aperture_0 = initial_aperture_deg
sigma_0 = initial_sigma
centre_y_0 = initial_centre[0]
centre_x_0 = initial_centre[1]

x_rel0 = x - centre_x_0
y_rel0 = y - centre_y_0

angles0 = np.arctan2(y_rel0, x_rel0)
r0 = np.sqrt(x_rel0**2 + y_rel0**2)
angles_deg0 = np.degrees(angles0)
angles_deg0 = (angles_deg0 + 360) % 360
angles_deg0[angles_deg0 > 180] -= 360

half_aperture_0 = aperture_0 / 2
angular_mask0 = np.zeros_like(angles_deg0, dtype=np.float32)
angular_mask0[np.abs(angles_deg0) <= half_aperture_0] = 1.0
edge_distance0 = np.clip(half_aperture_0 - np.abs(angles_deg0), 0, 1)
angular_mask0 *= edge_distance0
radial_mask0 = np.exp(-r0 / decay_length)
mask0 = angular_mask0 * radial_mask0

blurred0 = gaussian_filter(mask0, sigma=sigma_0)

# Compute minimum nonzero value inside the cone
cone_pixels0 = blurred0[blurred0 > 0]
initial_cone_min = cone_pixels0.min()
background_base_level = 0.5 * initial_cone_min

print(f"Initial cone minimum value: {initial_cone_min:.5f}")
print(f"Background base level: {background_base_level:.5f}")


for i in tqdm(range(num_frames)):
    # Interpolation factor
    t = i / (num_frames - 1)
    
    # Interpolate aperture and sigma
    aperture = initial_aperture_deg + (final_aperture_deg - initial_aperture_deg) * t
    sigma = initial_sigma + (final_sigma - initial_sigma) * t
    
    # Interpolate moving apex position
    current_centre_y = initial_centre[0]
    current_centre_x = initial_centre[1] + (final_centre[1] - initial_centre[1]) * t

    # Relative coordinates
    x_rel = x - current_centre_x
    y_rel = y - current_centre_y

    angles = np.arctan2(y_rel, x_rel)
    r = np.sqrt(x_rel**2 + y_rel**2)

    angles_deg = np.degrees(angles)
    angles_deg = (angles_deg + 360) % 360
    angles_deg[angles_deg > 180] -= 360

    # Angular mask
    half_aperture = aperture / 2
    angular_mask = np.zeros_like(angles_deg, dtype=np.float32)
    angular_mask[np.abs(angles_deg) <= half_aperture] = 1.0
    
    # Smooth edge
    edge_distance = np.clip(half_aperture - np.abs(angles_deg), 0, 1)
    angular_mask *= edge_distance

    # Radial decay
    radial_mask = np.exp(-r / decay_length)

    # Final mask (cone signal)
    mask = angular_mask * radial_mask

    # Blur the cone
    blurred_cone = gaussian_filter(mask, sigma=sigma)

    # Add noise to the cone region
    signal_noise = np.random.normal(loc=0.0, scale=noise_amplitude_signal, size=blurred_cone.shape)
    noisy_cone = blurred_cone + signal_noise
    noisy_cone = np.clip(noisy_cone, 0.0, 1.0)

    # Create background noise
    background_noise = np.random.normal(loc=0.0, scale=noise_amplitude_background, size=noisy_cone.shape)
    background = background_base_level + background_noise
    background = np.clip(background, 0.0, 1.0)

    # Create a mask for cone vs background
    cone_region = (blurred_cone > 0)

    # Combine cone and background
    final_frame = np.where(cone_region, noisy_cone, background)

    frames.append(final_frame)
frames=np.array(frames)


# Create output directory if it does not exist
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)


# Save an image of a few frames
if save_frames:
    inferno_cmap = cm.inferno
    for idx, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"frame_{idx:03d}.png")
        # Apply colormap: map [0,1] --> RGBA with inferno
        frame_coloured = inferno_cmap(frame)  # returns RGBA array
        # Drop alpha channel (keep only RGB)
        frame_rgb = (255 * frame_coloured[..., :3]).astype(np.uint8)
            # Save RGB image
        imageio.imwrite(filename, frame_rgb)


# save the sim to disk
np.save('Simulation.npy', frames)
