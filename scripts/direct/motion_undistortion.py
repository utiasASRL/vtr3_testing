## this file aims to undistort the radar images using the current odometry estimates
## as well as all the other preprocessing steps

import os
import cv2
import numpy as np
import torch
import torchvision
import yaml
from pylgmath import Transformation

T_radar_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.025],
                                                 [0.000, -1.000 , 0.000, -0.002],
                                                 [0.000 ,0.000, -1.000 , 1.032],
                                                 [0.000 , 0.000 ,0.000, 1.000]]))

def preprocessing_polar_image(polar_image, device):
    polar_intensity = torch.tensor(polar_image).to(device)
    polar_std = torch.std(polar_intensity, dim=1)
    polar_mean = torch.mean(polar_intensity, dim=1)
    polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
    polar_intensity[polar_intensity < 0] = 0
    polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
    polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
    polar_intensity[torch.isnan(polar_intensity)] = 0

    return polar_intensity


def radar_polar_to_cartesian(fft_data, azimuths, radar_resolution=0.040308, cart_resolution=0.2384, cart_pixel_width=640,
                             interpolate_crossover=False, fix_wobble=False):
    # TAKEN FROM PYBOREAS
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels)
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    # print("in radar_polar_to_cartesian")
    # Compute the range (m) captured by pixels in cartesian scan
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    
    # Compute the value of each cartesian pixel, centered at 0
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)

    X, Y = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution

    # print("------")
    # print("sample_angle.shape",sample_angle.shape)
    # print("azimuths[0]",azimuths[0])
    # print("azimuth step shape" ,azimuth_step.shape)

    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    M = azimuths.shape[0]
    azms = azimuths.squeeze()
    if fix_wobble:
        c3 = np.searchsorted(azms, sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azms[c3]
        diff = sample_angle.squeeze() - a3
        a2 = azms[c2]
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)


def motion_undistortion(
    polar_image: np.ndarray,
    azimuth_angles: np.ndarray,
    azimuth_timestamps: np.ndarray,
    T_v_w: np.ndarray,
    dt: float,
    device: torch.device,
    radar_res: float = 0.040308,      # m per range-bin (CIR-204 default)
) -> torch.Tensor:
    """
    Undistort a polar radar scan by **applying a 3 × 3 homogeneous
    transform for every beam**.  Nearest-neighbour remapping only.
    Output frame = radar pose at the first beam of the scan.
    """
    T_sr_np   = T_radar_robot.matrix()                  # robot ➔ radar
    T_sr_inv  = np.linalg.inv(T_sr_np)

    # ------------------------------------------------------------------
    # 0.  Pre-processing
    # ------------------------------------------------------------------
    polar_intensity = preprocessing_polar_image(polar_image, device)
    A, R = polar_intensity.shape      # usually 400 × 1712

    # ------------------------------------------------------------------
    # 1.  Constant planar twist  (vx, vy, ω)  from the inter-scan Δpose
    # ------------------------------------------------------------------
    Tw_radar = torch.as_tensor(
        T_sr_np @ T_v_w @ T_sr_inv, device=device, dtype=torch.float32
    )                                            # radar₂ ➔ radar₁

    v_xy = Tw_radar[:2, 3] / dt                  # m s⁻¹ in radar axes
    print(f"v_xy = {v_xy}")
    yaw  = torch.atan2(Tw_radar[1, 0], Tw_radar[0, 0])
    w_z  = yaw / dt

    # ------------------------------------------------------------------
    # 2.  Acquisition-time metadata for each beam
    # ------------------------------------------------------------------
    az = torch.as_tensor(azimuth_angles.squeeze(), device=device, dtype=torch.float32)
    t_beam = torch.as_tensor(azimuth_timestamps.squeeze(), device=device, dtype=torch.float32)
    t_beam  = t_beam - t_beam[0]                           # τᵢ per beam (A,)

    # ------------------------------------------------------------------
    # 3.  Cartesian coordinates for every pixel in the *measured* scan
    # ------------------------------------------------------------------
    range_bins = torch.arange(R, device=device, dtype=torch.float32)
    r_grid  = range_bins.unsqueeze(0).repeat(A, 1)         # (A,R)
    # AFTER  (use azimuth relative to the first beam)
    # assumes azimuth is measured relative to +x axis
    az_grid = az.unsqueeze(1).repeat(1, R)      # (A,R)
    x_i = r_grid * radar_res * torch.cos(az_grid)
    y_i = r_grid * radar_res * torch.sin(az_grid)

    # ------------------------------------------------------------------
    # 4.  Build a 3 × 3 homogeneous transform Tᵢ  for *every* beam
    #         Tᵢ  maps beam-i coordinates → first-beam coordinates
    #         P₀ = Tᵢ · Pᵢ
    # ------------------------------------------------------------------
    τ          = t_beam.view(A, 1)                         # (A,1)
    dx, dy     = (v_xy[0] * τ, v_xy[1] * τ)                # (A,1)
    dθ         = w_z * τ                                   # (A,1)
    cosθ, sinθ = torch.cos(dθ), torch.sin(dθ)              # (A,1)

    # Assemble Tᵢ as [R  p; 0 1]  where R = R(+ωτ),  p = v τ
    T = torch.zeros(A, 3, 3, device=device)
    T[:, 0, 0] =  cosθ.squeeze()
    T[:, 0, 1] = -sinθ.squeeze()
    T[:, 0, 2] =  dx.squeeze()
    T[:, 1, 0] =  sinθ.squeeze()
    T[:, 1, 1] =  cosθ.squeeze()
    T[:, 1, 2] =  dy.squeeze()
    T[:, 2, 2] =  1.0

    # ------------------------------------------------------------------
    # 5.  Apply the transforms in batch
    # ------------------------------------------------------------------
    ones   = torch.ones_like(x_i)
    pts_i  = torch.stack((x_i, y_i, ones), dim=-1)         # (A,R,3)
    pts_i  = pts_i.unsqueeze(-1)                           # (A,R,3,1)

    # Broadcast multiply: (A,1,3,3) × (A,R,3,1) → (A,R,3,1)
    T_exp  = T.unsqueeze(1)                                # (A,1,3,3)
    pts_0  = torch.matmul(T_exp, pts_i)                    # (A,R,3,1)

    x_u = pts_0[:, :, 0, 0]
    y_u = pts_0[:, :, 1, 0]

    # ------------------------------------------------------------------
    # 6.  Convert back to polar indices (nearest-neighbour)
    # ------------------------------------------------------------------
    r_idx = torch.round(torch.sqrt(x_u ** 2 + y_u ** 2) / radar_res).long()
    θ     = torch.atan2(y_u, x_u);   θ += (θ < 0) * 2 * torch.pi
    az_step = (az[-1] - az[0]) / A
    az_idx = torch.round((θ - az[0]) / az_step).long()

    # clamp inside bounds
    az_idx = torch.clamp(az_idx, 0, A - 1)
    r_idx  = torch.clamp(r_idx,  0, R - 1)

    # ------------------------------------------------------------------
    # 7.  Scatter-copy values into the undistorted canvas
    # ------------------------------------------------------------------
    undistorted = torch.zeros_like(polar_intensity)
    undistorted[az_idx.view(-1), r_idx.view(-1)] = polar_intensity.view(-1)
    # print(undistorted-polar_intensity)
    return undistorted

# ---------------------------------------------------------------------
#  Helpers -------------------------------------------------------------
# ---------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalise_to_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return (img * 255).astype(np.uint8)


def dummy_test(device=torch.device("cpu"),
               out_dir: str = "dummy_out") -> None:
    """
    Fabricate a toy scan, undistort it, save PNGs of
    the RAW (still distorted) and UND (undistorted) Cartesian images.
    """
    # ------------------------------------------------- synthetic scan
    A, R        = 400, 1712
    radar_res   = 0.040308
    azimuths    = np.linspace(0.0, 2*np.pi*(A-1)/A, A).astype(np.float32)
    print(f"azimuths are {azimuths} with size {azimuths.size}")
    az_t        = np.linspace(0.0, 0.1, A).astype(np.float32)   # dt = 0.1 s
    print(f"az_t is {az_t} with size {az_t.size}")

    polar_raw   = np.zeros((A, R), dtype=np.float32)
    polar_raw[:, 150] = 1.0                                     # bright ring

    # ------------------------------------------------- fake motion (robot frame)
    fwd, yaw_deg = 150*radar_res*2, 0.0
    yaw_rad      = np.deg2rad(yaw_deg)
    T_v_w = np.eye(4, dtype=np.float32)
    T_v_w[0, 3] = fwd
    T_v_w[:2, :2] = [[ np.cos(yaw_rad), -np.sin(yaw_rad)],
                     [ np.sin(yaw_rad),  np.cos(yaw_rad)]]

    # ------------------------------------------------- call undistortion
    polar_und = motion_undistortion(
        polar_raw, azimuths, az_t, T_v_w, dt=0.1, device=device
    ).cpu().numpy()
    
    # ------------------------------------------------- Cartesian renderings
    cart_raw = radar_polar_to_cartesian(
        polar_raw.astype(np.float32), azimuths, radar_resolution=radar_res
    )
    cart_und = radar_polar_to_cartesian(
        polar_und.astype(np.float32), azimuths, radar_resolution=radar_res
    )
    nonzero_y, nonzero_x = np.nonzero(cart_und)  # row, col indices
    idx_pairs = list(zip(nonzero_y, nonzero_x))  # (row, col) = (y, x)

    print(f"Found {len(idx_pairs)} non-zero pixels:")
    for yx in idx_pairs:
        print(yx)

    # ------------------------------------------------- save PNGs
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "dummy_raw.png"),
                normalise_to_u8(cart_raw))
    cv2.imwrite(os.path.join(out_dir, "dummy_und.png"),
                normalise_to_u8(cart_und))

    # ------------------------------------------------- simple numeric check
    mad = np.mean(np.abs(polar_raw - polar_und))
    print(f"mean|Δ| between RAW and UND = {mad:.4f}")
    print("PNG files written to", out_dir)


# ---------------------------------------------------------------------
#  MAIN ----------------------------------------------------------------
# ---------------------------------------------------------------------
def main(parent_folder: str = "/home/sahakhsh/Documents/vtr3_testing") -> None:
    """
    * Reads the *teach* run produced by your VTR scripts,
    * undistorts every scan with `motion_undistortion`,
    * and writes side-by-side PNGs to

          …/scripts/direct/grassy_t2_r3/local_map_vtr/<timestamp>.png
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------- reproduce your folder logic --------------
    config_direct = load_config(
        os.path.join(parent_folder, "scripts/direct/direct_configs/direct_config_hshmat.yaml")
    )
    result_folder   = config_direct.get("output")
    out_path_folder = os.path.join(result_folder, "grassy_t2_r3")
    teach_folder    = os.path.join(out_path_folder, "teach")

    local_map_path = (
        "/home/sahakhsh/Documents/vtr3_testing/scripts/direct/grassy_t2_r3/motion_distortion_fix/"
    )
    os.makedirs(local_map_path, exist_ok=True)

    # ----------------------- load teach data --------------------------
    teach = np.load(os.path.join(teach_folder, "teach.npz"), allow_pickle=True)
    polar_imgs      = teach["teach_polar_imgs"]          # (N,400,1712)
    az_angles_all   = teach["teach_azimuth_angles"]      # (N,400,1)
    az_stamps_all   = teach["teach_azimuth_timestamps"]  # (N,400,1)
    vtx_stamps      = teach["teach_vertex_timestamps"]   # (N,1)
    vtx_T_world     = teach["teach_vertex_transforms"]   # (N,4,4)

    N = polar_imgs.shape[0]
    if N < 2:
        raise RuntimeError("Need at least two radar frames for motion estimates.")

    print(f"Processing {N-1} scans …")

    # ----------------------- iterate ----------------------------------
    for k in range(N - 1):
        raw     = torch.tensor(polar_imgs[k]).to(device).float()
        az      = az_angles_all[k].squeeze()
        az_t    = az_stamps_all[k].squeeze()
        dt      = float(vtx_stamps[k + 1, 0] - vtx_stamps[k, 0])

        T_curr = np.asarray(list(vtx_T_world[k][0].values())[0].matrix())
        T_next = np.asarray(list(vtx_T_world[k + 1][0].values())[0].matrix())
        T_v_w   = np.linalg.inv(T_curr) @ T_next              # next←curr

        und     = motion_undistortion(
            raw, az, az_t, T_v_w, dt, device=device
        ).cpu().numpy()

        # 1.  Run the same preprocessing on the raw scan
        prep_tensor = preprocessing_polar_image(raw, device)          # (400,1712) torch
        prep_np     = prep_tensor.cpu().numpy().astype(np.float32)

        # `und` is already the motion-undistorted *and* preprocessed tensor → NumPy
        und_np = und.astype(np.float32)

        # 2.  Polar → Cartesian    (640 × 640, CIR-204 defaults)
        prep_cart = radar_polar_to_cartesian(
            prep_np,
            az.astype(np.float32),
            radar_resolution=0.040308
        )
        und_cart  = radar_polar_to_cartesian(
            und_np,
            az.astype(np.float32),
            radar_resolution=0.040308
        )

        prep_u8 = normalise_to_u8(prep_cart)   # “raw” (still motion-distorted)
        und_u8  = normalise_to_u8(und_cart)    # motion-undistorted

        mid_scan_timestamp = vtx_stamps[k][0]

        cv2.imwrite(
            os.path.join(local_map_path, f"{mid_scan_timestamp}_raw.png"),
            prep_u8,
        )
        cv2.imwrite(
            os.path.join(local_map_path, f"{mid_scan_timestamp}_und.png"),
            und_u8,
        )
        print(f"saved  {mid_scan_timestamp}.png")

    print("✅  all scans processed and saved.")


# ---------------------------------------------------------------------
# Call just once when you run the script directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    dummy_test()

