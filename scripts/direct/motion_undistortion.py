## this file aims to undistort the radar images using the current odometry estimates
## as well as all the other preprocessing steps

import os
import cv2
import numpy as np
import torch
import torchvision
import yaml
from pylgmath import Transformation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import radar_polar_to_cartesian # this is a custom function to convert polar radar images to cartesian

T_radar_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.025],
                                                 [0.000, -1.000 , 0.000, -0.002],
                                                 [0.000 ,0.000, -1.000 , 1.032],
                                                 [0.000 , 0.000 ,0.000, 1.000]]))



def motion_undistortion(
    polar_image: np.ndarray,
    azimuth_angles: np.ndarray,
    azimuth_timestamps: np.ndarray,
    T_v_w: np.ndarray,
    dt: float,
    device: torch.device,
    radar_res: float = 0.040308, #m
    t_idx: int = 0
) -> torch.Tensor:
    """
    Undistort a polar radar scan by **applying a 3 × 3 homogeneous
    transform for every beam**.  Nearest-neighbour remapping only.
    Output frame = radar pose at the first beam of the scan.
    """
    if not isinstance(polar_image, torch.Tensor):
        polar_image = torch.from_numpy(polar_image).to(device=device, dtype=torch.float64)

    T_sr_np   = T_radar_robot.matrix()                  # robot ➔ radar
    T_sr_inv  = np.linalg.inv(T_sr_np)
    # ------------------------------------------------------------------
    # 0.  Pre-processing
    # ------------------------------------------------------------------
    # polar_intensity = preprocessing_polar_image(polar_image, device)
    # polar_intensity = torch.from_numpy(polar_image).float().to(device)
    # polar_intensity = preprocessing_polar_image(polar_intensity, device)
    # print(f"polar_intensity shape: {polar_intensity.shape}")

    A, R = polar_image.shape      # usually 400 × 1712

    # ------------------------------------------------------------------
    # 1.  Constant planar twist  (vx, vy, ω)  from the inter-scan Δpose
    # ------------------------------------------------------------------
    Tw_radar = torch.as_tensor(
        T_sr_np @ T_v_w @ T_sr_inv, device=device, dtype=torch.float64
    )               
    # print("Tw_radar =", Tw_radar)

    v_xy = Tw_radar[:2, 3] / dt                  # m s⁻¹ in radar axes
    # v_xy = torch.tensor([0., 5000.], dtype=torch.float64)
    yaw  = torch.atan2(Tw_radar[1, 0], Tw_radar[0, 0])
    w_z  = yaw / dt
    # w_z = 0.
    print(f"v_xy = {v_xy}")
    print(f"w_z = {w_z}")
    print(f"dt = {dt}")

    # ------------------------------------------------------------------
    # 2.  Acquisition-time metadata for each beam
    # ------------------------------------------------------------------
    az = torch.as_tensor(azimuth_angles.squeeze(), device=device, dtype=torch.float64)
    t_beam = torch.as_tensor(azimuth_timestamps.squeeze(), device=device, dtype=torch.float64) / 1e9
    # print("t_beam is: ", t_beam.squeeze())
    t_beam  = t_beam - t_beam[t_idx]                           # τᵢ per beam (A,)

    # ------------------------------------------------------------------
    # 3.  Cartesian coordinates for every pixel in the *measured* scan
    # ------------------------------------------------------------------
    range_bins = torch.arange(R, device=device, dtype=torch.float64)
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
    # print("t is: ", τ)
    dx, dy     = (v_xy[0] * τ, v_xy[1] * τ)                # (A,1)
    dθ         =   w_z * τ                                   # (A,1)
    cosθ, sinθ = torch.cos(dθ), torch.sin(dθ)              # (A,1)
    # print("t_idx =", t_idx)
    # print("t_beam[t_idx] =", t_beam[t_idx])
    # print("t_beam[100]   =", t_beam[100])
    # print("τ[100]        =", τ[100])


    # Assemble Tᵢ as [R  p; 0 1]  where R = R(+ωτ),  p = v τ
    T = torch.zeros(A, 3, 3, device=device, dtype=torch.float64)
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
    ones = torch.ones_like(x_i, dtype=torch.float64)
    pts_i  = torch.stack((x_i, y_i, ones), dim=-1)         # (A,R,3)
    pts_i  = pts_i.unsqueeze(-1)                           # (A,R,3,1)
    # print("sample pt is: ", pts_i[100,100])

    # Broadcast multiply: (A,1,3,3) × (A,R,3,1) → (A,R,3,1)
    # print(f"v_xy = {v_xy.cpu().numpy()}")
    print(f"max τ = {torch.max(τ).item()}  -->  max dx = {torch.max(abs(dx)).item()}")
    # print(f"T[100] = \n{T[100]}")

    T_exp  = T.unsqueeze(1)                                # (A,1,3,3)
    pts_0  = torch.matmul(T_exp, pts_i)                    # (A,R,3,1)
    # print("sample pt after matmul is: ", pts_0[100,100])

    x_u = pts_0[:, :, 0, 0]
    y_u = pts_0[:, :, 1, 0]


    # Important note: stay in cartesian sapce as converting to polar indices introduces rounding errors (maybe just return the )

    # ------------------------------------------------------------------
    # 6.  Convert back to polar indices (nearest-neighbour)
    # ------------------------------------------------------------------
    r_idx = torch.round(torch.sqrt(x_u ** 2 + y_u ** 2) / radar_res).long()
    θ     = torch.atan2(y_u, x_u);   θ += (θ < 0) * 2 * torch.pi
    az_step = (az[-1] - az[0]) / (A - 1)
    az_idx = torch.round((θ - az[0]) / az_step).long()
    # print("az_idx before clamping", az_idx)
    # print("r_idx before clamping", r_idx)

    # clamp inside bounds
    az_idx = torch.clamp(az_idx, 0, A - 1)
    # print("az_idx is: ", az_idx)
    r_idx  = torch.clamp(r_idx,  0, R - 1)
    # print("r_idx is: ", r_idx)
    # ------------------------------------------------------------------
    # 7.  Scatter-copy values into the undistorted canvas
    # ------------------------------------------------------------------
    undistorted = torch.zeros_like(polar_image)
    counts = torch.zeros_like(polar_image)

    flat_idx_az = az_idx.view(-1)
    flat_idx_r = r_idx.view(-1)
    flat_vals = polar_image.view(-1)

    undistorted.index_put_((flat_idx_az, flat_idx_r), flat_vals, accumulate=True)
    counts.index_put_((flat_idx_az, flat_idx_r), torch.ones_like(flat_vals), accumulate=True)

    # Avoid division by zero
    counts[counts == 0] = 1
    undistorted = undistorted / counts

    # print(undistorted-polar_intensity)

    return polar_image, undistorted


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalise_to_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return (img * 255).astype(np.uint8)


# def dummy_test(device=torch.device("cpu"),
#                out_dir: str = "dummy_out") -> None:
#     """
#     Fabricate a toy scan, undistort it, save PNGs of
#     the RAW (still distorted) and UND (undistorted) Cartesian images.
#     """
#     # ------------------------------------------------- synthetic scan
#     A, R        = 400, 1712
#     radar_res   = 0.040308
#     azimuths    = np.linspace(0.0, 2*np.pi*(A-1)/A, A).astype(np.float64)
#     print(f"azimuths are {azimuths} with size {azimuths.size}")
#     az_t        = np.linspace(0.0, 0.1, A).astype(np.float64)   # dt = 0.1 s
#     print(f"az_t is {az_t} with size {az_t.size}")

#     polar_raw   = np.zeros((A, R), dtype=np.float64)
#     polar_raw[:, 150] = 1.0                                     # bright ring

#     # ------------------------------------------------- fake motion (robot frame)
#     fwd, yaw_deg = 150*radar_res*2, 0.0
#     yaw_rad      = np.deg2rad(yaw_deg)
#     T_v_w = np.eye(4, dtype=np.float64)
#     T_v_w[0, 3] = fwd
#     T_v_w[:2, :2] = [[ np.cos(yaw_rad), -np.sin(yaw_rad)],
#                      [ np.sin(yaw_rad),  np.cos(yaw_rad)]]

#     # ------------------------------------------------- call undistortion
#     polar_und = motion_undistortion(
#         polar_raw, azimuths, az_t, T_v_w, dt=0.1, device=device
#     ).cpu().numpy()
    
#     # ------------------------------------------------- Cartesian renderings
#     cart_raw = radar_polar_to_cartesian(
#         polar_raw.astype(np.float64), azimuths, radar_resolution=radar_res
#     )
#     cart_und = radar_polar_to_cartesian(
#         polar_und.astype(np.float64), azimuths, radar_resolution=radar_res
#     )

#     nonzero_y, nonzero_x = np.nonzero(cart_und)  # row, col indices
#     idx_pairs = list(zip(nonzero_y, nonzero_x))  # (row, col) = (y, x)

#     print(f"Found {len(idx_pairs)} non-zero pixels:")
#     for yx in idx_pairs:
#         print(yx)

#     # ------------------------------------------------- save PNGs
#     os.makedirs(out_dir, exist_ok=True)
#     cv2.imwrite(os.path.join(out_dir, "dummy_raw.png"),
#                 normalise_to_u8(cart_raw))
#     cv2.imwrite(os.path.join(out_dir, "dummy_und.png"),
#                 normalise_to_u8(cart_und))

#     # ------------------------------------------------- simple numeric check
#     mad = np.mean(np.abs(polar_raw - polar_und))
#     print(f"mean|Δ| between RAW and UND = {mad:.4f}")
#     print("PNG files written to", out_dir)


# def main(parent_folder: str = "/home/sahakhsh/Documents/vtr3_testing") -> None:
#     """
#     * Reads the *teach* run produced by your VTR scripts,
#     * undistorts every scan with `motion_undistortion`,
#     * and writes side-by-side PNGs to

#           …/scripts/direct/grassy_t2_r3/local_map_vtr/<timestamp>.png
#     """
#     device = torch.device("cpu")

#     # ----------------------- reproduce your folder logic --------------
#     config_direct = load_config(
#         os.path.join(parent_folder, "scripts/direct/direct_configs/direct_config_hshmat.yaml")
#     )
#     result_folder = config_direct.get("output")
#     out_path_folder = os.path.join(result_folder, "grassy_t2_r3")
#     teach_folder = os.path.join(out_path_folder, "teach")

#     local_map_path = (
#         "/home/sahakhsh/Documents/vtr3_testing/scripts/direct/grassy_t2_r3/motion_distortion_fix/"
#     )
#     os.makedirs(local_map_path, exist_ok=True)

#     # ----------------------- load teach data --------------------------
#     teach = np.load(os.path.join(teach_folder, "teach.npz"), allow_pickle=True)
#     polar_imgs = teach["teach_polar_imgs"]          # (N,400,1712)
#     az_angles_all = teach["teach_azimuth_angles"]      # (N,400,1)
#     az_stamps_all = teach["teach_azimuth_timestamps"]  # (N,400,1)
#     vtx_stamps = teach["teach_vertex_timestamps"]   # (N,1)
#     vtx_T_world = teach["teach_vertex_transforms"]   # (N,4,4)

#     N = polar_imgs.shape[0]
#     if N < 2:
#         raise RuntimeError("Need at least two radar frames for motion estimates.")

#     print(f"Processing {N-1} scans …")

#     # ----------------------- iterate ----------------------------------
#     for k in range(N - 1):
#         polar_image = torch.tensor(polar_imgs[k], dtype=torch.float64, device=device)
#         azimuth_angles = az_angles_all[k].squeeze()
#         azimuth_timestamps = az_stamps_all[k].squeeze()
#         dt = float(vtx_stamps[k + 1, 0] - vtx_stamps[k, 0])

#         T_curr = np.asarray(list(vtx_T_world[k][0].values())[0].matrix())
#         T_next = np.asarray(list(vtx_T_world[k + 1][0].values())[0].matrix())
#         T_v_w   = T_curr @ np.linalg.inv(T_next)              # next←curr
#         print(T_v_w)
#         # input()
#         # call the motion undistortion function
#         polar_intensity, undistorted = motion_undistortion(
#             polar_image, azimuth_angles, azimuth_timestamps, T_v_w, dt, torch.device('cpu')
#         )
#         print(polar_intensity)
#         # input()
#         print(undistorted)

#         # print("undistorted shape:", undistorted.shape)

#         cart_image = radar_polar_to_cartesian(
#             polar_intensity.numpy().astype(np.float64),
#             azimuth_angles.astype(np.float64),
#             radar_resolution=0.040308
#         )

#         cart_undistorted = radar_polar_to_cartesian(
#             undistorted.numpy().astype(np.float64),
#             azimuth_angles.astype(np.float64),
#             radar_resolution=0.040308
#         )

#         print("cart_image shape:", cart_image.shape)
#         print("cart_undistorted shape:", cart_undistorted.shape)

#         # Compute absolute difference
#         diff = cv2.absdiff(cart_image, cart_undistorted)
#         # cv2.imshow('Difference', diff)
#         # cv2.waitKey(0)


#         # # Count non-zero pixels
#         # num_different_pixels = np.count_nonzero(thresh)
#         # print(f"Number of different pixels: {num_different_pixels}")

#         # # i want to display two images side by side
#         # # # plt.ion()
#         # plt.figure(figsize=(12, 6))
#         # plt.subplot(1, 3, 1)
#         # plt.imshow(cart_image, cmap='gray')
#         # plt.title("Polar Intensity")
#         # plt.axis('off')
#         # plt.subplot(1, 3, 2)
#         # plt.imshow(cart_undistorted, cmap='gray')
#         # plt.title("Undistorted Image")
#         # plt.axis('off')
#         # plt.subplot(1, 3, 3)
#         # plt.imshow(diff, cmap='gray')
#         # plt.title("Difference Image")
#         # plt.axis('off')
#         # plt.tight_layout()
#         # # interactive mode
#         # plt.show()

#     print("✅  all scans processed and saved.")


if __name__ == "__main__":
    # here is the game plan:

    # I load teach.npz and then save undistorted images 
    # lets see if this script does what I want it to do
    cart_imgs          = []      # raw polar→cartesian
    cart_undist_imgs   = []      # undistorted polar→cartesian
    diff_imgs          = []      # pixel-wise difference

    parent_folder = "/home/samqiao/ASRL/vtr3_testing"

    config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_sam.yaml'))
    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    SAVE = False
    print("SAVE:",SAVE)
    PLOT = db_bool.get('PLOT')
    DEBUG = db_bool.get('DEBUG')

    result_folder = config.get('output')

    # change here
    out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
    if not os.path.exists(out_path_folder):
        os.makedirs(out_path_folder)
        print(f"Folder '{out_path_folder}' created.")
    else:
        print(f"Folder '{out_path_folder}' already exists.")    

    # print(result_folder)

    sequence = "grassy_t2_r3"

    sequence_path = os.path.join(result_folder, sequence)
    if not os.path.exists(sequence_path):
        print("ERROR: No sequence found in " + sequence_path)
        exit(0)

    TEACH_FOLDER = os.path.join(sequence_path, "teach")
    REPEAT_FOLDER = os.path.join(sequence_path, "repeat")
    RESULT_FOLDER = os.path.join(sequence_path, "direct")

    if not os.path.exists(TEACH_FOLDER):
        raise FileNotFoundError(f"Teach folder {TEACH_FOLDER} does not exist.")
    if not os.path.exists(REPEAT_FOLDER):
        raise FileNotFoundError(f"Repeat folder {REPEAT_FOLDER} does not exist.")
    if not os.path.exists(RESULT_FOLDER):
        raise FileNotFoundError(f"Result folder {RESULT_FOLDER} does not exist.")

    teach_df = np.load(os.path.join(TEACH_FOLDER, "teach.npz"),allow_pickle=True)
    repeat_df = np.load(os.path.join(REPEAT_FOLDER, "repeat.npz"),allow_pickle=True)

    # in the teach
    # 1. (932,400,1712) images
    teach_polar_imgs = teach_df['teach_polar_imgs']
    # 2. (932,400, 1) azimuth angles
    teach_azimuth_angles = teach_df['teach_azimuth_angles']
    # 3. (932,400, 1) azimuth timestamps
    teach_azimuth_timestamps = teach_df['teach_azimuth_timestamps']
    # 4. (932,1) vertex timestamps
    teach_vertex_timestamps = teach_df['teach_vertex_timestamps']
    # 5. Pose at each vertex: (932,4,4)
    teach_vertex_transforms = teach_df['teach_vertex_transforms']
    # 6. teach vertext time
    teach_times = teach_df['teach_times']
    # 7. teach vertex ids
    teach_vertex_ids = teach_df['teach_vertex_ids']

    teach_polar_imgs_undistorted = np.zeros_like(teach_polar_imgs)

    # do the same for repeat
    # in the repeat
    repeat_times = repeat_df['repeat_times']
    repeat_polar_imgs = repeat_df['repeat_polar_imgs']
    repeat_azimuth_angles = repeat_df['repeat_azimuth_angles']
    repeat_azimuth_timestamps = repeat_df['repeat_azimuth_timestamps']
    repeat_vertex_timestamps = repeat_df['repeat_vertex_timestamps']
    repeat_edge_transforms = repeat_df['repeat_edge_transforms']
    vtr_estimated_ptr = repeat_df['dist']
    repeat_vertex_ids = repeat_df['repeat_vertex_ids']


    repeat_polar_imgs_undistorted = np.zeros_like(repeat_polar_imgs)

    with torch.no_grad():
        print("Starting motion undistortion for teach vertices...")
        for teach_vertex_idx in range(0,teach_times.shape[0]): # there is an edge case for the last vertex but we can handle that later
            print("-------------processing teach vertex idx:",teach_vertex_idx, "-----------------")
            # get the pose at the teach vertex
            teach_vertex_id_k = teach_vertex_ids[teach_vertex_idx]
            
            # find the next vertex witha different vertex id
            for next_idx in range(teach_vertex_idx+1, teach_times.shape[0]):
                if teach_vertex_ids[next_idx] != teach_vertex_id_k:
                    teach_vertex_id_next = teach_vertex_ids[next_idx]
                    teach_vertex_idx_next = next_idx
                    print("Next vertex id found:", teach_vertex_id_next)
                    print("Next vertex index found:", teach_vertex_idx_next)
                    break


            teach_vertex_time_k = teach_times[teach_vertex_idx]
            teach_vertex_time_k_p_1 = teach_vertex_timestamps[teach_vertex_idx_next]

            dt = teach_vertex_time_k_p_1[0] - teach_vertex_time_k[0]

            T_teach_world_current = teach_vertex_transforms[teach_vertex_idx][0][teach_vertex_time_k[0]]
            # T_gps_world_teach = T_novatel_robot @ T_teach_world

            T_teach_world_next = teach_vertex_transforms[teach_vertex_idx_next][0][teach_vertex_time_k_p_1[0]]

            T_increment = T_teach_world_current.inverse() @ T_teach_world_next

            # print("T_increment:",T_increment.matrix())
            # print("dt :",dt)


            polar_image = teach_polar_imgs[teach_vertex_idx]
            # polar_image[:, :] = 0.
            # polar_image[:, 150:200] = 100.0
            azimuth_angles = teach_azimuth_angles[teach_vertex_idx].squeeze()
            print(len(azimuth_angles))
            azimuth_timestamps = teach_azimuth_timestamps[teach_vertex_idx].squeeze()


            # print("polar_image shape:", polar_image.shape)
            # print("azimuth_angles shape:", azimuth_angles.shape)
            # print("azimuth_timestamps shape:", azimuth_timestamps.shape)

            # call the motion undistortion function
            polar_intensity, undistorted = motion_undistortion(
                polar_image, azimuth_angles, azimuth_timestamps, T_increment.matrix(), dt, device=torch.device("cpu"), t_idx=199
            )

            # print("undistorted shape:", undistorted.shape)

            cart_image = radar_polar_to_cartesian(
                polar_intensity.numpy().astype(np.float64),
                azimuth_angles.astype(np.float64),
                radar_resolution=0.040308
            )

            cart_undistorted = radar_polar_to_cartesian(
                undistorted.numpy().astype(np.float64),
                azimuth_angles.astype(np.float64),
                radar_resolution=0.040308
            )

            # print("cart_image shape:", cart_image.shape)
            # print("cart_undistorted shape:", cart_undistorted.shape)

            # Compute absolute difference
            diff = cv2.absdiff(cart_image, cart_undistorted)
            # cv2.imshow('Difference', diff)
            # cv2.waitKey(0)

            # # Count non-zero pixels
            # num_different_pixels = np.count_nonzero(thresh)
            # print(f"Number of different pixels: {num_different_pixels}")

            # # i want to display two images side by side
            # # plt.ion()
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 3, 1)
            # plt.imshow(cart_image, cmap='gray')
            # plt.title("Polar Intensity")
            # plt.axis('off')
            # plt.subplot(1, 3, 2)
            # plt.imshow(cart_undistorted, cmap='gray')
            # plt.title("Undistorted Image")
            # plt.axis('off')
            # plt.subplot(1, 3, 3)
            # plt.imshow(diff, cmap='gray')
            # plt.title("Difference Image")
            # plt.axis('off')
            # plt.tight_layout()
            # # interactive mode
            # plt.show()
            cart_imgs.append(cart_image)
            cart_undist_imgs.append(cart_undistorted)
            diff_imgs.append(diff)

            # lets handle the last vertex case
            if teach_vertex_idx == teach_times.shape[0] - 1:
            # I would just use the previous velocity estimates

                print("Last vertex, using the previous velocity estimates")
                # find the next vertex witha different vertex id
                for prev_idx in range(teach_vertex_idx, 0,-1):
                    if teach_vertex_ids[prev_idx] != teach_vertex_id_k:
                        teach_vertex_id_prev = teach_vertex_ids[prev_idx]
                        teach_vertex_idx_prev = prev_idx
                        print("Previous vertex id found:", teach_vertex_id_prev)
                        print("Previous vertex index found:", teach_vertex_idx_prev)
                        break

                teach_vertex_time_k_p_1 = teach_times[teach_vertex_idx_prev]

                T_teach_world_current = teach_vertex_transforms[teach_vertex_idx][0][teach_vertex_time_k[0]]

                T_teach_world_prev = teach_vertex_transforms[teach_vertex_idx_prev][0][teach_vertex_time_k_p_1[0]]

                T_increment = T_teach_world_prev.inverse() @ T_teach_world_current

                dt = teach_vertex_time_k[0] - teach_vertex_time_k_p_1[0]


                polar_intensity, undistorted = motion_undistortion(
                    polar_image, azimuth_angles, azimuth_timestamps, T_increment.matrix(), dt, device=torch.device("cpu")
                )

                teach_vertex_id_k = teach_vertex_ids[teach_vertex_idx]

            # print("polar image:", polar_image)
            # print("undistorted plar image", undistorted)
            # print("undistorted shape:", undistorted.shape)
            # save the undistorted image in the teach_polar_imgs_undistorted array
    
            teach_polar_imgs_undistorted[teach_vertex_idx] = undistorted.detach().cpu().clone().numpy().reshape(400, 1712)



            # # lets plot them 2 images side by side
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(undistorted.detach().numpy().reshape(400, 1712), cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(teach_polar_imgs_undistorted[teach_vertex_idx], cmap='gray')
            # plt.title(f"Teach Vertex {teach_vertex_idx} - ID: {teach_vertex_id_k}")
            # # plt.imshow(teach_polar_imgs_undistorted[teach_vertex_idx], cmap='gray')
            # plt.show()


            # break
        
    print("Starting motion undistortion for repeat vertices...")
    for repeat_vertex_idx in range(0,repeat_times.shape[0]): # there is
        print("-------------processing repeat vertex idx:",repeat_vertex_idx, "-----------------")
        # get the pose at the repeat vertex
        repeat_vertex_id_k = repeat_vertex_ids[repeat_vertex_idx]
        # find the next vertex witha different vertex id
        for next_idx in range(repeat_vertex_idx+1, repeat_times.shape[0]):
            if repeat_vertex_ids[next_idx] != repeat_vertex_id_k:
                repeat_vertex_id_next = repeat_vertex_ids[next_idx]
                repeat_vertex_idx_next = next_idx
                print("Next vertex id found:", repeat_vertex_id_next)
                print("Next vertex index found:", repeat_vertex_idx_next)
                break
        repeat_vertex_time_k = repeat_times[repeat_vertex_idx]
        repeat_vertex_time_k_p_1 = repeat_vertex_timestamps[repeat_vertex_idx_next]
        dt = repeat_vertex_time_k_p_1[0] - repeat_vertex_time_k[0]
        T_repeat_world_current = repeat_edge_transforms[repeat_vertex_idx][0][repeat_vertex_time_k[0]]
        # T_gps_world_repeat = T_novatel_robot @ T_repeat_world
        T_repeat_world_next = repeat_edge_transforms[repeat_vertex_idx_next][0][repeat_vertex_time_k_p_1[0]]
        T_increment = T_repeat_world_current.inverse() @ T_repeat_world_next
        # print("T_increment:",T_increment.matrix())
        # print("dt :",dt)
        polar_image = repeat_polar_imgs[repeat_vertex_idx]
        # polar_image[:, :] = 0.
        # polar_image[:, 150:200] = 100.0
        azimuth_angles = repeat_azimuth_angles[repeat_vertex_idx].squeeze()
        print(len(azimuth_angles))
        azimuth_timestamps = repeat_azimuth_timestamps[repeat_vertex_idx].squeeze()
        # print("polar_image shape:", polar_image.shape)
        # print("azimuth_angles shape:", azimuth_angles.shape)
        # print("azimuth_timestamps shape:", azimuth_timestamps.shape)
        # call the motion undistortion function
        polar_intensity, undistorted = motion_undistortion(
            polar_image, azimuth_angles, azimuth_timestamps, T_increment.matrix(), dt, device=torch.device("cpu"), t_idx=199
        )

        # # print("undistorted shape:", undistorted.shape)
        # cart_image = radar_polar_to_cartesian(
        #     polar_intensity.numpy().astype(np.float64),
        #     azimuth_angles.astype(np.float64),
        #     radar_resolution=0.040308
        # )

        # cart_undistorted = radar_polar_to_cartesian(
        #     undistorted.numpy().astype(np.float64),
        #     azimuth_angles.astype(np.float64),
        #     radar_resolution=0.040308
        # )


        # lets handle the last vertex case
        if repeat_vertex_idx == repeat_times.shape[0] - 1:
            # I would just use the previous velocity estimates
            print("Last vertex, using the previous velocity estimates")
            # find the next vertex witha different vertex id
            for prev_idx in range(repeat_vertex_idx, 0,-1):
                if repeat_vertex_ids[prev_idx] != repeat_vertex_id_k:
                    repeat_vertex_id_prev = repeat_vertex_ids[prev_idx]
                    repeat_vertex_idx_prev = prev_idx
                    print("Previous vertex id found:", repeat_vertex_id_prev)
                    print("Previous vertex index found:", repeat_vertex_idx_prev)
                    break

            repeat_vertex_time_k_p_1 = repeat_times[repeat_vertex_idx_prev]

            T_repeat_world_current = repeat_edge_transforms[repeat_vertex_idx][0][repeat_vertex_time_k[0]]
            T_repeat_world_prev = repeat_edge_transforms[repeat_vertex_idx_prev][0][repeat_vertex_time_k_p_1[0]]

            T_increment = T_repeat_world_prev.inverse() @ T_repeat_world_current
            dt = repeat_vertex_time_k[0] - repeat_vertex_time_k_p_1[0]
            polar_intensity, undistorted = motion_undistortion(
                polar_image, azimuth_angles, azimuth_timestamps, T_increment.matrix(), dt, device=torch.device("cpu")
            )
            repeat_vertex_id_k = repeat_vertex_ids[repeat_vertex_idx]
        # print("polar image:", polar_image)
        # print("undistorted plar image", undistorted)
        # print("undistorted shape:", undistorted.shape)
        # save the undistorted image in the teach_polar_imgs_undistorted array
        repeat_polar_imgs_undistorted[repeat_vertex_idx] = undistorted.detach().cpu().clone().numpy().reshape(400, 1712)

        # # lets plot them 2 images side by side
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(undistorted.detach().numpy().reshape(400, 1712
        # ), cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(repeat_polar_imgs_undistorted[repeat_vertex_idx], cmap
        # 'gray')
        # plt.title(f"Repeat Vertex {repeat_vertex_idx} - ID: {repeat_vertex
        # k}")
        # # plt.imshow(teach_polar_imgs_undistorted[teach_vertex_idx],
        # cmap='gray')
        # plt.show()

    # print("cart_imgs:", cart_imgs)
    # print("cart_undist_imgs:", cart_undist_imgs)
    # print("diff_imgs:", diff_imgs)
    # print("teach_polar_imgs_undistorted:", teach_polar_imgs_undistorted)
    print("All scans processed and saved.")
    # ------------------------------------------------------------------
        # break

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    titles = ["Polar Intensity", "Undistorted", "Difference"]

    # draw the first frame to get the AxesImage handles
    ims = [
        axes[i].imshow(cart_imgs[0] if i == 0 else
                    cart_undist_imgs[0] if i == 1 else
                    diff_imgs[0],
                    cmap='gray',
                    animated=True)
        for i in range(3)
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")

    def update(frame):
        ims[0].set_array(cart_imgs[frame])
        ims[1].set_array(cart_undist_imgs[frame])
        ims[2].set_array(diff_imgs[frame])
        return ims   # blitting needs a list/tuple of artists

    ani = animation.FuncAnimation(fig,
                                update,
                                frames=len(cart_imgs),
                                interval=50,      # ms between frames
                                blit=True)
 
    from matplotlib.animation import writers
    Writer = writers["ffmpeg"]
    writer = Writer(
        fps=20,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",          # 4:2:0 chroma subsampling (mandatory!)
            "-profile:v", "baseline",       # use the simplest H.264 profile
            "-level", "3.0",                # ensures compatibility with older players
            "-movflags", "+faststart"       # enables web/streaming playback
        ]
    )

    ani.save(os.path.join(out_path_folder, "undistortion.mp4"),
            writer=writer, dpi=100)
    # SAVE TEACH CONTENT IN THE TEACH FOLDER
    print("Saving undistorted polar images to the teach folder...")
    np.savez_compressed(TEACH_FOLDER + "/teach_undistorted.npz",
                        teach_polar_imgs=teach_polar_imgs_undistorted,
                        teach_azimuth_angles=teach_azimuth_angles,
                        teach_azimuth_timestamps=teach_azimuth_timestamps,
                        teach_vertex_timestamps=teach_vertex_timestamps,
                        teach_vertex_transforms=teach_vertex_transforms,
                        teach_times=teach_times, teach_vertex_ids=teach_vertex_ids)
    
    print("Saving undistorted polar images to the repeat folder...")
# SAVE REPEAT CONTENT IN THE REPEAT FOLDER
np.savez_compressed(REPEAT_FOLDER + "/repeat_undistorted.npz",
                    repeat_polar_imgs=repeat_polar_imgs_undistorted,
                    repeat_azimuth_angles=repeat_azimuth_angles,
                    repeat_azimuth_timestamps=repeat_azimuth_timestamps,
                    repeat_vertex_timestamps=repeat_vertex_timestamps,
                    repeat_edge_transforms=repeat_edge_transforms,
                    repeat_times=repeat_times, repeat_vertex_ids = repeat_vertex_ids,dist=vtr_estimated_ptr)

