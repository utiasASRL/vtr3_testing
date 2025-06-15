import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import os

import pyboreas as pb

from scripts.direct.motion_models import *

class GPStateEstimator:
    def __init__(self, opts, res):
        with torch.no_grad():
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #self.device = 'cpu'

            self.kImgPadding = 200
            self.diplay_intensity_normalisation = None


            radar_res = res
            self.radar_res = torch.tensor(radar_res).to(self.device)
            if 'optimisation_first_step' in opts['estimation']:
                self.optimisation_first_step = opts['estimation']['optimisation_first_step']
            else:
                self.optimisation_first_step = 0.1

            # Load the motion model
            self.use_doppler = torch.tensor(opts['estimation']['doppler_cost']).to(self.device)
            self.use_direct = torch.tensor(opts['estimation']['direct_cost']).to(self.device)
            self.use_gyro = torch.tensor('gyro' in opts['estimation']['motion_model']).to(self.device)
            self.estimate_ang_vel = torch.tensor('const_w' in opts['estimation']['motion_model']).to(self.device)
            self.doppler_radar = torch.tensor(opts['radar']['doppler_enabled']).to(self.device)
            if not self.use_doppler and not self.use_direct:
                raise ValueError("Invalid estimation parameters: need at least one cost function to be enabled")
            if self.use_doppler and not self.doppler_radar:
                raise ValueError("Doppler cost function enabled but no doppler radar")
            if self.estimate_ang_vel and not self.use_direct:
                raise ValueError("Angular velocity estimation enabled but no direct cost function")
            if self.use_doppler and not self.use_direct:
                if not self.use_gyro:
                    self.motion_model = MotionModel_lut['const_vel'](device=self.device)
                else:
                    self.motion_model = MotionModel_lut['const_body_vel_gyro'](device=self.device)
            else:
                self.motion_model = MotionModel_lut[opts['estimation']['motion_model']](device=self.device)
            
            self.state_init = self.motion_model.getInitialState()
            # if self.use_direct and not self.estimate_ang_vel and not self.use_gyro:
            #     raise ValueError("Direct cost function enabled but no angular velocity estimation or gyro: problem with choice of motion model")
            self.pose_estimation = self.use_gyro or self.estimate_ang_vel

            self.vy_bias = torch.tensor(opts['estimation']['vy_bias_prior']).to(self.device)


            # Initialise the GP parameters
            kNeighbourhoodFactor = 1.0
            l_az = float(opts['gp']['lengthscale_az'])
            l_range = float(opts['gp']['lengthscale_range'])
            self.size_az = torch.tensor(int(kNeighbourhoodFactor*l_az)).to(self.device)
            self.size_range = torch.tensor(int(kNeighbourhoodFactor*l_range)).to(self.device)

            # Radar parameters (for the doppler and convolution)
            df_dt = float(opts['radar']['del_f']) * float(opts['radar']['meas_freq'])
            self.radar_beta = torch.tensor(float(opts['radar']['beta_corr_fact']) * (float(opts['radar']['ft']) + float(opts['radar']['del_f'])/2.0) / df_dt).to(self.device)
            self.vel_to_bin = torch.tensor(2*self.radar_beta / radar_res).to(self.device)
            range_start = int(np.ceil(float(opts['doppler']['min_range']) / radar_res))
            range_end = int(np.floor(float(opts['doppler']['max_range']) / radar_res))
            self.nb_bins = torch.tensor(range_end - range_start + 1 + 2*self.kImgPadding).to(self.device)


            # Prepare the GP convolutions for the image interlacing
            x = np.arange(-self.size_az.cpu().numpy(), self.size_az.cpu().numpy()+1)
            mask_smooth = x%2 == 0
            mask_interp = x%2 != 0
            x_smooth = x[mask_smooth].astype(np.float32)
            x_interp = x[mask_interp].astype(np.float32)
            y = np.arange(-self.size_range.cpu().numpy(), self.size_range.cpu().numpy()+1)
            XX_smooth, YY_smooth = np.meshgrid(x_smooth, y)
            XX_interp, YY_interp = np.meshgrid(x_interp, y)
            self.X_smooth = np.vstack((XX_smooth.T.flatten(), YY_smooth.T.flatten())).T
            self.X_interp = np.vstack((XX_interp.T.flatten(), YY_interp.T.flatten())).T

            sz = float(opts['gp']['sz'])
            n_smooth = self.X_smooth.shape[0]
            K_smooth = self.seKernel_(self.X_smooth, self.X_smooth, l_az, l_range) + sz**2*np.eye(n_smooth)
            Kinv_smooth = np.linalg.inv(K_smooth)
            ks_smooth = self.seKernel_(np.array([[0, 0]]), self.X_smooth, l_az, l_range)
            self.beta_smooth = (ks_smooth@Kinv_smooth).flatten()

            n_interp = self.X_interp.shape[0]
            K_interp = self.seKernel_(self.X_interp, self.X_interp, l_az, l_range) + sz**2*np.eye(n_interp)
            Kinv_interp = np.linalg.inv(K_interp)
            ks_interp = self.seKernel_(np.array([[0, 0]]), self.X_interp, l_az, l_range)
            self.beta_interp = (ks_interp@Kinv_interp).flatten()

            self.beta_smooth_torch_conv = torch.nn.Conv2d(1, 1, (len(x_smooth), len(y)), bias=False, padding=(len(x_smooth)//2, len(y)//2))
            beta_smooth_tensor = torch.tensor(self.beta_smooth.reshape((1, 1, len(x_smooth), len(y))).astype(np.float32)).to(self.device)
            self.beta_smooth_torch_conv.weight = torch.nn.Parameter(beta_smooth_tensor)
            self.beta_interp_torch_conv = torch.nn.Conv2d(1, 1, (len(x_interp), len(y)), bias=False, padding=(len(x_interp)//2, len(y)//2))
            beta_interp_tensor = torch.tensor(self.beta_interp.reshape((1, 1, len(x_interp), len(y))).astype(np.float32)).to(self.device)
            self.beta_interp_torch_conv.weight = torch.nn.Parameter(beta_interp_tensor)


            # Doppler range bounds
            self.max_range_idx = torch.tensor(int(np.floor(float(opts['doppler']['max_range']) / radar_res))).to(self.device)
            self.min_range_idx = torch.tensor(int(np.ceil(float(opts['doppler']['min_range']) / radar_res))).to(self.device)


            if self.use_direct:
                # Direct range bounds
                self.max_range_idx_direct = torch.tensor(int(np.floor(opts['direct']['max_range'] / radar_res))).to(self.device)
                self.min_range_idx_direct = torch.tensor(int(np.ceil(opts['direct']['min_range'] / radar_res))).to(self.device)

                # Prepare the local_map
                local_map_res = float(opts['direct']['local_map_res'])
                max_local_map_range = float(opts['direct']['max_local_map_range'])
                local_map_size = int(max_local_map_range/local_map_res)*2 + 1
                self.local_map = torch.zeros((local_map_size, local_map_size)).to(self.device)
                temp_x = (torch.arange( -local_map_size//2, local_map_size//2, 1).to(self.device) + 1) * local_map_res
                X = -temp_x.unsqueeze(0).T.repeat(1,local_map_size)
                Y = temp_x.unsqueeze(0).repeat(local_map_size,1)
                self.local_map_xy = torch.stack((X, Y), dim=2).unsqueeze(-1).to(self.device)
                self.local_map_res = torch.tensor(local_map_res).to(self.device)
                self.local_map_zero_idx = torch.tensor(int(max_local_map_range/local_map_res)).to(self.device)
                self.local_map_polar = self.localMapToPolarCoord_(radar_res, 400)
                self.local_map_mask = (self.local_map_polar[:,:,1] < max_local_map_range) & (self.local_map_polar[:,:,1] > float(opts['direct']['min_range']))

                local_map_update_alpha = float(opts['direct']['local_map_update_alpha'])
                self.one_minus_alpha = torch.tensor(1 - local_map_update_alpha).to(self.device)
                self.alpha = torch.tensor(local_map_update_alpha).to(self.device)


                # Doppler shift to range
                self.shift_to_range = torch.tensor(radar_res / 2.0).to(self.device)
                self.range_vec = torch.arange(self.max_range_idx_direct).to(self.device).float() * self.radar_res + (self.radar_res/2.0)

            self.current_rot = torch.tensor(0.0).to(self.device).double()
            self.current_pos = torch.zeros(2).to(self.device).double()

            self.max_acc = torch.tensor(float(opts['estimation']['max_acceleration'])).to(self.device)

            self.previous_vel = torch.tensor(0.0).to(self.device)
            self.max_diff_vel = self.max_acc * 0.25

            self.bin_vec = torch.arange(self.nb_bins, device=self.device).int()

            self.step_counter = 0



            


            self.kImgPadding = torch.tensor(self.kImgPadding).to(self.device)

    def setIntialState(self, state):
        with torch.no_grad():
            self.state_init = state


    def seKernel_(self, X1, X2, l_az, l_range):
        temp_X1 = X1.copy()
        temp_X2 = X2.copy()
        temp_X1[:, 0] = temp_X1[:, 0] / l_az
        temp_X2[:, 0] = temp_X2[:, 0] / l_az
        temp_X1[:, 1] = temp_X1[:, 1] / l_range
        temp_X2[:, 1] = temp_X2[:, 1] / l_range
        dist = pairwise_distances(X1, X2, metric='sqeuclidean')
        return np.exp(-dist/2)


    #@torch.compile
    def getUpDownPolarImages(self, img):
        mean_even = np.mean(img[::2, :])
        mean_odd = np.mean(img[1::2, :])
        in_even = img[::2, :] - mean_even
        in_odd = img[1::2, :] - mean_odd

        # Prepare the input for the torch convolution
        with torch.no_grad():
            in_even_device = torch.tensor(in_even).unsqueeze(0).unsqueeze(0).to(self.device)
            in_odd_device = torch.tensor(in_odd).unsqueeze(0).unsqueeze(0).to(self.device)

            even_smooth_torch = self.beta_smooth_torch_conv(in_even_device)
            even_interp_torch = self.beta_interp_torch_conv(in_even_device)
            odd_smooth_torch = self.beta_smooth_torch_conv(in_odd_device)
            odd_interp_torch = self.beta_interp_torch_conv(in_odd_device)
            # Remove extra rows if the output of the convolution is larger than the input
            if even_smooth_torch.shape[2] > in_even.shape[0]:
                even_smooth_torch = even_smooth_torch[:, :, :-1, :]
            if even_interp_torch.shape[2] > in_even.shape[0]:
                even_interp_torch = even_interp_torch[:, :, :-1, :]
            if odd_smooth_torch.shape[2] > in_odd.shape[0]:
                odd_smooth_torch = odd_smooth_torch[:, :, :-1, :]
            if odd_interp_torch.shape[2] > in_odd.shape[0]:
                odd_interp_torch = odd_interp_torch[:, :, :-1, :]

            out_even = torch.zeros((1, 1, img.shape[0], img.shape[1]), dtype=torch.float32).to(self.device)
            out_odd = torch.zeros((1, 1, img.shape[0], img.shape[1]), dtype=torch.float32).to(self.device)
            out_even[:, :, ::2, :] = even_smooth_torch
            out_even[:, :, 1:-1:2, :] = even_interp_torch[:, :, 1:, :]
            out_odd[:, :, ::2, :] = odd_interp_torch
            out_odd[:, :, 1::2, :] = odd_smooth_torch
            out_odd[:, :, -1, :] = 0


            # Get standard deviation of each image (under the median)
            even_std = torch.std(out_even, dim=3, keepdim=True)
            odd_std = torch.std(out_odd)
            odd_std = torch.std(out_odd, dim=3, keepdim=True)
            out_even -= 2.0*even_std
            out_odd -= 2.0*odd_std
            out_even[out_even < 0] = 0
            out_odd[out_odd < 0] = 0

            # Add gaussian blur to the images
            out_even = torchvision.transforms.functional.gaussian_blur(out_even, (9,1), 3)
            out_odd = torchvision.transforms.functional.gaussian_blur(out_odd, (9,1), 3)

            # Normalise each row by the maximum value
            out_even = out_even / torch.max(out_even, dim=3, keepdim=True)[0]
            out_odd = out_odd / torch.max(out_odd, dim=3, keepdim=True)[0]

            # Replace NaN values by 0
            out_even[torch.isnan(out_even)] = 0
            out_odd[torch.isnan(out_odd)] = 0

            out_even = out_even.squeeze()
            out_odd = out_odd.squeeze()

            out_even[:self.size_az, :] = 0
            out_even[-self.size_az:, :] = 0
            out_odd[:self.size_az, :] = 0
            out_odd[-self.size_az:, :] = 0
            out_even[:, :self.size_range] = 0
            out_even[:, -self.size_range:] = 0
            out_odd[:, :self.size_range] = 0
            out_odd[:, -self.size_range:] = 0

            return out_odd, out_even


            
    # The image is a 400*N image, az_r is a 400*N*2 tensor of image coordinates
    #@torch.compile
    def bilinearInterpolation_(self, im, az_r, with_jac = False):
        with torch.no_grad():
            az0 = torch.floor(az_r[:, :, 0]).int()
            az1 = az0 + 1
            
            r0 = torch.floor(az_r[:, :, 1]).int()
            r1 = r0 + 1

            az0 = torch.clamp(az0, 0, im.shape[0]-1)
            az1 = torch.clamp(az1, 0, im.shape[0]-1)
            r0 = torch.clamp(r0, 0, im.shape[1]-1)
            r1 = torch.clamp(r1, 0, im.shape[1]-1)
            az_r[:,:,0] = torch.clamp(az_r[:,:,0], 0, im.shape[0]-1)
            az_r[:,:,1] = torch.clamp(az_r[:,:,1], 0, im.shape[1]-1)
            
            Ia = im[ az0, r0 ]
            Ib = im[ az1, r0 ]
            Ic = im[ az0, r1 ]
            Id = im[ az1, r1 ]
            
            local_1_minus_r = (r1.float()-az_r[:, :, 1])
            local_r = (az_r[:, :, 1]-r0.float())
            local_1_minus_az = (az1.float()-az_r[:, :, 0])
            local_az = (az_r[:, :, 0]-az0.float())
            wa = local_1_minus_az * local_1_minus_r
            wb = local_az * local_1_minus_r
            wc = local_1_minus_az * local_r
            wd = local_az * local_r

            img_interp = wa*Ia + wb*Ib + wc*Ic + wd*Id

            if not with_jac:
                return img_interp
            else:
                d_I_d_az_r = torch.empty((az_r.shape[0], az_r.shape[1], 1, 2), device=self.device)
                d_I_d_az_r[:, :, 0, 0] = (Ib - Ia)*local_1_minus_r + (Id - Ic)*local_r
                d_I_d_az_r[:, :, 0, 1] = (Ic - Ia)*local_1_minus_az + (Id - Ib)*local_az
                return img_interp, d_I_d_az_r

            
    # The image is a 400*N image, az_r is a 400*N*2 tensor of image coordinates
    #@torch.compile
    def bilinearInterpolationSparse_(self, im, az_r):
        with torch.no_grad():
            az0 = torch.floor(az_r[:, 0]).int()
            az1 = az0 + 1
            
            r0 = torch.floor(az_r[:, 1]).int()
            r1 = r0 + 1

            az0 = torch.clamp(az0, 0, im.shape[0]-1)
            az1 = torch.clamp(az1, 0, im.shape[0]-1)
            r0 = torch.clamp(r0, 0, im.shape[1]-1)
            r1 = torch.clamp(r1, 0, im.shape[1]-1)
            az_r[:,0] = torch.clamp(az_r[:,0], 0, im.shape[0]-1)
            az_r[:,1] = torch.clamp(az_r[:,1], 0, im.shape[1]-1)
            
            Ia = im[ az0, r0 ]
            Ib = im[ az1, r0 ]
            Ic = im[ az0, r1 ]
            Id = im[ az1, r1 ]
            
            local_1_minus_r = (r1.float()-az_r[:, 1])
            local_r = (az_r[:, 1]-r0.float())
            local_1_minus_az = (az1.float()-az_r[:, 0])
            local_az = (az_r[:, 0]-az0.float())
            wa = local_1_minus_az * local_1_minus_r
            wb = local_az * local_1_minus_r
            wc = local_1_minus_az * local_r
            wd = local_az * local_r

            img_interp = wa*Ia + wb*Ib + wc*Ic + wd*Id

            d_I_d_az_r = torch.empty((az_r.shape[0], 1, 2), device=self.device)
            d_I_d_az_r[:, 0, 0] = (Ib - Ia)*local_1_minus_r + (Id - Ic)*local_r
            d_I_d_az_r[:, 0, 1] = (Ic - Ia)*local_1_minus_az + (Id - Ib)*local_az
            return img_interp, d_I_d_az_r

        

    #@torch.compile
    def costFunctionAndJacobian_(self, state, doppler, direct, degraded=False, iter_num=0):
        with torch.no_grad():
            state_size = len(state)
            velocities, d_vel_d_state, pos, d_pos_d_state, rot, d_rot_d_state = self.motion_model.getVelPosRot(state, with_jac=True)
            velocities = velocities.reshape((-1,1,2))
            mask = velocities[:,0,0] > 3.0
            velocities[mask,0,1] = velocities[mask,0,1] + self.vy_bias
            velocities[~mask,0,1] = velocities[~mask,0,1] + velocities[~mask,0,0]*self.vy_bias/3.0
            d_vel_d_state[~mask,1,:] = d_vel_d_state[~mask,1,:] + self.vy_bias/3.0 * d_vel_d_state[~mask,0,:]
            shifts = (velocities @ self.vel_to_bin_vec.reshape((-1,2,1))).squeeze()
            # we can change the shifts to be zero

            d_shift_d_state = self.vel_to_bin_vec.reshape((-1,1,2)) @ d_vel_d_state
            if not self.chirp_up:
                shifts = -shifts
                d_shift_d_state = -d_shift_d_state

            # Doppler cost
            if doppler:
                interp_sparse, aligned_odd_coeff_sparse = self.imgDopplerInterpAndJacobianManual_(shifts)
                residual = interp_sparse * self.temp_even_img_sparse
                jacobian = aligned_odd_coeff_sparse.reshape((-1, 1, 1)) @ d_shift_d_state[self.doppler_az_ids_sparse,:,:] * (self.temp_even_img_sparse.unsqueeze(-1).unsqueeze(-1))
                residual = residual.flatten()
                jacobian = jacobian.reshape((-1,state_size))
                if degraded:
                    weights = ((torch.clip(torch.abs(interp_sparse - self.temp_even_img_sparse), 0, 1) - 1)**6 ).flatten().unsqueeze(-1)
                    jacobian = jacobian * weights
            # Direct cost
            if direct:
                cart_corrected_sparse, d_cart_d_rot_sparse, d_cart_d_shift_sparse = self.polarToCartCoordCorrectionSparse_(pos, rot, shifts)

                # Get the corresponding localMap coordinates
                cart_idx_sparse = self.cartToLocalMapIDSparse_(cart_corrected_sparse).squeeze()

                interp_direct_sparse, d_interp_direct_d_xy_sparse = self.bilinearInterpolationSparse_(self.local_map_blurred, cart_idx_sparse)
                residual_direct_sparse = interp_direct_sparse * (self.polar_intensity_sparse)

                d_cart_sparse_d_state = (d_cart_d_shift_sparse @ d_shift_d_state.reshape((-1,1,state_size)))[self.direct_az_ids_sparse,:,:]
                if d_rot_d_state is not None:
                    d_cart_sparse_d_state[:,:,-1] += (d_cart_d_rot_sparse@(d_rot_d_state[self.direct_az_ids_sparse].reshape((-1,1,1))) ).squeeze()
                d_cart_sparse_d_state += d_pos_d_state[self.direct_az_ids_sparse].reshape((-1,2,state_size))
                d_cart_sparse_d_state[:,0,:] = d_cart_sparse_d_state[:,0,:] / (-self.local_map_res)
                d_cart_sparse_d_state[:,1,:] = d_cart_sparse_d_state[:,1,:] / self.local_map_res


                # print("sam: I am in costFunctionAndJacobian_ and the shape of temp_polar_intensity_sparse is", temp_polar_intensity_sparse.shape)
               
                # fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

                # # # First plot
                # axs[0].set_title("Blurred Local Map")
                # im1 = axs[0].imshow(self.local_map_blurred.cpu().numpy())
                # fig.colorbar(im1, ax=axs[0])

                # # Second plot
                # axs[1].set_title("Sparse Polar Intensity Repeat Scan")
                # im2 = axs[1].imshow((temp_polar_intensity_sparse.cpu().numpy()*10000).astype(np.uint8))
                # fig.colorbar(im2, ax=axs[1])

                # # Adjust spacing and show
                # plt.tight_layout()

                
                jacobian_direct_sparse = ((d_interp_direct_d_xy_sparse @ d_cart_sparse_d_state) * (self.polar_intensity_sparse.unsqueeze(-1).unsqueeze(-1))).squeeze()


                residual_direct = residual_direct_sparse.flatten()
                jacobian_direct = jacobian_direct_sparse.reshape((-1,state_size))
                if degraded:
                    weights_direct = ((torch.clip(torch.abs(interp_direct_sparse - self.polar_intensity_sparse), 0, 1) - 1)**6 ).flatten().unsqueeze(-1)
                    jacobian_direct = jacobian_direct * weights_direct


            if doppler and direct:
                residual = torch.cat((residual, residual_direct), 0)
                jacobian = torch.cat((jacobian, jacobian_direct), 0)
                return residual, jacobian
            elif doppler:
                return residual, jacobian
            elif direct:
                return residual_direct, jacobian_direct
            

    def toLocalMapRegistration(self, teach_local_map, teach_frame, repeat_frame, chirp_up=True, potential_flip=False):
        with torch.no_grad():
            self.chirp_up = chirp_up
            self.timestamps = torch.tensor(repeat_frame.timestamps.flatten()).to(self.device).squeeze()
            # Prepare the data in torch
            self.azimuths = torch.tensor(repeat_frame.azimuths.flatten()).to(self.device).float()
            self.nb_azimuths = torch.tensor(len(repeat_frame.azimuths.flatten())).to(self.device) # number of azimuths 400
            self.motion_model.setTime(self.timestamps, self.timestamps[0])
            
            # Initialise the direction vectors
            # only use for doppler
            dirs = torch.empty((self.nb_azimuths, 2), device=self.device) # 400 by 2
            dirs[:, 0] = torch.cos(self.azimuths)
            dirs[:, 1] = torch.sin(self.azimuths)
            self.vel_to_bin_vec = self.vel_to_bin*dirs
            # doppler possible

            self.polar_intensity = torch.tensor(repeat_frame.polar).to(self.device)
            # print("polar_intensity shape", self.polar_intensity.shape)

            # normalization and smoothing
            polar_std = torch.std(self.polar_intensity, dim=1)
            polar_mean = torch.mean(self.polar_intensity, dim=1)
            self.polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
            self.polar_intensity[self.polar_intensity < 0] = 0
            self.polar_intensity = torchvision.transforms.functional.gaussian_blur(self.polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
            self.polar_intensity /= torch.max(self.polar_intensity, dim=1, keepdim=True)[0]
            self.polar_intensity[torch.isnan(self.polar_intensity)] = 0

            # print("sam: polar_intensity shape: ", self.polar_intensity.shape) 
            
            # sam no undertstand
            # run in debug mode and inspect matrix shape and structures 
            # looks like it is extracting sparse points from the polar image
            ### Preparation for the localMap update
            #print("TODO: See at not storing the full matrix")
            # 1712 * 
            range_vec = torch.arange(self.max_range_idx).to(self.device).float() * self.radar_res + (self.radar_res*0.5)
            # print("range_vec shape", range_vec.shape) # 4961
            self.polar_coord_raw_gp_infered = torch.zeros((self.nb_azimuths, self.max_range_idx, 2)).to(self.device)
            self.polar_coord_raw_gp_infered[:, :, 0] = self.azimuths.unsqueeze(1).repeat(1, self.max_range_idx)
            self.polar_coord_raw_gp_infered[:, :, 1] = range_vec.unsqueeze(0).repeat(self.nb_azimuths, 1)

            temp_intensity = self.polar_intensity[:, :self.max_range_idx_direct]
            mask_direct = temp_intensity != 0
            mask_direct[:, :self.min_range_idx_direct] = False
            # print("mask_direct shape", mask_direct.shape)
            self.polar_intensity_sparse = temp_intensity[mask_direct] #self.local_map_blurred

            # # I want to plot the polar intensity sparse in cart format 
            # print("sam: polar_intensity shape", self.polar_intensity.shape)
            # print("sam: polar_intensity_sparse shape", self.polar_intensity_sparse.shape)

            
            self.direct_r_sparse = self.range_vec.unsqueeze(0).repeat(self.nb_azimuths, 1)[mask_direct]
            # print("direct_r_sparse shape", self.direct_r_sparse.shape)
            self.direct_az_ids_sparse = torch.arange(self.nb_azimuths, device=self.device).unsqueeze(-1).repeat(1,self.max_range_idx_direct)[mask_direct]
            self.direct_r_ids_sparse = torch.arange(self.max_range_idx_direct, device=self.device).unsqueeze(0).repeat(self.nb_azimuths, 1)[mask_direct]
            self.mask_direct_even = torch.ones_like(self.polar_intensity_sparse, device=self.device, dtype=torch.bool)
            self.mask_direct_odd = torch.zeros_like(self.mask_direct_even, device=self.device, dtype=torch.bool)
            self.direct_nb_non_zero = torch.tensor(self.polar_intensity_sparse.shape[0], device=self.device)
            self.direct_r_ids_even = self.direct_r_ids_sparse[self.mask_direct_even]
            self.direct_r_ids_odd = self.direct_r_ids_sparse[self.mask_direct_odd]
            self.direct_r_even = self.direct_r_sparse[self.mask_direct_even]
            self.direct_r_odd = self.direct_r_sparse[self.mask_direct_odd]
            self.direct_az_ids_even = self.direct_az_ids_sparse[self.mask_direct_even]
            self.direct_az_ids_odd = self.direct_az_ids_sparse[self.mask_direct_odd]
            # sam no undertstand 

            self.local_map = torch.tensor(teach_local_map).to(self.device)
            
            # teach_frame_cart = pb.utils.radar.radar_polar_to_cartesian(teach_frame.azimuths.astype(np.float32), teach_frame.polar, 0.040308, 0.2384, 640, False, True)
            # repeat_frame_cart = pb.utils.radar.radar_polar_to_cartesian(repeat_frame.azimuths.astype(np.float32), self.polar_intensity.detach().cpu().numpy(), 0.040308, 0.2384, 640, False, True)
            # print("sam converted repeat scan to cartesian shape", repeat_frame_cart.shape)
            
            self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(self.local_map.unsqueeze(0).unsqueeze(0), 3).squeeze()
            normalizer = torch.max(self.local_map) / torch.max(self.local_map_blurred)
            self.local_map_blurred *= normalizer


            self.step_counter = 1
            result = self.solve_(self.state_init, 1000, 1e-6, 1e-5, verbose=True, degraded=False)
            
            self.state_init = result.clone()

            return result.detach().cpu().numpy()

    # leo's pairwise registration WIP
    # fix_frame, moving_frame (polar_image, azimuths, timestamps)
    def pairwiseRegistration(self, fix_frame, moving_frame, chirp_up=True, potential_flip=False):
        with torch.no_grad():
            self.chirp_up = chirp_up
            self.timestamps = torch.tensor(fix_frame.timestamps.flatten()).to(self.device).squeeze()
            # Prepare the data in torch
            self.azimuths = torch.tensor(fix_frame.azimuths.flatten()).to(self.device).float()
            self.nb_azimuths = torch.tensor(len(fix_frame.azimuths.flatten())).to(self.device) # number of azimuths 400
            self.motion_model.setTime(self.timestamps, self.timestamps[0])
            
            # Initialise the direction vectors
            # only use for doppler
            dirs = torch.empty((self.nb_azimuths, 2), device=self.device) # 400 by 2
            dirs[:, 0] = torch.cos(self.azimuths)
            dirs[:, 1] = torch.sin(self.azimuths)
            self.vel_to_bin_vec = self.vel_to_bin*dirs
            # doppler possible

            self.polar_intensity = torch.tensor(moving_frame.polar).to(self.device)
            # print("polar_intensity shape", self.polar_intensity.shape)

            # normalization and smoothing
            polar_std = torch.std(self.polar_intensity, dim=1)
            polar_mean = torch.mean(self.polar_intensity, dim=1)
            self.polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
            self.polar_intensity[self.polar_intensity < 0] = 0
            self.polar_intensity = torchvision.transforms.functional.gaussian_blur(self.polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
            self.polar_intensity /= torch.max(self.polar_intensity, dim=1, keepdim=True)[0]
            self.polar_intensity[torch.isnan(self.polar_intensity)] = 0
            
            # sam no undertstand
            # run in debug mode and inspect matrix shape and structures 
            # looks like it is extracting sparse points from the polar image
            ### Preparation for the localMap update
            #print("TODO: See at not storing the full matrix")
            # 1712 * 
            range_vec = torch.arange(self.max_range_idx).to(self.device).float() * self.radar_res + (self.radar_res*0.5)
            # print("range_vec shape", range_vec.shape) # 4961
            self.polar_coord_raw_gp_infered = torch.zeros((self.nb_azimuths, self.max_range_idx, 2)).to(self.device)
            self.polar_coord_raw_gp_infered[:, :, 0] = self.azimuths.unsqueeze(1).repeat(1, self.max_range_idx)
            self.polar_coord_raw_gp_infered[:, :, 1] = range_vec.unsqueeze(0).repeat(self.nb_azimuths, 1)

            temp_intensity = self.polar_intensity[:, :self.max_range_idx_direct]
            mask_direct = temp_intensity != 0
            mask_direct[:, :self.min_range_idx_direct] = False
            # print("mask_direct shape", mask_direct.shape)
            self.polar_intensity_sparse = temp_intensity[mask_direct] #self.local_map_blurred

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.title("Polar Intensity- pairwise")
            # plt.imshow(temp_intensity.cpu().numpy())
            # plt.colorbar()
            # plt.show()
            
            self.direct_r_sparse = self.range_vec.unsqueeze(0).repeat(self.nb_azimuths, 1)[mask_direct]
            # print("direct_r_sparse shape", self.direct_r_sparse.shape)
            self.direct_az_ids_sparse = torch.arange(self.nb_azimuths, device=self.device).unsqueeze(-1).repeat(1,self.max_range_idx_direct)[mask_direct]
            self.direct_r_ids_sparse = torch.arange(self.max_range_idx_direct, device=self.device).unsqueeze(0).repeat(self.nb_azimuths, 1)[mask_direct]
            self.mask_direct_even = torch.ones_like(self.polar_intensity_sparse, device=self.device, dtype=torch.bool)
            self.mask_direct_odd = torch.zeros_like(self.mask_direct_even, device=self.device, dtype=torch.bool)
            self.direct_nb_non_zero = torch.tensor(self.polar_intensity_sparse.shape[0], device=self.device)
            self.direct_r_ids_even = self.direct_r_ids_sparse[self.mask_direct_even]
            self.direct_r_ids_odd = self.direct_r_ids_sparse[self.mask_direct_odd]
            self.direct_r_even = self.direct_r_sparse[self.mask_direct_even]
            self.direct_r_odd = self.direct_r_sparse[self.mask_direct_odd]
            self.direct_az_ids_even = self.direct_az_ids_sparse[self.mask_direct_even]
            self.direct_az_ids_odd = self.direct_az_ids_sparse[self.mask_direct_odd]
            # sam no undertstand 
            
            fix_polar_intensity = torch.tensor(fix_frame.polar).to(self.device)
            polar_std = torch.std(fix_polar_intensity, dim=1)
            polar_mean = torch.mean(fix_polar_intensity, dim=1)

            # normalization and smoothing
            fix_polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
            fix_polar_intensity[fix_polar_intensity < 0] = 0
            fix_polar_intensity = torchvision.transforms.functional.gaussian_blur(fix_polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
            fix_polar_intensity /= torch.max(fix_polar_intensity, dim=1, keepdim=True)[0]
            fix_polar_intensity[torch.isnan(fix_polar_intensity)] = 0

            # sam no undertstand before: correct the map 
            temp_polar_to_interp = self.local_map_polar.clone()
            # print("temp_polar_to_interp shape", temp_polar_to_interp.shape) # 4001 by 4001 by 2
            temp_polar_to_interp[:,:,0] -= (self.azimuths[0])
            temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] = temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(self.device)
            temp_polar_to_interp[:,:,0] *= ((self.nb_azimuths) / (2*torch.pi))
            temp_polar_to_interp[:,:,1] -= (self.radar_res/2.0)
            temp_polar_to_interp[:,:,1] /= self.radar_res
            fix_polar_intensity = torch.concatenate((fix_polar_intensity, fix_polar_intensity[0,:].unsqueeze(0)), dim=0)
            local_map_update = self.bilinearInterpolation_(fix_polar_intensity, temp_polar_to_interp, with_jac=False)
            # sam no undertstand after 
            
            # plt.title("Local Map Update- pairwise")
            # plt.imshow(local_map_update.cpu().numpy())
            # plt.show()
            
            self.local_map[self.local_map_mask] = local_map_update[self.local_map_mask]
            # print("local map mask shape", self.local_map_mask.shape)

             # save one-to-one local map
            local_map_path = "/home/samqiao/ASRL/vtr3_testing/scripts/direct/grassy_t2_r3"
            local_map_path = local_map_path + '/local_map_one_to_one/'
            os.makedirs(local_map_path, exist_ok=True)
            if self.local_map is not None:
                    mid_scan_timestamp = fix_frame.timestamps[200]/1e9

                    # print("sam: local map timestamp one to one", fix_frame.timestamps[1]/1e9)
                    cv2.imwrite(local_map_path + str(mid_scan_timestamp) + '.png', self.local_map.detach().cpu().numpy()*255)

            self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(self.local_map.unsqueeze(0).unsqueeze(0), 3).squeeze()
            normalizer = torch.max(self.local_map) / torch.max(self.local_map_blurred)
            self.local_map_blurred *= normalizer
            self.step_counter = 1


            result = self.solve_(self.state_init, 1000, 1e-6, 1e-5, verbose=True, degraded=False)
            self.state_init = result.clone()
            return result.detach().cpu().numpy()
        

    def localMaptoLocalMapRegistration(self, teach_local_map, repeat_local_map, chirp_up=True, potential_flip=False):
        # use DRO to generate local maps for teach and repeat sequences
        with torch.no_grad():
            self.local_map = torch.tensor(teach_local_map).to(self.device)

            self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(self.local_map.unsqueeze(0).unsqueeze(0), 3).squeeze()
            normalizer = torch.max(self.local_map) / torch.max(self.local_map_blurred)
            self.local_map_blurred *= normalizer


            repeat_local_map = torch.tensor(repeat_local_map).to(self.device)

            radar_res = 0.040308
            self.local_map_polar = self.localMapToPolarCoord_(radar_res, 400)
            # self.local_map_mask = (self.local_map_polar[:,:,1] < max_local_map_range) & (self.local_map_polar[:,:,1] > float(opts['direct']['min_range']))

            # repeat_local_map_polar = self.
            # should be a point where I apply motion distortion

            self.polar_intensity_sparse  = self.bi


            self.step_counter = 1

            ### need help on this function..... :((((((()))))))


            result = self.solve_(self.state_init, 1000, 1e-6, 1e-5, verbose=True, degraded=False)
            self.state_init = result.clone()
            return result.detach().cpu().numpy()



    # The shift is in pixels (thus should be alternating signs)
    #@torch.compile
    def perLineInterpolation_(self, img, shift):
        with torch.no_grad():
            shift_int = torch.floor(shift).int()
            shift_frac = shift - shift_int.float()
            az = torch.tile(torch.arange(img.shape[0]).unsqueeze(1), (1, img.shape[1])).to(self.device)
            r_0 = torch.tile(torch.arange(img.shape[1]).unsqueeze(0), (img.shape[0], 1)).to(self.device)
            r_0 = r_0 + shift_int.reshape(-1, 1)
            r_1 = r_0 + 1
            r_0 = torch.clamp(r_0, 0, img.shape[1]-1)
            r_1 = torch.clamp(r_1, 0, img.shape[1]-1)
            Ia = img[az, r_0]
            Ib = img[az, r_1]
            interp = (1-shift_frac).reshape(-1,1)*Ia + shift_frac.reshape(-1,1)*Ib
            return interp

    #@torch.compile    
    def localMapToPolarCoord_(self, radar_res, nb_azimuths):
        with torch.no_grad():
            # Get the polar coordinates of the image
            cart = self.local_map_xy

            # Get the new polar coordinates
            polar = torch.zeros((cart.shape[0], cart.shape[1], 2)).to(self.device)
            polar[:, :, 0] = torch.atan2(cart[:, :, 1, 0], cart[:, :, 0, 0])
            polar[:, :, 1] = torch.sqrt(cart[:, :, 0, 0]**2 + cart[:, :, 1, 0]**2)

            #polar[polar[:,:,0]<0] = polar[polar[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(self.device)
            #polar[:,:,0] *= (nb_azimuths / (2*torch.pi))
            #polar[:,:,1] /= radar_res
            return polar



    #@torch.compile    
    def polarToCartCoordCorrectionSparse_(self, pos, rot, doppler_shift):
        with torch.no_grad():
            # Get the polar coordinates of the image
            c_az_min = torch.cos(self.azimuths)
            s_az_min = torch.sin(self.azimuths)
            c_az = c_az_min[self.direct_az_ids_sparse]
            s_az = s_az_min[self.direct_az_ids_sparse]
            even_range = self.range_vec[self.direct_r_ids_even] - doppler_shift[self.direct_az_ids_even] * self.shift_to_range
            odd_range = self.range_vec[self.direct_r_ids_odd] + doppler_shift[self.direct_az_ids_odd] * self.shift_to_range
            x = torch.empty(self.direct_nb_non_zero, device=self.device)
            x[self.mask_direct_even] = c_az[self.mask_direct_even] * even_range
            x[self.mask_direct_odd] = c_az[self.mask_direct_odd] * odd_range
            y = torch.empty(self.direct_nb_non_zero, device=self.device)
            y[self.mask_direct_even] = s_az[self.mask_direct_even] * even_range
            y[self.mask_direct_odd] = s_az[self.mask_direct_odd] * odd_range


            # Rotate the coordinates
            c_rot_min = torch.cos(rot.squeeze())
            s_rot_min = torch.sin(rot.squeeze())
            c_rot = c_rot_min[self.direct_az_ids_sparse]
            s_rot = s_rot_min[self.direct_az_ids_sparse]
            x_c_rot = x * c_rot
            y_s_rot = y * s_rot
            x_s_rot = x * s_rot
            y_c_rot = y * c_rot
            x_rot = x_c_rot - y_s_rot
            y_rot = x_s_rot + y_c_rot

            # Translate the coordinates
            x_trans = x_rot + pos.squeeze()[self.direct_az_ids_sparse, 0]
            y_trans = y_rot + pos.squeeze()[self.direct_az_ids_sparse, 1]

            # Stack the coordinates
            cart = torch.stack((x_trans.unsqueeze(-1), y_trans.unsqueeze(-1)), dim=1)

            # Compute the jacobians
            d_cart_d_rot = torch.zeros((self.direct_nb_non_zero, 2, 1), device=self.device)
            d_cart_d_rot[:, 0, 0] = -y_rot
            d_cart_d_rot[:, 1, 0] = x_rot


            d_cart_d_shift = torch.empty((self.nb_azimuths, 2, 1), device=self.device)
            d_cart_d_shift[::2, 0, 0] = c_az_min[::2]*-self.shift_to_range
            d_cart_d_shift[1::2, 0, 0] = c_az_min[1::2]*self.shift_to_range
            d_cart_d_shift[::2, 1, 0] = s_az_min[::2]*-self.shift_to_range
            d_cart_d_shift[1::2, 1, 0] = s_az_min[1::2]*self.shift_to_range

            d_trans_d_cart = torch.empty((self.nb_azimuths, 2, 2), device=self.device)
            d_trans_d_cart[:,0,0] = c_rot_min
            d_trans_d_cart[:,0,1] = -s_rot_min
            d_trans_d_cart[:,1,0] = s_rot_min
            d_trans_d_cart[:,1,1] = c_rot_min


            d_cart_d_shift = d_trans_d_cart @ d_cart_d_shift

            return cart, d_cart_d_rot, d_cart_d_shift



    #@torch.compile
    def polarCoordCorrection_(self, pos, rot):
        with torch.no_grad():
            # Get the polar coordinates of the image
            polar_coord = self.polar_coord_raw_gp_infered

            c_az = torch.cos(polar_coord[:, :, 0])
            s_az = torch.sin(polar_coord[:, :, 0])
            x = c_az * polar_coord[:, :, 1]
            y = s_az * polar_coord[:, :, 1]

            # Rotate the coordinates
            c_rot = torch.cos(rot)
            s_rot = torch.sin(rot)
            x_rot = x * c_rot - y * s_rot
            y_rot = x * s_rot + y * c_rot

            # Translate the coordinates
            x_trans = x_rot+pos[:, 0]
            y_trans = y_rot+pos[:, 1]

            # Get the new polar coordinates
            polar = torch.zeros((self.nb_azimuths, polar_coord.shape[1], 2)).to(self.device)
            polar[:, :, 0] = torch.atan2(y_trans, x_trans)
            sq_norm = x_trans**2 + y_trans**2
            polar[:, :, 1] = torch.sqrt(sq_norm)

            return polar




    #@torch.compile
    def imgDopplerInterpAndJacobianManual_(self, shift):
        with torch.no_grad():
            shift_int = torch.floor(-shift).int()
            bin_mat_shifted_int = self.doppler_bin_vec_sparse + shift_int[self.doppler_az_ids_sparse]
            shift_frac = (-shift - shift_int)[self.doppler_az_ids_sparse]
            aligned_odd_coeff = self.odd_coeff[self.doppler_az_ids_sparse, bin_mat_shifted_int]
            odd_interp = shift_frac*aligned_odd_coeff + self.odd_bias[self.doppler_az_ids_sparse, bin_mat_shifted_int]

            return odd_interp, -aligned_odd_coeff


    # Gauss Newton optimisation
    #@torch.compile
    def solve_(self, state_init, nb_iter=20, cost_tol=1e-6, step_tol=1e-6, verbose=False, degraded=False):
        self.degraded_log = degraded
        with torch.no_grad():
            if self.estimate_ang_vel and self.step_counter == 0:
                remove_angular = torch.tensor(True).to(self.device)
            else:
                remove_angular = torch.tensor(False).to(self.device)
            if self.step_counter == 0 and not self.use_doppler:
                return state_init

            if degraded and isinstance(self.motion_model, ConstBodyAccGyro):
                self.motion_model.block_acc = True
            state = state_init.clone()
            first_cost = torch.tensor(np.inf).to(self.device)
            prev_cost = first_cost
            first_quantum = self.optimisation_first_step
            step_quantum = first_quantum
            last_decreasing_state = state.clone()
            last_decreasing_grad = torch.zeros_like(state)
            for i in torch.arange(nb_iter, device=self.device):
                
                res, jac = self.costFunctionAndJacobian_(state, self.use_doppler, self.use_direct and (self.step_counter>0), degraded)

                if remove_angular and not self.use_gyro:
                    jac = jac[:, :-1]


                grad = 3*torch.sum(res.flatten().unsqueeze(-1)**2 * jac.reshape((-1,jac.shape[-1])), 0)
                cost = torch.sum((res**3).flatten())

                if i == 0:
                    last_decreasing_grad = grad.clone()
                else:
                    if cost < prev_cost:
                        state = last_decreasing_state.clone()
                        grad = last_decreasing_grad.clone()
                        step_quantum = step_quantum / 2
                    else:
                        last_decreasing_state = state.clone()
                        last_decreasing_grad = grad.clone()

                grad_norm = torch.linalg.norm(grad)

                if step_quantum < 1e-5:
                    if verbose:
                        print("Step quantum too small, breaking")
                    break


                if grad_norm < 1e-9:
                    print("breaking because of grad_norm < 1e-9")
                    break
                step = (grad / grad_norm) * step_quantum
                #step = grad * step_quantum

                
                if remove_angular and not self.use_gyro:
                    step = torch.cat((step, torch.zeros(1).to(self.device)), dim=0)
                
                # print("step shape", step.shape)
                state += step


                step_norm = torch.linalg.norm(step)
                cost_change = cost - prev_cost

                if i == 0:
                    first_cost = cost
                
                # Print iter cost step_norm cost_change with 3 decimals and scientific notation
                # if verbose:
                #     print("Iter: ", i, " - Cost: ", "{:.3e}".format(cost), " - Step norm: ", "{:.3e}".format(step_norm), " - Cost change: ", "{:.3e}".format(cost_change))

                if step_norm < step_tol:
                    print("breaking because of step_norm < step_tol")
                    break

                if torch.abs(cost_change/cost) < cost_tol:
                    print("breaking because of abs(cost_change/cost) < cost_tol")
                    break
                prev_cost = cost


            #print("\nNumber of iterations: ", i.cpu().numpy(), " - State step norm: ", torch.norm(state - state_init).cpu().numpy())

            vel, _, _ = self.motion_model.getVelPosRot(state, with_jac=False)
            if (torch.abs(torch.norm(vel[-1,:]) - self.previous_vel) > self.max_diff_vel):
                if not degraded:
                    state = self.solve_(state_init, nb_iter=nb_iter, cost_tol=cost_tol, step_tol=step_tol, verbose=verbose, degraded=True)
                    if isinstance(self.motion_model, ConstBodyAccGyro):
                        state[-1] = 0
                #elif (torch.abs(torch.norm(vel[-1,:]) - self.previous_vel) > self.max_diff_vel):
                #    state = state_init.clone()
                #    if isinstance(self.motion_model, ConstBodyAccGyro):
                #        state[-1] = 0

            if not degraded:
                vel, _, _ = self.motion_model.getVelPosRot(state, with_jac=False)
                self.previous_vel = torch.norm(vel[-1,:])
                self.max_diff_vel = self.motion_model.time[-1] * self.max_acc
            

            # fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

            # # # First plot
            # axs[0].set_title("Blurred Local Map")
            # im1 = axs[0].imshow(self.local_map_blurred.cpu().numpy())
            # fig.colorbar(im1, ax=axs[0])


            # # Second plot
            # axs[1].set_title("Sparse Polar Intensity Repeat Scan")
            # im2 = axs[1].imshow((temp_polar_intensity_sparse.cpu().numpy()*10000).astype(np.uint8))
            # fig.colorbar(im2, ax=axs[1])

            # # Adjust spacing and show
            # plt.tight_layout()
            # plt.show()
            
            return state

    #@torch.compile
    def cartToLocalMapID_(self, xy):
        out = torch.empty_like(xy, device=self.device)
        out[:,:,0,0] = (xy[:,:,0,0] / (-self.local_map_res)) + self.local_map_zero_idx
        out[:,:,1,0] = (xy[:,:,1,0] / (self.local_map_res)) + self.local_map_zero_idx
        return out

    #@torch.compile
    def cartToLocalMapIDSparse_(self, xy):
        out = torch.empty_like(xy, device=self.device)
        out[:,0,0] = (xy[:,0,0] / (-self.local_map_res)) + self.local_map_zero_idx
        out[:,1,0] = (xy[:,1,0] / (self.local_map_res)) + self.local_map_zero_idx
        return out



    # Move localMap
    def moveLocalMap_(self, pos, rot):
        with torch.no_grad():
            # Set to zero the first and last row and column of the localMap
            self.local_map[0, :] = 0
            self.local_map[-1, :] = 0
            self.local_map[:, 0] = 0
            self.local_map[:, -1] = 0

            # Get the coordinate of the new localMap in the former localMap
            temp_rot_mat = torch.tensor([[torch.cos(rot), -torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]]).to(self.device)
            temp_pos = pos.reshape((-1,1))

            # Get the new coordinates
            new_xy = temp_rot_mat @ self.local_map_xy + temp_pos
            new_idx = self.cartToLocalMapID_(new_xy)

            # Get the new localMap via bilinear interpolation
            self.local_map = self.bilinearInterpolation_(self.local_map, new_idx).squeeze().float()

            

    def odometryStep(self, polar_image, azimuths, timestamps, chirp_up=True, potential_flip=False):
        with torch.no_grad():
            if isinstance(self.motion_model, ConstBodyAccGyro):
                self.motion_model.block_acc = potential_flip

            self.chirp_up = chirp_up
            self.timestamps = torch.tensor(timestamps).to(self.device).squeeze()

            # Prepare the template according to the first timestamp
            if self.pose_estimation:

                if self.step_counter > 0:
                    # Get the velocities
                    vel_body, prev_scan_pos, prev_scan_rot = self.motion_model.getVelPosRot(self.state_init, with_jac=False)


                    frame_pos, frame_rot = self.motion_model.getPosRotSingle(self.state_init, self.timestamps[0])

                    # Update the current position and rotation
                    rot_mat = torch.tensor([[torch.cos(self.current_rot), -torch.sin(self.current_rot)], [torch.sin(self.current_rot), torch.cos(self.current_rot)]]).to(self.device)
                    self.current_pos = self.current_pos + rot_mat @ frame_pos.double()
                    self.current_rot = self.current_rot + frame_rot.double()
                    
                    if self.use_direct:
                        # Get the shift for each line 
                        shift = (vel_body.reshape((-1,1,2)) @ self.vel_to_bin_vec.reshape((-1,2,1))).squeeze()
                        per_line_shift = shift/2.0
                        if not self.prev_chirp_up:
                            per_line_shift = -per_line_shift
                        if self.doppler_radar:
                            per_line_shift[1::2] *= -1
                        # Get the interpolated image
                        prev_shifted = self.perLineInterpolation_(self.polar_intensity, per_line_shift)
                        ######## Get the pos and rot to project at the end of the scan
                        rot_mats_transposed = torch.concatenate((torch.cos(prev_scan_rot), torch.sin(prev_scan_rot), -torch.sin(prev_scan_rot), torch.cos(prev_scan_rot)), dim=1).reshape((-1,2,2))
                        prev_scan_pos = prev_scan_pos.reshape((-1,2,1))
                        pos = rot_mats_transposed@(-prev_scan_pos + frame_pos.reshape((-1,2,1))) 
                        rot = -prev_scan_rot + frame_rot

                        # Undistort the polar image to the end of the scan
                        polar_coord_corrected = self.polarCoordCorrection_(pos, rot)
                        polar_coord_corrected[:,:,0] -= (self.azimuths[0])
                        polar_coord_corrected[polar_coord_corrected[:,:,0]<0] = polar_coord_corrected[polar_coord_corrected[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(self.device)
                        polar_coord_corrected[:,:,0] *= ((self.nb_azimuths) / (2*torch.pi))
                        polar_coord_corrected[:,:,1] -= (self.radar_res/2.0)
                        polar_coord_corrected[:,:,1] /= self.radar_res

                        polar_target = self.bilinearInterpolation_(prev_shifted, polar_coord_corrected, with_jac=False)


                        # Probably should use an openCV-like remap function to prevent pixelisation
                        temp_polar_to_interp = self.local_map_polar.clone()
                        temp_polar_to_interp[:,:,0] -= (self.azimuths[0])
                        temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] = temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(self.device)
                        temp_polar_to_interp[:,:,0] *= ((self.nb_azimuths) / (2*torch.pi))
                        temp_polar_to_interp[:,:,1] -= (self.radar_res/2.0)
                        temp_polar_to_interp[:,:,1] /= self.radar_res
                        local_map_update = self.bilinearInterpolation_(polar_target, temp_polar_to_interp, with_jac=False)

                        if self.step_counter == 1:
                            self.local_map[self.local_map_mask] = local_map_update[self.local_map_mask]
                        else:
                            self.moveLocalMap_(frame_pos, frame_rot)
                            self.local_map[self.local_map_mask] = self.one_minus_alpha * self.local_map[self.local_map_mask] + self.alpha * local_map_update[self.local_map_mask]
                        #self.local_map = local_map_update

                        # TODO check different values for the gaussian blur
                        self.local_map_blurred = torchvision.transforms.functional.gaussian_blur(self.local_map.unsqueeze(0).unsqueeze(0), 3).squeeze()
                        normalizer = torch.max(self.local_map) / torch.max(self.local_map_blurred)
                        self.local_map_blurred *= normalizer


            # Query the GP images
            #print("TODO: Change inside the getUpDownPolarImages to remove computations when no doppler is used")
            if self.use_doppler:
                odd_img, even_img = self.getUpDownPolarImages(polar_image[:,self.min_range_idx:(self.max_range_idx+1)])
            


            # Prepare the data in torch
            self.azimuths = torch.tensor(azimuths).to(self.device).float()
            self.nb_azimuths = torch.tensor(len(azimuths)).to(self.device)
            self.motion_model.setTime(self.timestamps, self.timestamps[0])

            # Initialise the direction vectors
            dirs = torch.empty((self.nb_azimuths, 2), device=self.device)
            dirs[:, 0] = torch.cos(self.azimuths)
            dirs[:, 1] = torch.sin(self.azimuths)
            self.vel_to_bin_vec = self.vel_to_bin*dirs



            ### Preparation for the doppler
            if self.use_doppler:
                # Padding the images for the doppler cost
                self.temp_even_img = torch.cat((torch.zeros((self.nb_azimuths, self.kImgPadding), dtype=torch.float32).to(self.device), even_img, torch.zeros((self.nb_azimuths, self.kImgPadding), dtype=torch.float32).to(self.device)), dim=1)
                temp_odd_img = torch.cat((torch.zeros((self.nb_azimuths, self.kImgPadding), dtype=torch.float32).to(self.device), odd_img, torch.zeros((self.nb_azimuths, self.kImgPadding), dtype=torch.float32).to(self.device)), dim=1)

                # Coefficients for the interpolation
                self.odd_coeff = torch.empty_like(self.temp_even_img, device=self.device)
                self.odd_coeff[:, :-1] = temp_odd_img[:, 1:] - temp_odd_img[:, :-1]
                self.odd_coeff[:, -1] = 0
                self.odd_bias = temp_odd_img.clone()

                mask_doppler = self.temp_even_img != 0
                self.temp_even_img_sparse = self.temp_even_img[mask_doppler]
                # Get the idx of the non zero values
                self.doppler_az_ids_sparse = torch.arange(self.nb_azimuths, device=self.device).unsqueeze(-1).repeat(1,self.temp_even_img.shape[1])[mask_doppler]
                self.doppler_bin_vec_sparse = torch.arange(self.nb_bins, device=self.device).unsqueeze(0).repeat(self.nb_azimuths,1)[mask_doppler]


            ### Prerparation for the direct cost
            # Create the polar coordinates for the image
            if self.use_direct:
                if self.use_doppler:
                    self.polar_intensity = torch.zeros((len(azimuths), self.min_range_idx + odd_img.shape[1])).to(self.device)
                    self.polar_intensity[::2, self.min_range_idx:] = even_img[::2, :]
                    self.polar_intensity[1::2, self.min_range_idx:] = odd_img[1::2, :]
                else:
                    self.polar_intensity = torch.tensor(polar_image).to(self.device)
                    polar_std = torch.std(self.polar_intensity, dim=1)
                    polar_mean = torch.mean(self.polar_intensity, dim=1)
                    self.polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
                    self.polar_intensity[self.polar_intensity < 0] = 0
                    self.polar_intensity = torchvision.transforms.functional.gaussian_blur(self.polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
                    self.polar_intensity /= torch.max(self.polar_intensity, dim=1, keepdim=True)[0]
                    self.polar_intensity[torch.isnan(self.polar_intensity)] = 0
                

                ### Preparation for the localMap update
                #print("TODO: See at not storing the full matrix")
                range_vec = torch.arange(self.max_range_idx).to(self.device).float() * self.radar_res + (self.radar_res*0.5)
                self.polar_coord_raw_gp_infered = torch.zeros((self.nb_azimuths, self.max_range_idx, 2)).to(self.device)
                self.polar_coord_raw_gp_infered[:, :, 0] = self.azimuths.unsqueeze(1).repeat(1, self.max_range_idx)
                self.polar_coord_raw_gp_infered[:, :, 1] = range_vec.unsqueeze(0).repeat(self.nb_azimuths, 1)

                temp_intensity = self.polar_intensity[:, :self.max_range_idx_direct]
                mask_direct = temp_intensity != 0
                mask_direct[:, :self.min_range_idx_direct] = False
                self.polar_intensity_sparse = temp_intensity[mask_direct]

                self.direct_r_sparse = self.range_vec.unsqueeze(0).repeat(self.nb_azimuths, 1)[mask_direct]
                self.direct_az_ids_sparse = torch.arange(self.nb_azimuths, device=self.device).unsqueeze(-1).repeat(1,self.max_range_idx_direct)[mask_direct]
                self.direct_r_ids_sparse = torch.arange(self.max_range_idx_direct, device=self.device).unsqueeze(0).repeat(self.nb_azimuths, 1)[mask_direct]
                if self.doppler_radar:
                    self.mask_direct_even = torch.empty_like(mask_direct, device=self.device)
                    self.mask_direct_even[1::2] = False
                    self.mask_direct_even[::2] = True
                    self.mask_direct_even = self.mask_direct_even[mask_direct]
                    self.mask_direct_odd = torch.empty_like(mask_direct, device=self.device)
                    self.mask_direct_odd[::2] = False
                    self.mask_direct_odd[1::2] = True
                    self.mask_direct_odd = self.mask_direct_odd[mask_direct]
                #elif not chirp_up:
                else:
                    self.mask_direct_even = torch.ones_like(self.polar_intensity_sparse, device=self.device, dtype=torch.bool)
                    self.mask_direct_odd = torch.zeros_like(self.mask_direct_even, device=self.device, dtype=torch.bool)

                self.direct_nb_non_zero = torch.tensor(self.polar_intensity_sparse.shape[0], device=self.device)
                self.direct_r_ids_even = self.direct_r_ids_sparse[self.mask_direct_even]
                self.direct_r_ids_odd = self.direct_r_ids_sparse[self.mask_direct_odd]
                self.direct_r_even = self.direct_r_sparse[self.mask_direct_even]
                self.direct_r_odd = self.direct_r_sparse[self.mask_direct_odd]
                self.direct_az_ids_even = self.direct_az_ids_sparse[self.mask_direct_even]
                self.direct_az_ids_odd = self.direct_az_ids_sparse[self.mask_direct_odd]


            ### Perform the optimisation
            if self.motion_model.state_size == 3 and self.use_gyro:
                self.state_init[:2] = self.state_init[:2]*(1+self.state_init[2]*0.250)
            if torch.norm(self.state_init[:2]) < 0.75:
                #self.state_init[:2] = 0.0
                self.state_init[:] = 0.0
            result = self.solve_(self.state_init, 250, 1e-6, 1e-5)
            self.state_init = result.clone()

            self.prev_chirp_up = chirp_up
            self.step_counter += 1
            return result.detach().cpu().numpy()

    def getDopplerVelocity(self):
        if not self.use_doppler:
            raise ValueError("Doppler not used")
        #if not self.use_direct:
        #    return self.state_init[:2].detach().cpu().numpy()
        
        save_use_direct = self.use_direct
        self.use_direct = False
        if isinstance(self.motion_model, ConstBodyAccGyro):
            save_block_acc = self.motion_model.block_acc
            self.motion_model.block_acc = True
        result = self.solve_(self.state_init, 250, 1e-6, 1e-5)

        self.use_direct = save_use_direct
        if isinstance(self.motion_model, ConstBodyAccGyro):
            self.motion_model.block_acc = save_block_acc

        return result[:2].detach().cpu().numpy()



    def getAzPosRot(self):
        if self.pose_estimation:
            rot_mat = torch.tensor([[torch.cos(self.current_rot), -torch.sin(self.current_rot)], [torch.sin(self.current_rot), torch.cos(self.current_rot)]]).to(self.device)

            _, scan_pos, scan_rot = self.motion_model.getVelPosRot(self.state_init, with_jac=False)
            pos = rot_mat @ scan_pos.double() + self.current_pos.unsqueeze(1)
            rot = scan_rot.double() + self.current_rot

            return pos.detach().cpu().numpy(), rot.detach().cpu().numpy()
        else:
            return None, None




    def generateVisualisation(self, radar_frame, img_size, img_res, inverted=False, text=True):
        # Detach and put to numpy the images
        polar_img = radar_frame.polar
        polar_img_odd, polar_img_even = self.getUpDownPolarImages(radar_frame.polar)
        polar_img_odd = polar_img_odd.detach().cpu().numpy()
        polar_img_even = polar_img_even.detach().cpu().numpy()

        text_color = (255, 255, 255)
        vel_color = (0, 255, 0)
        if inverted:
            text_color = (0, 0, 0)
            vel_color = (255, 0, 0)
        

        # Normalising paramters for the display
        if self.diplay_intensity_normalisation is None:
            self.diplay_intensity_normalisation = 1.0/(1.5*np.max(polar_img_even))


        # Convert to cartesian for display
        radar_cartesian = pb.utils.radar.radar_polar_to_cartesian(radar_frame.azimuths.astype(np.float32), polar_img, radar_frame.resolution, img_res, img_size, False, True)
        cartesian_img_odd = pb.utils.radar.radar_polar_to_cartesian(radar_frame.azimuths.astype(np.float32), polar_img_odd, radar_frame.resolution, img_res, img_size, False, True)
        cartesian_img_even = pb.utils.radar.radar_polar_to_cartesian(radar_frame.azimuths.astype(np.float32), polar_img_even, radar_frame.resolution, img_res, img_size, False, True)
        if inverted:
            radar_cartesian = np.clip(1.0 - radar_cartesian, 0, 1)
            cartesian_img_odd = np.clip(1.0 - cartesian_img_odd*self.diplay_intensity_normalisation, 0, 1)
            cartesian_img_even = np.clip(1.0 - cartesian_img_even*self.diplay_intensity_normalisation, 0, 1)
        
            # Get the difference image between the odd and even images
            diff_img = np.ones((cartesian_img_even.shape[0], cartesian_img_even.shape[1], 3))
            diff_raw = cartesian_img_even - cartesian_img_odd
            error_pos = diff_raw.copy()
            error_neg = diff_raw.copy()
            error_pos[error_pos<0] = 0
            error_neg[error_neg>0] = 0
            diff_img[:, :, 0] = 1 - error_pos
            diff_img[:, :, 1] = 1 - np.clip(error_pos - error_neg, 0, 1)
            diff_img[:, :, 2] = 1 + error_neg
            diff_img *= 255
        else:
            diff = polar_img_odd - polar_img_even
            diff = pb.utils.radar.radar_polar_to_cartesian(radar_frame.azimuths.astype(np.float32), diff, radar_frame.resolution, img_res, img_size, False, False)
            diff_img = np.zeros((diff.shape[0], diff.shape[1], 3))
            diff_img[:, :, 0] = np.clip(diff, 0, np.inf)
            diff_img[:, :, 2] = np.clip(-diff, 0, np.inf)
            diff_img *= self.diplay_intensity_normalisation*400
        


        # Create image with radar cartesian raw, the odd and even images and the disparity image
        sub_size = cartesian_img_odd.shape[0]
        img = np.zeros((2*sub_size, 3*sub_size, 3), dtype=np.uint8)
        img[:sub_size, :sub_size, :] = cv2.cvtColor(radar_cartesian*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        img[:sub_size, sub_size:2*sub_size, :] = cv2.cvtColor((cartesian_img_odd)*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        img[:sub_size, 2*sub_size:, :] = cv2.cvtColor((cartesian_img_even)*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        img[sub_size:, :sub_size, :] = diff_img.astype(np.uint8)

        # Add the velocity vector to the images (arrow or line over the cartesian images)
        velocity = self.state_init[:2].cpu().numpy()
        kScaleArrow = 5 
        #cv2.arrowedLine(img, (int(sub_size/2), int(sub_size/2)), (int(sub_size/2 + velocity[1]*kScaleArrow), int(sub_size/2 - velocity[0]*kScaleArrow)), vel_color, 2)
        #cv2.arrowedLine(img, (int(3*sub_size/2), int(sub_size/2)), (int(3*sub_size/2 + velocity[1]*kScaleArrow), int(sub_size/2 - velocity[0]*kScaleArrow)), vel_color, 2)
        #cv2.arrowedLine(img, (int(5*sub_size/2), int(sub_size/2)), (int(5*sub_size/2 + velocity[1]*kScaleArrow), int(sub_size/2 - velocity[0]*kScaleArrow)), vel_color, 2)
        cv2.arrowedLine(img, (int(sub_size/2), int(3*sub_size/2)), (int(sub_size/2 + velocity[1]*kScaleArrow), int(3*sub_size/2 - velocity[0]*kScaleArrow)), vel_color, 2)

        # Add text legend to the image
        if text:
            cv2.putText(img, 'Raw radar', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(img, 'GP image odd azimuths (filtered)', (10+cartesian_img_odd.shape[1], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(img, 'GP image even azimuths (filtered)', (10+2*cartesian_img_odd.shape[1], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(img, 'GP odd/even difference (no prior)', (10, 20+cartesian_img_odd.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(img, 'TO ADD', (10+cartesian_img_odd.shape[1], 20+cartesian_img_odd.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        if self.use_direct:
            resized_local_map = (cv2.resize(self.local_map.clone().detach().cpu().numpy(), (img_size, img_size))*255)
            resized_local_map = resized_local_map.clip(0, 255).astype(np.uint8)
            if inverted:
                resized_local_map = 255 - resized_local_map
            img[sub_size:, 2*sub_size:, :] = resized_local_map.reshape((resized_local_map.shape[0], resized_local_map.shape[1], 1)).repeat(3, axis=2)
            cv2.arrowedLine(img, (int(5*sub_size/2), int(3*sub_size/2)), (int(5*sub_size/2 + velocity[1]*kScaleArrow), int(3*sub_size/2 - velocity[0]*kScaleArrow)), vel_color, 2)
            if text:
                cv2.putText(img, 'On-the-fly local map', (10+2*cartesian_img_odd.shape[1], 20+cartesian_img_odd.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            if text:
                cv2.putText(img, 'No local map in doppler-only mode', (10+2*cartesian_img_odd.shape[1], 20+cartesian_img_odd.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        return img