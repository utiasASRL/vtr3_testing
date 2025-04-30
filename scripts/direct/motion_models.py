from enum import Enum
import numpy as np
import torch

class MotionModels(str, Enum):
    const_vel_const_w = "const_vel_const_w",
    const_body_vel_const_w = "const_body_vel_const_w",
    const_body_vel_gyro = "const_body_vel_gyro",
    const_body_acc_gyro = "const_body_acc_gyro",
    const_acc_const_w  = "const_acc_const_w",
    const_vel = "const_vel",
    no_slip  = "no_slip"
    const_rotation_const_translation = "const_rotation_const_translation"


class MotionModel:
    def __init__(self, state_size, device='cpu'):
        self.state_size = torch.tensor(state_size).to(device)
        self.device = device
        self.time = None
        self.num_steps = None
        self.t0 = None

    def setTime(self, time, t0):
        self.time = (time - t0).float() * torch.tensor(1.0e-6).to(self.device)
        self.t0 = t0
        self.num_steps = torch.tensor(len(time)).to(self.device)

    def getLocalTime(self, time):
        return (time - self.t0).float() * 1.0e-6

    def getInitialState(self):
        return torch.zeros(self.state_size, device=self.device)
    
    def setInitialState(self, state):
        self.initial_state = state

# Sam: need to implement a function that takes the state and the azimuths and returns the velocity, position and rotation
class ConstRotationConstTranslation(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(3, device)
    
    # Make sure that state and azimuths are torch
    def getVelPosRot(self, state, with_jac = False):
    # the input is different now x,y,theta 
        with torch.no_grad():
            # print("State shape", state.shape)
            # The state is an array of size 2. Duplicate it for each azimuth

            ones = torch.ones_like(self.time)
            zeros = torch.zeros_like(self.time)

            rot = state[2]*ones.unsqueeze(1)
            pos = torch.stack((state[0]*ones, state[1]*ones), dim=1).unsqueeze(2)
            
            
            x_y = state[:2].unsqueeze(0).unsqueeze(2).clone()

            # # Transform the velocities to body centric
            x_y_body = x_y.clone()*ones

            vel_body = torch.zeros((400,2)).to(self.device)

            


            identity_tensor = torch.zeros(400, 2, 3)

            # Set the diagonal elements (0,0) and (1,1) to 1 for all 400 entries
            identity_tensor[:, 0, 0] = 1.0  # First diagonal element
            identity_tensor[:, 1, 1] = 1.0  # Second diagonal element
            
            d_rot_d_state = torch.ones((self.num_steps, 1, 1)).to(self.device)
            d_pos_d_state = identity_tensor.clone().to(self.device)

            d_vel_body_d_state = torch.zeros((self.num_steps, 2, 3), device=self.device)



            identity_tensor_1_by_1 = torch.ones(400, 1, 1)
            # identity_tensor_1_by_1[:, 0, 0] = 1.0  # First diagonal element


            # print("d_vel_body_d_state shape", d_vel_body_d_state.shape)
            # print("d_pos_d_state shape", d_pos_d_state.shape)
            # print("d_rot_d_state shape", d_rot_d_state.shape)
            # print("vel_body shape", vel_body.shape)
            # print("pos shape", pos.shape)
            # print("rot shape", rot.shape)
            if not with_jac:
                return vel_body, pos, rot

            return vel_body, d_vel_body_d_state, pos, d_pos_d_state, rot, d_rot_d_state

    def getPosRotSingle(self, state, time):
        with torch.no_grad():
            # local_time = self.getLocalTime(time)
            rot = state[2]
            pos = torch.stack((state[0], state[1]))
            return pos, rot
# Sam



class ConstVelConstW(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(3, device)
    
    # Make sure that state and azimuths are torch tensors
    #@torch.compile
    def getVelPosRot(self, state, with_jac = False):
        with torch.no_grad():
            print("State shape", state.shape)
            # The state is an array of size 2. Duplicate it for each azimuth
            rot = state[2]*self.time.unsqueeze(1)
            pos = torch.stack((state[0]*self.time, state[1]*self.time), dim=1).unsqueeze(2)
            vel = state[:2].unsqueeze(0).unsqueeze(2).clone()

            # Transform the velocities to body centric
            #vel_body = vel.clone()
            c_rot = torch.cos(rot)
            s_rot = torch.sin(rot)
            vx_c_rot = vel[:, 0]*c_rot
            vy_s_rot = vel[:, 1]*s_rot
            vx_s_rot = vel[:, 0]*s_rot
            vy_c_rot = vel[:, 1]*c_rot
            #vel_body = torch.cat((vel[:, 0].reshape(-1, 1)*c_rot + vel[:, 1].reshape(-1, 1)*s_rot, -vel[:, 0].reshape(-1, 1)*s_rot + vel[:, 1].reshape(-1, 1)*c_rot), dim=1)
            vel_body = torch.cat((vx_c_rot + vy_s_rot, vy_c_rot - vx_s_rot), dim=1)

            if not with_jac:
                return vel_body, pos, rot

            d_rot_d_state = torch.zeros((self.num_steps, 1, 1)).to(self.device)
            d_rot_d_state[:, :, 0] = self.time.unsqueeze(1)

            d_pos_d_state = torch.zeros((self.num_steps, 2, 3)).to(self.device)
            d_pos_d_state[:, 0, 0] = self.time
            d_pos_d_state[:, 1, 1] = self.time

            d_vel_body_d_state = torch.empty((self.num_steps, 2, 3), device=self.device)
            d_vel_body_d_state[:, 0, 0] = c_rot.squeeze()
            d_vel_body_d_state[:, 0, 1] = s_rot.squeeze()
            d_vel_body_d_state[:, 1, 0] = -s_rot.squeeze()
            d_vel_body_d_state[:, 1, 1] = c_rot.squeeze()
            d_vel_body_d_state[:, 0, 2] = vy_c_rot.squeeze()*self.time -vx_s_rot.squeeze()*self.time
            d_vel_body_d_state[:, 1, 2] = -vx_c_rot.squeeze()*self.time -vy_s_rot.squeeze()*self.time

            print("d_vel_body_d_state shape", d_vel_body_d_state.shape)
            print("d_pos_d_state shape", d_pos_d_state.shape)
            print("d_rot_d_state shape", d_rot_d_state.shape)
            print("vel_body shape", vel_body.shape)
            print("pos shape", pos.shape)
            print("rot shape", rot.shape)

            # print("rot", rot)

            return vel_body, d_vel_body_d_state, pos, d_pos_d_state, rot, d_rot_d_state

    def getPosRotSingle(self, state, time):
        with torch.no_grad():
            local_time = self.getLocalTime(time)
            rot = state[2]*local_time
            pos = torch.stack((state[0]*local_time, state[1]*local_time))
            return pos, rot
        
class ConstBodyVelGyro(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(2, device)

        self.initialised = False

    def setGyroData(self, gyro_time, gyro_yaw):
        self.first_gyro_time = gyro_time[0]
        self.gyro_time = torch.tensor(gyro_time - self.first_gyro_time).double().to(self.device)
        self.gyro_yaw = torch.tensor(gyro_yaw).double().to(self.device)
        self.gyro_yaw_original = self.gyro_yaw.clone()

        self.bin_integral = (self.gyro_yaw[1:] + self.gyro_yaw[:-1])*(self.gyro_time[1:] - self.gyro_time[:-1])/2.0
        self.coeff = (self.gyro_yaw[1:] - self.gyro_yaw[:-1])/(self.gyro_time[1:] - self.gyro_time[:-1])
        self.coeff = torch.cat((self.coeff, self.coeff[-1].unsqueeze(0)), dim=0)
        self.offset = self.gyro_yaw[:-1] - self.coeff[:-1]*self.gyro_time[:-1]
        self.offset = torch.cat((self.offset, self.offset[-1].unsqueeze(0)), dim=0)

        self.initialised = True

    def setGyroBias(self, gyro_bias):
        self.gyro_yaw = self.gyro_yaw_original - gyro_bias

        self.bin_integral = (self.gyro_yaw[1:] + self.gyro_yaw[:-1])*(self.gyro_time[1:] - self.gyro_time[:-1])/2.0
        self.coeff = (self.gyro_yaw[1:] - self.gyro_yaw[:-1])/(self.gyro_time[1:] - self.gyro_time[:-1])
        self.coeff = torch.cat((self.coeff, self.coeff[-1].unsqueeze(0)), dim=0)
        self.offset = self.gyro_yaw[:-1] - self.coeff[:-1]*self.gyro_time[:-1]
        self.offset = torch.cat((self.offset, self.offset[-1].unsqueeze(0)), dim=0)

    def setTime(self, time, t0):
        if not self.initialised:
            raise ValueError("Gyro data not set")
        with torch.no_grad():
            if not self.initialised:
                raise ValueError("Gyro data not set")
            super().setTime(time, t0)
            # Get the integral be
            time_local = time.double()*1e-6 - self.first_gyro_time
            time_start = t0.double()*1e-6 - self.first_gyro_time
            start_idx = torch.searchsorted(self.gyro_time, time_start)
            end_idx = torch.searchsorted(self.gyro_time, time_local)
            if end_idx[-1] == 0:
                raise ValueError("Time is before the first gyro data: Currently not supported")
            mask = start_idx == end_idx
            self.r = torch.zeros_like(time).float()
            self.r[mask] = ((time_local[mask]*self.coeff[start_idx-1] + self.offset[start_idx-1] + time_start*self.coeff[start_idx-1] + self.offset[start_idx-1])*0.5*(time_local[mask] - time_start)).float()

            first_bucket = (self.coeff[start_idx-1]*time_start + self.offset[start_idx-1] + self.gyro_yaw[start_idx])*0.5*(self.gyro_time[start_idx]-time_start)

            last_bucket = (self.coeff[end_idx-1]*time_local + self.offset[end_idx-1] + self.gyro_yaw[end_idx-1])*0.5*(time_local - self.gyro_time[end_idx-1])

            mask_one = start_idx + 1 == end_idx
            self.r[mask_one] = (first_bucket + last_bucket[mask_one]).float()
            mask_rest = start_idx + 1 < end_idx
            cumulative_integral = torch.cumsum(self.bin_integral[start_idx:torch.max(end_idx)], dim=0)
            self.r[mask_rest] = (first_bucket + cumulative_integral[end_idx[mask_rest]-2-start_idx] + last_bucket[mask_rest]).float()
            


            self.cos_r = torch.cos(self.r)
            self.sin_r = torch.sin(self.r)

            delta_time = time_local[1:] - time_local[:-1]

            cumulative_cos_r = torch.cumsum((self.cos_r[:-1]+self.cos_r[1:])*0.5*delta_time, dim=0)
            cumulative_sin_r = torch.cumsum((self.sin_r[:-1]+self.sin_r[1:])*0.5*delta_time, dim=0)

            self.R_integral = torch.empty((len(time), 2, 2), device=self.device)
            self.R_integral[0, :, :] = 0.0
            self.R_integral[1:, 0, 0] = cumulative_cos_r
            self.R_integral[1:, 0, 1] = -cumulative_sin_r
            self.R_integral[1:, 1, 0] = cumulative_sin_r
            self.R_integral[1:, 1, 1] = cumulative_cos_r
            


            #print("R: ", self.r)

    def getVelPosRot(self, state, with_jac = False):
        with torch.no_grad():
            # The state is an array of size 2. Duplicate it for each azimuth
            rot = self.r.unsqueeze(1).clone()
            body_vel = state.unsqueeze(0).clone()
            pos = self.R_integral @ body_vel.unsqueeze(2)

            if not with_jac:
                return body_vel, pos, rot
            
            d_rot_d_state = None

            d_vel_body_d_state = torch.zeros((1, 2, 2), device=self.device)
            d_vel_body_d_state[:, 0, 0] = 1.0
            d_vel_body_d_state[:, 1, 1] = 1.0

            d_pos_d_state = self.R_integral.clone()

            return body_vel, d_vel_body_d_state, pos, d_pos_d_state, rot, d_rot_d_state

    def getPosRotSingle(self, state, time):
        times = torch.arange(self.t0, time, 625, dtype=torch.int64).to(self.device)
        if times[-1] != time:
            times = torch.cat((times, time.unsqueeze(0)))
        self.setTime(times, self.t0)

        pos = (self.R_integral[-1, :, :] @ state.unsqueeze(1)).squeeze()
        rot = self.r[-1]
        return pos, rot

class ConstBodyAccGyro(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(3, device)

        self.initialised = False
        self.block_acc = False

    def setGyroData(self, gyro_time, gyro_yaw):
        self.first_gyro_time = gyro_time[0]
        self.gyro_time = torch.tensor(gyro_time - self.first_gyro_time).double().to(self.device)
        self.gyro_yaw = torch.tensor(gyro_yaw).double().to(self.device)
        self.gyro_yaw_original = self.gyro_yaw.clone()

        self.bin_integral = (self.gyro_yaw[1:] + self.gyro_yaw[:-1])*(self.gyro_time[1:] - self.gyro_time[:-1])/2.0
        self.coeff = (self.gyro_yaw[1:] - self.gyro_yaw[:-1])/(self.gyro_time[1:] - self.gyro_time[:-1])
        self.coeff = torch.cat((self.coeff, self.coeff[-1].unsqueeze(0)), dim=0)
        self.offset = self.gyro_yaw[:-1] - self.coeff[:-1]*self.gyro_time[:-1]
        self.offset = torch.cat((self.offset, self.offset[-1].unsqueeze(0)), dim=0)

        self.initialised = True

    def setGyroBias(self, gyro_bias):
        self.gyro_yaw = self.gyro_yaw_original - gyro_bias

        self.bin_integral = (self.gyro_yaw[1:] + self.gyro_yaw[:-1])*(self.gyro_time[1:] - self.gyro_time[:-1])/2.0
        self.coeff = (self.gyro_yaw[1:] - self.gyro_yaw[:-1])/(self.gyro_time[1:] - self.gyro_time[:-1])
        self.coeff = torch.cat((self.coeff, self.coeff[-1].unsqueeze(0)), dim=0)
        self.offset = self.gyro_yaw[:-1] - self.coeff[:-1]*self.gyro_time[:-1]
        self.offset = torch.cat((self.offset, self.offset[-1].unsqueeze(0)), dim=0)


    def setTime(self, time, t0):
        if not self.initialised:
            raise ValueError("Gyro data not set")
        with torch.no_grad():
            if not self.initialised:
                raise ValueError("Gyro data not set")
            super().setTime(time, t0)
            # Get the integral be
            time_local = time.double()*1e-6 - self.first_gyro_time
            time_start = t0.double()*1e-6 - self.first_gyro_time
            start_idx = torch.searchsorted(self.gyro_time, time_start)
            end_idx = torch.searchsorted(self.gyro_time, time_local)
            if end_idx[-1] == 0:
                raise ValueError("Time is before the first gyro data: Currently not supported")
            mask = start_idx == end_idx
            self.r = torch.zeros_like(time).float()
            self.r[mask] = ((time_local[mask]*self.coeff[start_idx-1] + self.offset[start_idx-1] + time_start*self.coeff[start_idx-1] + self.offset[start_idx-1])*0.5*(time_local[mask] - time_start)).float()

            first_bucket = (self.coeff[start_idx-1]*time_start + self.offset[start_idx-1] + self.gyro_yaw[start_idx])*0.5*(self.gyro_time[start_idx]-time_start)

            last_bucket = (self.coeff[end_idx-1]*time_local + self.offset[end_idx-1] + self.gyro_yaw[end_idx-1])*0.5*(time_local - self.gyro_time[end_idx-1])

            mask_one = start_idx + 1 == end_idx
            self.r[mask_one] = (first_bucket + last_bucket[mask_one]).float()
            mask_rest = start_idx + 1 < end_idx
            cumulative_integral = torch.cumsum(self.bin_integral[start_idx:torch.max(end_idx)], dim=0)
            self.r[mask_rest] = (first_bucket + cumulative_integral[end_idx[mask_rest]-2-start_idx] + last_bucket[mask_rest]).float()
            


            self.cos_r = torch.cos(self.r)
            self.sin_r = torch.sin(self.r)

            delta_time = time_local[1:] - time_local[:-1]

            cumulative_cos_r = torch.cumsum((self.cos_r[:-1]+self.cos_r[1:])*0.5*delta_time, dim=0)
            cumulative_sin_r = torch.cumsum((self.sin_r[:-1]+self.sin_r[1:])*0.5*delta_time, dim=0)

            self.R_integral = torch.empty((len(time), 2, 2), device=self.device)
            self.R_integral[0, :, :] = 0.0
            self.R_integral[1:, 0, 0] = cumulative_cos_r
            self.R_integral[1:, 0, 1] = -cumulative_sin_r
            self.R_integral[1:, 1, 0] = cumulative_sin_r
            self.R_integral[1:, 1, 1] = cumulative_cos_r
            
            cumulative_t_cos_r = torch.cumsum(((self.cos_r[:-1]*self.time[:-1]+self.cos_r[1:]*self.time[1:])*0.5)*delta_time, dim=0)
            cumulative_t_sin_r = torch.cumsum(((self.sin_r[:-1]*self.time[:-1]+self.sin_r[1:]*self.time[1:])*0.5)*delta_time, dim=0)

            self.R_t_integral = torch.empty((len(time), 2, 2), device=self.device)
            self.R_t_integral[0, :, :] = 0.0
            self.R_t_integral[1:, 0, 0] = cumulative_t_cos_r
            self.R_t_integral[1:, 0, 1] = -cumulative_t_sin_r
            self.R_t_integral[1:, 1, 0] = cumulative_t_sin_r
            self.R_t_integral[1:, 1, 1] = cumulative_t_cos_r
            


            #print("R: ", self.r)

    def getVelPosRot(self, state, with_jac = False):
        with torch.no_grad():
            # The state is an array of size 2. Duplicate it for each azimuth
            rot = self.r.unsqueeze(1)
            if self.block_acc:
                body_vel = state[:2].unsqueeze(0)
                pos = self.R_integral @ body_vel.unsqueeze(2)
            else:
                body_vel = state[:2] * (1 + state[2]*self.time.unsqueeze(1))
                pos = (self.R_integral + state[2]*self.R_t_integral) @ state[:2].unsqueeze(1)

            if not with_jac:
                return body_vel, pos, rot
            
            d_rot_d_state = None

            if self.block_acc:
                d_vel_body_d_state = torch.zeros((1, 2, 3), device=self.device)
                d_vel_body_d_state[:, 0, 0] = 1.0
                d_vel_body_d_state[:, 1, 1] = 1.0

                d_pos_d_state = torch.zeros((self.num_steps, 2, 3), device=self.device)
                d_pos_d_state[:, 0, 0] = self.R_integral[:, 0, 0]
                d_pos_d_state[:, 0, 1] = self.R_integral[:, 0, 1]
                d_pos_d_state[:, 1, 0] = self.R_integral[:, 1, 0]
                d_pos_d_state[:, 1, 1] = self.R_integral[:, 1, 1]
            else:
                d_vel_body_d_state = torch.zeros((self.num_steps, 2, 3), device=self.device)
                d_vel_body_d_state[:, 0, 0] = 1.0 + self.time*state[2]
                d_vel_body_d_state[:, 1, 1] = 1.0 + self.time*state[2]
                d_vel_body_d_state[:, 0, 2] = self.time * state[0]
                d_vel_body_d_state[:, 1, 2] = self.time * state[1]

                d_pos_d_state = torch.empty((self.num_steps, 2, 3), device=self.device)
                d_pos_d_state[:, :, :2] = self.R_integral + state[2]*self.R_t_integral
                d_pos_d_state[:, :, 2] = (self.R_t_integral @ state[:2].unsqueeze(1)).squeeze()

            return body_vel, d_vel_body_d_state, pos, d_pos_d_state, rot, d_rot_d_state

    def getPosRotSingle(self, state, time):
        times = torch.arange(self.t0, time, 625, dtype=torch.int64).to(self.device)
        if times[-1] != time:
            times = torch.cat((times, time.unsqueeze(0)))
        self.setTime(times, self.t0)

        pos = ((self.R_integral[-1, :, :] + state[2]*self.R_t_integral[-1, :, :]) @ state[:2].unsqueeze(1)).squeeze()
        rot = self.r[-1]
        return pos, rot
    
        


class ConstBodyVelConstW(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(3, device)
    
    # Make sure that state and azimuths are torch tensors
    #@torch.compile
    def getVelPosRot(self, state, with_jac = False):
        with torch.no_grad():
            # The state is an array of size 2. Duplicate it for each azimuth
            rot = state[2]*self.time.unsqueeze(1)
            body_vel = state[:2].unsqueeze(0)

            c_rot = torch.cos(rot)
            s_rot = torch.sin(rot)

            if torch.abs(state[2]) <= 1e-3:
                pos = torch.stack((state[0]*self.time, state[1]*self.time), dim=1).unsqueeze(2)
            else:
                vbx_c_rot = body_vel[:, 0]*c_rot
                vby_s_rot = body_vel[:, 1]*s_rot
                vbx_s_rot = body_vel[:, 0]*s_rot
                vby_c_rot = body_vel[:, 1]*c_rot
                offset = torch.stack(( state[1].reshape(1,1), -state[0].reshape(1,1)), dim=1)
                pos = (1.0/state[2])*(torch.stack((vbx_s_rot + vby_c_rot, -vbx_c_rot + vby_s_rot), dim=1) - offset)


            if not with_jac:
                return body_vel, pos, rot

            d_rot_d_state = torch.zeros((self.num_steps, 1, 1)).to(self.device)
            d_rot_d_state[:, :, 0] = self.time.unsqueeze(1)

            d_vel_body_d_state = torch.zeros((1, 2, 3), device=self.device)
            d_vel_body_d_state[:, 0, 0] = 1.0
            d_vel_body_d_state[:, 1, 1] = 1.0

            d_pos_d_state = torch.zeros((self.num_steps, 2, 3)).to(self.device)
            if torch.abs(state[2]) <= 1e-3:
                d_pos_d_state[:, 0, 0] = self.time
                d_pos_d_state[:, 1, 1] = self.time
            else:
                d_pos_d_state[:, 0, 0] = 1/state[2]*(s_rot.squeeze())
                d_pos_d_state[:, 0, 1] = 1/state[2]*(c_rot.squeeze() - 1)
                d_pos_d_state[:, 1, 0] = 1/state[2]*(-c_rot.squeeze() + 1)
                d_pos_d_state[:, 1, 1] = 1/state[2]*(s_rot.squeeze())
                d_pos_d_state[:, :, 2] = -(1.0/state[2]) * pos.squeeze() 
                d_pos_d_state[:, 0, 2] += (1.0/state[2])*self.time.squeeze()*(vbx_c_rot.squeeze() - vby_s_rot.squeeze())
                d_pos_d_state[:, 1, 2] += (1.0/state[2])*self.time.squeeze()*(vbx_s_rot.squeeze() + vby_c_rot.squeeze())


            return body_vel, d_vel_body_d_state, pos, d_pos_d_state, rot, d_rot_d_state

    def getPosRotSingle(self, state, time):
        with torch.no_grad():
            local_time = self.getLocalTime(time)
            rot = state[2]*local_time
            if torch.abs(state[2]) <= 1e-3:
                pos = torch.stack((state[0]*local_time, state[1]*local_time))
            else:
                pos = (1/state[2])*(torch.stack((state[0]*torch.sin(rot) + state[1]*torch.cos(rot), -state[0]*torch.cos(rot) + state[1]*torch.sin(rot))) - torch.stack((state[1], -state[0])))
            return pos, rot


class ConstAccConstW(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(5, device)

    # Make sure that state and azimuths are torch tensors
    #@torch.compile
    def getVelPosRot(self, state, with_jac = False):
        with torch.no_grad():
            half_time_sq = 0.5*(self.time**2)
            pos = torch.stack((state[0]*self.time + state[2]*half_time_sq, state[1]*self.time + state[3]*half_time_sq), dim=1).unsqueeze(2)
            vel = torch.stack((state[0] + state[2]*self.time, state[1] + state[3]*self.time), dim=1).unsqueeze(2)
            rot = state[4]*self.time.unsqueeze(1)

            # Transform the velocities to body centric
            #vel_body = vel.clone()
            c_rot = torch.cos(rot)
            s_rot = torch.sin(rot)
            vx_c_rot = vel[:, 0]*c_rot
            vy_s_rot = vel[:, 1]*s_rot
            vx_s_rot = vel[:, 0]*s_rot
            vy_c_rot = vel[:, 1]*c_rot
            #vel_body = torch.cat((vel[:, 0].reshape(-1, 1)*c_rot + vel[:, 1].reshape(-1, 1)*s_rot, -vel[:, 0].reshape(-1, 1)*s_rot + vel[:, 1].reshape(-1, 1)*c_rot), dim=1)
            vel_body = torch.cat((vx_c_rot + vy_s_rot, vy_c_rot - vx_s_rot), dim=1)

            if not with_jac:
                return vel_body, pos, rot

            d_rot_d_state = torch.empty((self.num_steps, 1, 1), device=self.device)
            d_rot_d_state[:, :, 0] = self.time.unsqueeze(1)

            d_pos_d_state = torch.empty((self.num_steps, 2, 5), device = self.device)
            d_pos_d_state[:, 0, 1] = 0
            d_pos_d_state[:, 1, 0] = 0
            d_pos_d_state[:, 0, 0] = self.time
            d_pos_d_state[:, 1, 1] = self.time
            d_pos_d_state[:, 0, 2] = half_time_sq
            d_pos_d_state[:, 1, 3] = half_time_sq
            d_pos_d_state[:, 0, 3] = 0
            d_pos_d_state[:, 1, 2] = 0
            d_pos_d_state[:, :, 4] = 0

            d_vel_d_state = torch.zeros((self.num_steps, 2, 5), device = self.device)
            d_vel_d_state[:, 0, 0] = 1
            d_vel_d_state[:, 1, 1] = 1
            d_vel_d_state[:, 0, 2] = self.time
            d_vel_d_state[:, 1, 3] = self.time

            #d_vel_body_d_state = d_vel_d_state.clone()
            d_vel_body_d_vel = torch.zeros((self.num_steps, 2, 2), device = self.device)
            d_vel_body_d_vel[:, 0, 0] = c_rot.squeeze()
            d_vel_body_d_vel[:, 0, 1] = s_rot.squeeze()
            d_vel_body_d_vel[:, 1, 0] = -s_rot.squeeze()
            d_vel_body_d_vel[:, 1, 1] = c_rot.squeeze()

            d_vel_body_d_state = d_vel_body_d_vel @ d_vel_d_state
            d_vel_body_d_state[:, 0, 4] = vy_c_rot.squeeze()*self.time -vx_s_rot.squeeze()*self.time
            d_vel_body_d_state[:, 1, 4] = -vx_c_rot.squeeze()*self.time -vy_s_rot.squeeze()*self.time

            return vel_body, d_vel_body_d_state, pos, d_pos_d_state, rot, d_rot_d_state
        
    def getPosRotSingle(self, state, time):
        with torch.no_grad():
            local_time = self.getLocalTime(time)
            half_time_sq = 0.5*(local_time**2)
            pos = torch.stack((state[0]*local_time + state[2]*half_time_sq, state[1]*local_time + state[3]*half_time_sq))
            rot = state[4]*local_time
            return pos, rot
        
class ConstVel(MotionModel):
    def __init__(self, device='cpu'):
        super().__init__(2, device)

    def getVelPosRot(self, state, with_jac = False):
        with torch.no_grad():
            vel = state.unsqueeze(0).clone()

            if not with_jac:
                return vel, None , None

            d_vel_d_state = torch.zeros((1, 2, 2), device=self.device)
            d_vel_d_state[:, 0, 0] = 1
            d_vel_d_state[:, 1, 1] = 1


            return vel, d_vel_d_state, None, None, None, None

            
        
    def getPosRotSingle(self, state, time):
        raise ValueError("Querying single position and rotation is not sensible for constant velocity model")



class NoSlip:
    def __init__(self, scan_freq=4.0, device='cpu'):
        print("Warning: With no slip there might be a need for calibration paramter")
        self.device = device
        self.az_to_time = torch.tensor(1.0/(2*np.pi*scan_freq)).to(self.device)
        self.state_size = 2
    
    # Make sure that state and azimuths are torch tensors
    def getVelPosRot(self, state, azimuths, with_jac = False):
        raise ValueError("No slip not implemented yet")

# If const_vel_const_w, state is v_0_x, v_0_y, w
# If const_acc_const_w, state is v_0_x, v_0_y, a_x, a_y, w
# If no_slip, state is v_0_x, v_0_y
MotionModel_lut = {
    MotionModels.const_vel_const_w: ConstVelConstW,
    MotionModels.const_body_vel_const_w: ConstBodyVelConstW,
    MotionModels.const_body_vel_gyro: ConstBodyVelGyro,
    MotionModels.const_body_acc_gyro: ConstBodyAccGyro,
    MotionModels.const_acc_const_w: ConstAccConstW,
    MotionModels.const_vel: ConstVel,
    MotionModels.no_slip: NoSlip,
    MotionModels.const_rotation_const_translation: ConstRotationConstTranslation
}
