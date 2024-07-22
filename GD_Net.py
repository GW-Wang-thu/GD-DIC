import torch.nn as nn
import torch
import torch.nn.functional as F
from interpolation import interpolator
from torchvision.transforms import transforms as tf
import numpy as np


class GD_DIC(nn.Module):
    def __init__(self, imsize, device='cuda', zf=1, norm_method='FE', gauge_size=5, init_disp=None):
        super(GD_DIC, self).__init__()
        self.imsize = imsize
        self.zf = zf
        if init_disp == None:
            self.trainable_uv = torch.nn.Parameter(torch.zeros(2, imsize[0]//zf, imsize[1]//zf))
        else:
            self.trainable_uv = torch.nn.Parameter(init_disp)
        self.grid_y, self.grid_x = torch.meshgrid(torch.arange(imsize[0]), torch.arange(imsize[1]))
        self.grid_y, self.grid_x = self.grid_y.to(device), self.grid_x.to(device)
        self.interpolator = interpolator(device=device)
        self.norm = norm_method
        if norm_method == 'Gauge':
            self.gauge_conv = nn.Conv2d(1, 2, gauge_size, stride=1, padding=gauge_size//2)
            self.gauge_conv.weight.data = self.__init_gauge__(gauge_size).to(device)
            for param in self.gauge_conv.parameters():
                param.requires_grad = False
        elif norm_method == 'FE':
            self.__init_KBDB__(device)

    def __init_gauge__(self, gauge_size):
        grid_x, grid_y = torch.meshgrid(torch.arange(gauge_size), torch.arange(gauge_size))
        grid_x = grid_x - gauge_size//2
        grid_y = grid_y - gauge_size//2
        grid_x_f = grid_x / torch.sqrt(grid_y**2+grid_x**2 + 1e-5)
        grid_y_f = grid_y / torch.sqrt(grid_y**2+grid_x**2 + 1e-5)
        return torch.stack([torch.stack([grid_x_f], dim=0),
                            torch.stack([grid_y_f], dim=0)], dim=0)

    def __init_KBDB__(self, device):
        D = np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 0.35]])
        self.KBDB = []
        self.B = []
        for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
            for eta in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                tmp_B = np.array([[-0.5*(1-eta), 0, 0.5*(1-eta), 0, 0.5*(1+eta), 0, -0.5*(1+eta), 0],
                                  [0, -0.5*(1-xi), 0, -0.5*(1+xi), 0, 0.5*(1+xi), 0, 0.5*(1-xi)],
                                  [-0.5*(1-xi), -0.5*(1-eta), -0.5*(1+xi),0.5*(1-eta), 0.5*(1+xi), 0.5*(1+eta), 0.5*(1-xi), -0.5*(1+eta)]])
                tmp_BDB = np.dot(np.dot(tmp_B.T, D), tmp_B)
                self.KBDB.append(torch.from_numpy(tmp_BDB.astype('float32')).to(device))
                self.B.append(tmp_B)

    def __calc_energy__(self):
        u1 = self.upsampled_uv[0, :-1, :-1]
        v1 = self.upsampled_uv[1, :-1, :-1]
        u2 = self.upsampled_uv[0, 1:, :-1]
        v2 = self.upsampled_uv[1, 1:, :-1]
        u3 = self.upsampled_uv[0, 1:, 1:]
        v3 = self.upsampled_uv[1, 1:, 1:]
        u4 = self.upsampled_uv[0, :-1, 1:]
        v4 = self.upsampled_uv[1, :-1, 1:]
        energy_map = torch.zeros_like(u1)
        for KBDB in self.KBDB:
            e1 = KBDB[0, 0]* u1 + KBDB[1, 0]* v1 + KBDB[2, 0]* u2 + KBDB[3, 0]* v2 + KBDB[4, 0]* u3 + KBDB[5, 0]* v3 + KBDB[6, 0]* u4 + KBDB[7, 0]* v4
            e2 = KBDB[0, 1]* u1 + KBDB[1, 1]* v1 + KBDB[2, 1]* u2 + KBDB[3, 1]* v2 + KBDB[4, 1]* u3 + KBDB[5, 1]* v3 + KBDB[6, 1]* u4 + KBDB[7, 1]* v4
            e3 = KBDB[0, 2]* u1 + KBDB[1, 2]* v1 + KBDB[2, 2]* u2 + KBDB[3, 2]* v2 + KBDB[4, 2]* u3 + KBDB[5, 2]* v3 + KBDB[6, 2]* u4 + KBDB[7, 2]* v4
            e4 = KBDB[0, 3]* u1 + KBDB[1, 3]* v1 + KBDB[2, 3]* u2 + KBDB[3, 3]* v2 + KBDB[4, 3]* u3 + KBDB[5, 3]* v3 + KBDB[6, 3]* u4 + KBDB[7, 3]* v4
            e5 = KBDB[0, 4]* u1 + KBDB[1, 4]* v1 + KBDB[2, 4]* u2 + KBDB[3, 4]* v2 + KBDB[4, 4]* u3 + KBDB[5, 4]* v3 + KBDB[6, 4]* u4 + KBDB[7, 4]* v4
            e6 = KBDB[0, 5]* u1 + KBDB[1, 5]* v1 + KBDB[2, 5]* u2 + KBDB[3, 5]* v2 + KBDB[4, 5]* u3 + KBDB[5, 5]* v3 + KBDB[6, 5]* u4 + KBDB[7, 5]* v4
            e7 = KBDB[0, 6]* u1 + KBDB[1, 6]* v1 + KBDB[2, 6]* u2 + KBDB[3, 6]* v2 + KBDB[4, 6]* u3 + KBDB[5, 6]* v3 + KBDB[6, 6]* u4 + KBDB[7, 6]* v4
            e8 = KBDB[0, 7]* u1 + KBDB[1, 7]* v1 + KBDB[2, 7]* u2 + KBDB[3, 7]* v2 + KBDB[4, 7]* u3 + KBDB[5, 7]* v3 + KBDB[6, 7]* u4 + KBDB[7, 7]* v4
            energy_map += e1 * u1 + e2 * v1 + e3 * u2+ e4 * v2 + e5 * u3+ e6 * v3 + e7 * u4 + e8 * v4
        return energy_map * 1.0e4

    def __calc_energy_init__(self):
        u1 = self.trainable_uv[0, :-1, :-1]
        v1 = self.trainable_uv[1, :-1, :-1]
        u2 = self.trainable_uv[0, 1:, :-1]
        v2 = self.trainable_uv[1, 1:, :-1]
        u3 = self.trainable_uv[0, 1:, 1:]
        v3 = self.trainable_uv[1, 1:, 1:]
        u4 = self.trainable_uv[0, :-1, 1:]
        v4 = self.trainable_uv[1, :-1, 1:]
        energy_map = torch.zeros_like(u1)
        for i in range(len(self.KBDB)):
            KBDB = self.KBDB[i]
            e1 = KBDB[0, 0]* u1 + KBDB[1, 0]* v1 + KBDB[2, 0]* u2 + KBDB[3, 0]* v2 + KBDB[4, 0]* u3 + KBDB[5, 0]* v3 + KBDB[6, 0]* u4 + KBDB[7, 0]* v4
            e2 = KBDB[0, 1]* u1 + KBDB[1, 1]* v1 + KBDB[2, 1]* u2 + KBDB[3, 1]* v2 + KBDB[4, 1]* u3 + KBDB[5, 1]* v3 + KBDB[6, 1]* u4 + KBDB[7, 1]* v4
            e3 = KBDB[0, 2]* u1 + KBDB[1, 2]* v1 + KBDB[2, 2]* u2 + KBDB[3, 2]* v2 + KBDB[4, 2]* u3 + KBDB[5, 2]* v3 + KBDB[6, 2]* u4 + KBDB[7, 2]* v4
            e4 = KBDB[0, 3]* u1 + KBDB[1, 3]* v1 + KBDB[2, 3]* u2 + KBDB[3, 3]* v2 + KBDB[4, 3]* u3 + KBDB[5, 3]* v3 + KBDB[6, 3]* u4 + KBDB[7, 3]* v4
            e5 = KBDB[0, 4]* u1 + KBDB[1, 4]* v1 + KBDB[2, 4]* u2 + KBDB[3, 4]* v2 + KBDB[4, 4]* u3 + KBDB[5, 4]* v3 + KBDB[6, 4]* u4 + KBDB[7, 4]* v4
            e6 = KBDB[0, 5]* u1 + KBDB[1, 5]* v1 + KBDB[2, 5]* u2 + KBDB[3, 5]* v2 + KBDB[4, 5]* u3 + KBDB[5, 5]* v3 + KBDB[6, 5]* u4 + KBDB[7, 5]* v4
            e7 = KBDB[0, 6]* u1 + KBDB[1, 6]* v1 + KBDB[2, 6]* u2 + KBDB[3, 6]* v2 + KBDB[4, 6]* u3 + KBDB[5, 6]* v3 + KBDB[6, 6]* u4 + KBDB[7, 6]* v4
            e8 = KBDB[0, 7]* u1 + KBDB[1, 7]* v1 + KBDB[2, 7]* u2 + KBDB[3, 7]* v2 + KBDB[4, 7]* u3 + KBDB[5, 7]* v3 + KBDB[6, 7]* u4 + KBDB[7, 7]* v4
            energy_map += e1 * u1 + e2 * v1 + e3 * u2+ e4 * v2 + e5 * u3+ e6 * v3 + e7 * u4 + e8 * v4

        return energy_map * self.zf * self.zf

    def __calc_strain__(self):
        u1 = self.trainable_uv[0, :-1, :-1]
        v1 = self.trainable_uv[1, :-1, :-1]
        u2 = self.trainable_uv[0, 1:, :-1]
        v2 = self.trainable_uv[1, 1:, :-1]
        u3 = self.trainable_uv[0, 1:, 1:]
        v3 = self.trainable_uv[1, 1:, 1:]
        u4 = self.trainable_uv[0, :-1, 1:]
        v4 = self.trainable_uv[1, :-1, 1:]
        strain_map_exx = torch.zeros_like(u1)
        strain_map_exy = torch.zeros_like(u1)
        strain_map_eyy = torch.zeros_like(u1)
        for i in range(len(self.KBDB)):
            B = self.B[i]
            strain_map_exx += B[0, 0] * u1 + B[0, 1] * v1 + B[0, 2] * u2 + B[0, 3] * v2 + B[0, 4] * u3 + B[0, 5] * v3 + B[0, 6] * u4 + B[0, 7] * v4
            strain_map_eyy += B[1, 0] * u1 + B[1, 1] * v1 + B[1, 2] * u2 + B[1, 3] * v2 + B[1, 4] * u3 + B[1, 5] * v3 + B[1, 6] * u4 + B[1, 7] * v4
            strain_map_exy += B[2, 0] * u1 + B[2, 1] * v1 + B[2, 2] * u2 + B[2, 3] * v2 + B[2, 4] * u3 + B[2, 5] * v3 + B[2, 6] * u4 + B[2, 7] * v4
        return strain_map_exx, strain_map_exy, strain_map_eyy



    def forward(self, img_def, calc_strain=False):
        # 灰度
        self.upsampled_uv = nn.functional.interpolate(self.trainable_uv.unsqueeze(0), size=self.imsize, mode='bicubic', align_corners=True)[0, :, :, :]
        warped_img = self.interpolator.interpolation(self.upsampled_uv[0, :, :] + self.grid_y, self.upsampled_uv[1, :, :] + self.grid_x, img_def, img_mode=False)
        # 位移光滑性约束
        if self.norm == 'Gauge':
            u_grads = self.gauge_conv(self.upsampled_uv[0, :, :].unsqueeze(0).unsqueeze(0))
            v_grads = self.gauge_conv(self.upsampled_uv[1, :, :].unsqueeze(0).unsqueeze(0))
            return warped_img, u_grads, v_grads
        elif self.norm == 'FE':
            energy = self.__calc_energy_init__()
            if calc_strain:
                exx, exy, eyy = self.__calc_strain__()
                return warped_img, energy, exx, exy, eyy
            return warped_img, energy, energy


