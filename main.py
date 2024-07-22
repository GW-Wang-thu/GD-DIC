import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from GD_Net import GD_DIC
import copy
import os
from GMFlow.gmflow import GMFlow

results_path = r'E:\Data\DATASET\LargeDeformationDIC\Star'
comp_path = r'F:\DATASET\GDDIC\Abaqus\Maltimatedial'
crack_path = r'F:\DATASET\GDDIC\Abaqus\Crack'
label_path = r'F:\DATASET\StrainNet_LD\Dataset_Normal\Valid'


def generate_gt():
    x_coord = np.expand_dims(np.arange(1, 2001), 1).repeat(501, 1)
    y_coord = np.expand_dims(np.arange(0, 501), 0).repeat(2000, 0) - 250
    lamda = 10 + x_coord * 140 / 2000
    omega = 2 * np.pi / (lamda)
    v_label = 0.5 * np.cos(omega * y_coord)
    return v_label


def eval_results_line(results, bd):
    gt =generate_gt()
    mask = np.zeros_like(gt)
    mask[bd:-bd, bd:-bd] = 1
    i = 0
    line_error = []
    line_mean = []
    for result in results:
        i += 1
        tmp_mask = copy.deepcopy(mask)
        tmp_mask[np.abs(result - gt) > 0.5] = 0
        mae = np.sum(tmp_mask * np.abs(result - gt), 1) / (np.sum(tmp_mask, 1) + 1.0)
        plt.plot(mae, label=str(i))
        line_error.append(mae)
        line_mean.append(result[:, 250])
    plt.legend()
    plt.show()
    return line_error, line_mean


def show_img(mat_list, mask, vmax=None, vmin=None, title=None):
    num_x = len(mat_list)
    num_y = len(mat_list[0])
    plt.figure(figsize=(num_y*8, num_x*8+1))
    for i in range(num_x):
        for j in range(num_y):
            plt.subplot(num_x, num_y, i * num_y + j + 1)
            plt.imshow(mat_list[i][j] * mask, vmin=vmin[i] if vmin is not None else None, vmax=vmax[i] if vmin is not None else None, cmap='jet')
            plt.colorbar()
    plt.title(title if title is not None else '')
    plt.show()


def charbonnier_loss(output, target, mask):
    loss = torch.sum((1e-6 + (target - output) ** 2 * mask)**0.7)
    return loss


def calculate_star():
    device = 'cuda'
    norm_method = 'FE'

    ref_img = torch.from_numpy(cv2.imread('samples/star/star1.tif', cv2.IMREAD_GRAYSCALE).astype('float32') * 10 - 1000).to(device)
    cur_img = torch.from_numpy(cv2.imread('samples/star/star2.tif', cv2.IMREAD_GRAYSCALE).astype('float32') * 10 - 1000).to(device)
    v_lable  = generate_gt().T
    mask = np.zeros_like(ref_img)
    mask[:3:, :3:] = 1
    mask = torch.from_numpy(mask).to(device)
    my_GDDIC = GD_DIC(imsize=ref_img.shape, device=device, zf=5, norm_method='FE', gauge_size=3).to(device).train()

    if norm_method== 'Gauge':
        my_GDDIC.gauge_conv.requires_grad_(False)
    optimizer = torch.optim.AdamW(my_GDDIC.parameters(), lr=0.1e-3, weight_decay=1e-4)

    disp_v_tra = np.loadtxt(fname=os.path.join(results_path, '1_fftcc_icgn1_r' + str(8) + '_o' + str(2) + '_v.csv'),
                            delimiter=',', usecols=range(2000), dtype='float32')
    num_epoch = 1000

    for n in range(num_epoch):
        warped_def_img, u_grads, v_grads = my_GDDIC(cur_img)
        # loss = torch.sum(torch.abs(warped_def_img - ref_img)**1 * mask)
        gauge_weight = 0.8 * charbonnier_loss(warped_def_img, ref_img, mask).item() / (0.01 * charbonnier_loss(warped_def_img, ref_img, mask).item() + (torch.sum(torch.abs(u_grads)**2) * mask + torch.sum(torch.abs(v_grads)**2) * mask).item())
        loss = charbonnier_loss(warped_def_img, ref_img, mask) + gauge_weight * (torch.sum(torch.abs(u_grads) * mask) + torch.sum(torch.abs(v_grads) * mask))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n%100 == 99:
            warped_def_img, eng, exx, exy, eyy = my_GDDIC(cur_img, calc_strain=True)
            print("Epoch: %d, loss: %.2f"%(n, loss.item()))
            print("Epoch: %d, V MAE: %.5f"%(n, np.sum(np.abs(my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy() - v_lable)) / (ref_img.shape[0]*ref_img.shape[1])))
            # plt.imshow((ref_img - warped_def_img).detach().cpu().numpy())
            # plt.show()
            # plt.imshow(my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy(), cmap='jet', vmin=-0.5, vmax=0.5)
            _, _ = eval_results_line(results=[my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T, disp_v_tra.T], bd=8)
            show_img([[my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T, disp_v_tra.T],
                      [my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T - v_lable.T, disp_v_tra.T - v_lable.T],
                      [u_grads.detach().cpu().numpy().T, u_grads.detach().cpu().numpy().T],
                      [exx.detach().cpu().numpy().T, eyy.detach().cpu().numpy().T]],
                     vmin=[-0.5, -0.5, np.min(u_grads.detach().cpu().numpy()), None],
                     vmax=[0.5, 0.5, np.max(u_grads.detach().cpu().numpy()), None], mask=mask)


def calculate_comp():
    device = 'cuda'
    norm_method = 'FE'

    ref_img = torch.from_numpy(cv2.imread('samples/multimaterial/mm1.tif', cv2.IMREAD_GRAYSCALE).astype('float32') * 10 - 1000).to(device)
    cur_img = torch.from_numpy(cv2.imread('samples/multimaterial/mm2.tif', cv2.IMREAD_GRAYSCALE).astype('float32') * 10 - 1000).to(device)
    u_lable  = np.load(os.path.join(comp_path, 'interpolated_values.npy'))[1, :, :] * 10
    v_lable  = np.load(os.path.join(comp_path, 'interpolated_values.npy'))[0, :, :] * 10 + 2.0
    exx_lable  = np.load(os.path.join(comp_path, 'interpolated_values.npy'))[3, :, :]
    eyy_lable  = np.load(os.path.join(comp_path, 'interpolated_values.npy'))[2, :, :]
    # mask = 1
    mask = np.zeros_like(u_lable)
    mask[3:-3, 3:-3] = 1
    mask = torch.from_numpy(mask).to(device)
    my_GDDIC = GD_DIC(imsize=ref_img.shape, device=device, zf=2, norm_method='FE', gauge_size=5).to(device).train()

    if norm_method== 'Gauge':
        my_GDDIC.gauge_conv.requires_grad_(False)
    optimizer = torch.optim.AdamW(my_GDDIC.parameters(), lr=0.01, weight_decay=1e-4)

    disp_u_tra = np.loadtxt(fname=os.path.join(comp_path, 'mm2_fftcc_icgn1_r9_o2_v.csv'), delimiter=',', usecols=range(2000), dtype='float32')
    disp_v_tra = np.loadtxt(fname=os.path.join(comp_path, 'mm2_fftcc_icgn1_r9_o2_u.csv'), delimiter=',', usecols=range(2000), dtype='float32')
    num_epoch = 10000

    for n in range(num_epoch):
        warped_def_img, u_grads, v_grads = my_GDDIC(cur_img)
        # loss = torch.sum(torch.abs(warped_def_img - ref_img)**1 * mask)
        gauge_weight = 100 * charbonnier_loss(warped_def_img, ref_img, mask).item() / (0.01 * charbonnier_loss(warped_def_img, ref_img, mask).item() + (torch.sum(torch.abs(u_grads)) + torch.sum(torch.abs(v_grads))).item())
        loss = charbonnier_loss(warped_def_img, ref_img, mask) + gauge_weight * (torch.sum(torch.abs(u_grads))  + torch.sum(torch.abs(v_grads)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n%100 == 19:
            warped_def_img, eng, exx, exy, eyy = my_GDDIC(cur_img, calc_strain=True)
            print("Epoch: %d, loss: %.2f"%(n, loss.item()))
            print("Epoch: %d, V MAE: %.5f"%(n, np.sum(np.abs(my_GDDIC.upsampled_uv[1, :, :].detach().cpu().numpy() - v_lable)) / (ref_img.shape[0]*ref_img.shape[1])))
            # plt.imshow((ref_img - warped_def_img).detach().cpu().numpy())
            # plt.show()
            # plt.imshow(my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy(), cmap='jet', vmin=-0.5, vmax=0.5)
            # _, _ = eval_results_line(results=[my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T, disp_u_tra.T], bd=8)
            show_img([[my_GDDIC.upsampled_uv[1, :, :].detach().cpu().numpy().T, disp_v_tra.T],
                      [my_GDDIC.upsampled_uv[1, :, :].detach().cpu().numpy().T - v_lable.T, disp_v_tra.T - v_lable.T],
                      # [u_grads.detach().cpu().numpy().T, u_grads.detach().cpu().numpy().T],
                      # [u_grads[0, 0, :, :].detach().cpu().numpy().T, u_grads[0, 1, :, :].detach().cpu().numpy().T],
                      [eyy.detach().cpu().numpy().T, eyy_lable.T]
                      ],
                     vmin=[-2, -0.1, None, -0.002],
                     vmax=[2, 0.1, 0.1, 0.05], mask=1)


def calculate_crack():
    device = 'cuda'
    norm_method = 'FE'
    zf = 1
    ref_img = torch.from_numpy(cv2.imread('samples/crack/crack0.bmp', cv2.IMREAD_GRAYSCALE).astype('float32') * 10 - 1000).to(device)
    cur_img = torch.from_numpy(cv2.imread('samples/crack/crack1.bmp', cv2.IMREAD_GRAYSCALE).astype('float32') * 10 - 1000).to(device)
    u_lable  = -np.load(os.path.join(crack_path, 'interpolated_values.npy'))[1, :, :] * 10
    v_lable  = -np.load(os.path.join(crack_path, 'interpolated_values.npy'))[0, :, :] * 10
    exx_lable  = np.load(os.path.join(crack_path, 'interpolated_values.npy'))[3, :, :]
    eyy_lable  = np.load(os.path.join(crack_path, 'interpolated_values.npy'))[2, :, :]
    # mask = 1
    maskn = np.zeros_like(u_lable)
    maskn[2:-2, 2:-2] = 1
    maskn[998:1003, 1001:] = 0
    mask = torch.from_numpy(maskn).to(device)
    mask_rn = cv2.resize(maskn, dsize=(mask.shape[1]//zf, mask.shape[0]//zf), interpolation=cv2.INTER_CUBIC)[1:, 1:]
    mask_rescale = torch.from_numpy(mask_rn).to(device)
    my_GDDIC = GD_DIC(imsize=ref_img.shape, device=device, zf=zf, norm_method='FE', gauge_size=5).to(device).train()

    if norm_method== 'Gauge':
        my_GDDIC.gauge_conv.requires_grad_(False)
    optimizer = torch.optim.AdamW(my_GDDIC.parameters(), lr=0.01, weight_decay=1e-4)

    disp_u_tra = -np.loadtxt(fname=os.path.join(crack_path, 'crack1_fftcc_icgn1_r10_o2_v.csv'), delimiter=',', usecols=range(2001), dtype='float32')
    disp_v_tra = -np.loadtxt(fname=os.path.join(crack_path, 'crack1_fftcc_icgn1_r10_o2_u.csv'), delimiter=',', usecols=range(2001), dtype='float32')
    num_epoch = 10000

    for n in range(num_epoch):
        warped_def_img, u_grads, v_grads = my_GDDIC(cur_img)
        # loss = torch.sum(torch.abs(warped_def_img - ref_img)**1 * mask)
        gauge_weight = 1000 * charbonnier_loss(warped_def_img, ref_img, mask).item() / (0.01 * charbonnier_loss(warped_def_img, ref_img, mask).item() + (torch.sum(torch.abs(u_grads)) + torch.sum(torch.abs(v_grads))).item())
        loss = charbonnier_loss(warped_def_img, ref_img, mask) + gauge_weight * (torch.sum(torch.abs(u_grads) * mask_rescale) + torch.sum(torch.abs(v_grads) * mask_rescale))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n%100 == 19:
            warped_def_img, eng, exx, exy, eyy = my_GDDIC(cur_img, calc_strain=True)
            print("Epoch: %d, loss: %.2f"%(n, loss.item()))
            print("Epoch: %d, V MAE: %.5f"%(n, np.sum(np.abs(my_GDDIC.upsampled_uv[1, :, :].detach().cpu().numpy() - v_lable)) / (ref_img.shape[0]*ref_img.shape[1])))
            # plt.imshow((ref_img - warped_def_img).detach().cpu().numpy())
            # plt.show()
            # plt.imshow(my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy(), cmap='jet', vmin=-0.5, vmax=0.5)
            # _, _ = eval_results_line(results=[my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T, disp_u_tra.T], bd=8)
            show_img([
                      [my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T * maskn.T, disp_u_tra.T],
                      [my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T * maskn.T - u_lable.T * maskn.T, disp_u_tra.T - u_lable.T],
                      # [u_grads.detach().cpu().numpy().T, u_grads.detach().cpu().numpy().T],
                      # [u_grads[0, 0, :, :].detach().cpu().numpy().T, u_grads[0, 1, :, :].detach().cpu().numpy().T],
                      [-exx.detach().cpu().numpy().T * mask_rn.T, exx_lable.T]
                      ],
                     vmin=[-2, -0.1, -0.002, -0.002],
                     vmax=[2, 0.1, 0.05, 0.05], mask=1)
            plt.plot(-eyy.detach().cpu().numpy()[1000, :1000])
            plt.plot(eyy_lable[1000, :1000])
            plt.show()


def calculate_1065():
    device = 'cuda'
    norm_method = 'FE'
    zf = 8
    ref_img = torch.from_numpy(cv2.imread('samples/1065/1065_0.bmp', cv2.IMREAD_GRAYSCALE).astype('float32') / 128 - 1).to(device)
    cur_img = torch.from_numpy(cv2.imread('samples/1065/1065_1.bmp', cv2.IMREAD_GRAYSCALE).astype('float32') / 128 - 1).to(device)

    all_channels = np.load(os.path.join(label_path, str(1065) + '_img&disp.npy'))
    mask = all_channels[0, :, :] != 0
    label_v = all_channels[2, :, :]
    label_u = all_channels[3, :, :]

    imsize = ref_img.shape
    mask = 1
    GM_Matcher = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   inchannel=3,
                   ).to(device).eval()
    GM_Matcher.load_state_dict(torch.load('GMFlow/GMFlowNet_Best.pth'))
    # model.load_state_dict(pre_trained_paramesters['model'])
    pred_flow = GM_Matcher(ref_img.unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1).cuda(), cur_img.unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1).cuda(),
                      attn_splits_list=[2],
                      corr_radius_list=[-1],
                      prop_radius_list=[-1])
    uv = pred_flow['flow_preds'][0]
    uv = torch.stack([uv[:, 1, :, :], uv[:, 0, :, :]], dim=1)
    uv_resize = torch.nn.functional.interpolate(uv, (imsize[0]//zf, imsize[1]//zf), mode='bicubic', align_corners=False)[0, :, :, :]
    my_GDDIC = GD_DIC(imsize=ref_img.shape, device=device, zf=zf, norm_method='FE', gauge_size=3, init_disp=uv_resize).to(device).train()
    if norm_method== 'Gauge':
        my_GDDIC.gauge_conv.requires_grad_(False)
    optimizer = torch.optim.AdamW(my_GDDIC.parameters(), lr=0.1, weight_decay=1e-4)

    num_epoch = 1000
    for n in range(num_epoch):
        warped_def_img, u_grads, v_grads = my_GDDIC(cur_img)
        # loss = torch.sum(torch.abs(warped_def_img - ref_img)**1 * mask)
        gauge_weight = 0.8 * charbonnier_loss(warped_def_img, ref_img, mask).item() / (0.01 * charbonnier_loss(warped_def_img, ref_img, mask).item() + (torch.sum(torch.abs(u_grads)**2) * mask + torch.sum(torch.abs(v_grads)**2) * mask).item())
        loss = charbonnier_loss(warped_def_img, ref_img, mask) + gauge_weight * (torch.sum(torch.abs(u_grads)) * mask + torch.sum(torch.abs(v_grads)) * mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n%10 == 0:
            print("Epoch: %d, loss: %.2f"%(n, loss.item()))
            print("Epoch: %d, V MAE: %.5f"%(n, np.sum(np.abs(my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy() - label_u)) / (ref_img.shape[0]*ref_img.shape[1])))

            # plt.imshow((ref_img - warped_def_img).detach().cpu().numpy())
            # plt.show()
            plt.figure(figsize=(10,8))
            # plt.imshow(my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy(), cmap='jet', vmin=-0.5, vmax=0.5)
            show_img([[my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T, my_GDDIC.upsampled_uv[1, :, :].detach().cpu().numpy().T],
                      [label_v.T - my_GDDIC.upsampled_uv[0, :, :].detach().cpu().numpy().T, label_u.T - my_GDDIC.upsampled_uv[1, :, :].detach().cpu().numpy().T],
                      [u_grads.detach().cpu().numpy().T, u_grads.detach().cpu().numpy().T]], vmin=[-100, -5, np.min(u_grads.detach().cpu().numpy())], vmax=[100, 5, np.max(u_grads.detach().cpu().numpy())], mask=mask)


if __name__ == '__main__':
    # calculate_star()
    # calculate_comp()
    calculate_crack()
    # calculate_1065()
