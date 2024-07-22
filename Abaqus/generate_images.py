import cv2
import numpy as np
import matplotlib.pyplot as plt
from Abaqus.inp_parser import inp_parser
import tqdm
from interpolation import interpolator
import torch

def assemble_element(node_list, element_list, dim=2):
    coord_list = []
    for tmp_nodes in element_list:
        tmp_coord = []
        for node_id in tmp_nodes:
            tmp_coord.append(node_list[node_id-1])
        if type(node_list) == type([]):
            coord_list.append(tmp_coord)
        else:
            coord_list.append(np.stack(tmp_coord))
    if type(node_list) == type([]):
        return np.array(coord_list).swapaxes(1, 2) # (N, 2, 4)
    else:
        return np.stack(coord_list, dim=0)  # (N, 2, 4)



def interpolation_results():
    inp_file = r'F:\DATASET\GDDIC\Abaqus/Crack/GDDIC_J2.inp'
    results_file = r'F:\DATASET\GDDIC\Abaqus/Crack/GDDIC_J2.csv'
    # concerned_points_x, concerned_points_y = np.meshgrid(np.linspace(-100, 100, 2000), np.linspace(-20, 20, 400))
    concerned_points_x, concerned_points_y = np.meshgrid(np.linspace(-10, 10, 2000), np.linspace(-10, 10, 2000))
    concerned_u = np.zeros_like(concerned_points_x)
    concerned_v = np.zeros_like(concerned_points_x)
    concerned_exx = np.zeros_like(concerned_points_x)
    concerned_exy = np.zeros_like(concerned_points_x)
    concerned_eyy = np.zeros_like(concerned_points_x)
    my_parser = inp_parser()
    all_nodes = my_parser.parser_nodes(inp_file)
    all_elements = my_parser.parser_elements(inp_file)
    coord_list = assemble_element(all_nodes, all_elements)
    print('Parser Done')

    # 找最近点坐标
    nearest_ele_id = []
    for i in tqdm.tqdm(range(concerned_points_x.shape[0])):
        for j in range(concerned_points_x.shape[1]):
            tmp_x = concerned_points_x[i, j]
            tmp_y = concerned_points_y[i, j]
            tmp_distance = np.sum((coord_list[:, 0, :] - tmp_x)**2 + (coord_list[:, 1, :] - tmp_y)**2, axis=-1)
            tmp_nearest_ele_id = np.argmin(tmp_distance)
            nearest_ele_id.append(tmp_nearest_ele_id)
    np.savetxt(r'F:\DATASET\GDDIC\Abaqus/Crack/nearest.csv', np.array(nearest_ele_id), delimiter=',')
    ### 后面从此开始
    nearest_ele_id = np.loadtxt(r'F:\DATASET\GDDIC\Abaqus/Crack/nearest.csv', delimiter=',')
    nearest_ele_id = np.reshape(nearest_ele_id, concerned_points_x.shape)
    results = np.loadtxt(results_file, skiprows=1, usecols=(4, 11, 12, 13, 14, 16, 17, 18, 20), delimiter=',')  # id, U, V, exx, eyy, exy, sxx, syy, sxy

    nodal_value_list = np.zeros(shape=(len(all_nodes), 8))
    for i in range(results.shape[0]):
        tmp_node_id = int(results[i, 0]) - 1
        nodal_value_list[tmp_node_id, :] = results[i, 1:]

    element_value_list = np.zeros(shape=(coord_list.shape[0], 4, 8))
    for i in range(element_value_list.shape[0]):
        tmp_nodes = all_elements[i]
        j = 0
        for nid in tmp_nodes:
            tmp_value = nodal_value_list[nid-1, :]
            element_value_list[i, j, :] = tmp_value
            j += 1

    # perform interpolation
    ele_coord_array = np.array(coord_list)
    x1 = ele_coord_array[:, 0, 0]
    x2 = ele_coord_array[:, 0, 1] - x1
    x3 = ele_coord_array[:, 0, 2] - x1
    x4 = ele_coord_array[:, 0, 3] - x1
    y1 = ele_coord_array[:, 1, 0]
    y2 = ele_coord_array[:, 1, 1] - y1
    y3 = ele_coord_array[:, 1, 2] - y1
    y4 = ele_coord_array[:, 1, 3] - y1

    inv = 1 / (-x2 * x3 * y3 * y4 + x2 * x4 * y4 * y3 + x2 * y2 * x3 * y4 - x2 * y2 * x4 * y3 - x3 * x4 * y4 * y2 + x3 * y3 * x4 * y2)
    inva = np.abs(inv)
    a_1 = (-x3 * y3 * y4 + x4 * y4 * y3) * inv
    a_2 = (x2 * y2 * y4 - x4 * y4 * y2) * inv
    a_3 = (-x2 * y2 * y3 + x3 * y3 * y2) * inv
    b_1 = (-x3 * x4 * y4 + x3 * y3 * x4) * inv
    b_2 = (x2 * x4 * y4 - x2 * y2 * x4) * inv
    b_3 = (-x2 * x3 * y3 + x2 * y2 * x3) * inv
    c_1 = (x3 * y4 - x4 * y3) * inv
    c_2 = (-x2 * y4 + x4 * y2) * inv
    c_3 = (x2 * y3 - x3 * y2) * inv

    for i in tqdm.tqdm(range(concerned_u.shape[0])):
        for j in range(concerned_u.shape[1]):
            tmp_ele_id = int(nearest_ele_id[i, j])
            nodal_coords = coord_list[tmp_ele_id]   # (2, 4)
            tmp_x = concerned_points_x[i, j] - nodal_coords[0, 0]
            tmp_y = concerned_points_y[i, j] - nodal_coords[1, 0]

            nodal_values = element_value_list[tmp_ele_id]   # (4, 8)
            a = a_1[tmp_ele_id] * (nodal_values[1, :] - nodal_values[0, :]) + \
                a_2[tmp_ele_id] * (nodal_values[2, :] - nodal_values[0, :]) + \
                a_3[tmp_ele_id] * (nodal_values[3, :] - nodal_values[0, :])

            b = b_1[tmp_ele_id] * (nodal_values[1, :] - nodal_values[0, :]) + \
                b_2[tmp_ele_id] * (nodal_values[2, :] - nodal_values[0, :]) + \
                b_3[tmp_ele_id] * (nodal_values[3, :] - nodal_values[0, :])

            c = c_1[tmp_ele_id] * (nodal_values[1, :] - nodal_values[0, :]) + \
                c_2[tmp_ele_id] * (nodal_values[2, :] - nodal_values[0, :]) + \
                c_3[tmp_ele_id] * (nodal_values[3, :] - nodal_values[0, :])
            if inva[tmp_ele_id] > 1e6:
                interpolated_values = nodal_values[0, :]
            else:
                interpolated_values = a * tmp_x + b * tmp_y + c * tmp_x*tmp_y + nodal_values[0, :]        # (8,)
            concerned_u[i, j] = interpolated_values[0]
            concerned_v[i, j] = interpolated_values[1]
            concerned_exx[i, j] = interpolated_values[2]
            concerned_eyy[i, j] = interpolated_values[3]
            concerned_exy[i, j] = interpolated_values[4]
    plt.imshow(concerned_exx, vmax=0.2e-2, vmin=-0.2e-3)
    plt.colorbar()
    plt.show()
    np.save(r'F:\DATASET\GDDIC\Abaqus/Crack/interpolated_values.npy', np.stack([concerned_u, concerned_v, concerned_exx, concerned_eyy, concerned_exy], axis=0))



def interpolation_results_sort():
    inp_file = r'F:\DATASET\GDDIC\Abaqus/Crack/GDDIC_J2.inp'
    results_file = r'F:\DATASET\GDDIC\Abaqus/Crack/GDDIC_J2.csv'
    # concerned_points_x, concerned_points_y = np.meshgrid(np.linspace(-100, 100, 2000), np.linspace(-20, 20, 400))
    concerned_points_x, concerned_points_y = np.meshgrid(np.linspace(-10, 10, 2001), np.linspace(-10, 10, 2001))
    concerned_u = np.zeros_like(concerned_points_x)
    concerned_v = np.zeros_like(concerned_points_x)
    concerned_exx = np.zeros_like(concerned_points_x)
    concerned_exy = np.zeros_like(concerned_points_x)
    concerned_eyy = np.zeros_like(concerned_points_x)
    my_parser = inp_parser()
    all_nodes = my_parser.parser_nodes(inp_file)
    all_elements = my_parser.parser_elements(inp_file)
    coord_list = assemble_element(all_nodes, all_elements)
    print('Parser Done')
    sorted_ids = np.argsort(coord_list[:, 0, 0])
    ele_length = coord_list.shape[0]
    # 找最近点坐标
    nearest_ele_id = []
    for i in tqdm.tqdm(range(concerned_points_x.shape[0])):
        for j in range(concerned_points_x.shape[1]):
            tmp_x = concerned_points_x[i, j]
            tmp_y = concerned_points_y[i, j]
            if j > 2:
                pos_s = int(max(pos - 1000, 0))
                pos_e = int(min(pos + 1000, ele_length))
                tmp_sorted_points = sorted_ids[pos_s:pos_e]
                selected_points = coord_list[tmp_sorted_points, :, :]
                tmp_distance = np.sum((selected_points[:, 0, :] - tmp_x) ** 2 + (selected_points[:, 1, :] - tmp_y) ** 2, axis=-1)
                pos = np.argmin(tmp_distance) + pos_s
                tmp_nearest_ele_id = sorted_ids[pos]
            else:
                tmp_distance = np.sum((coord_list[:, 0, :] - tmp_x)**2 + (coord_list[:, 1, :] - tmp_y)**2, axis=-1)
                tmp_nearest_ele_id = np.argmin(tmp_distance)
                pos = np.argwhere(sorted_ids==tmp_nearest_ele_id)
            nearest_ele_id.append(tmp_nearest_ele_id)
    np.savetxt(r'F:\DATASET\GDDIC\Abaqus/Crack/nearest.csv', np.array(nearest_ele_id), delimiter=',')
    ### 后面从此开始
    nearest_ele_id = np.loadtxt(r'F:\DATASET\GDDIC\Abaqus/Crack/nearest.csv', delimiter=',')
    nearest_ele_id = np.reshape(nearest_ele_id, concerned_points_x.shape)
    results = np.loadtxt(results_file, skiprows=1, usecols=(4, 11, 12, 13, 14, 16, 17, 18, 20), delimiter=',')  # id, U, V, exx, eyy, exy, sxx, syy, sxy

    nodal_value_list = np.zeros(shape=(len(all_nodes), 8))
    for i in range(results.shape[0]):
        tmp_node_id = int(results[i, 0]) - 1
        nodal_value_list[tmp_node_id, :] = results[i, 1:]

    element_value_list = np.zeros(shape=(coord_list.shape[0], 4, 8))
    for i in range(element_value_list.shape[0]):
        tmp_nodes = all_elements[i]
        j = 0
        for nid in tmp_nodes:
            tmp_value = nodal_value_list[nid-1, :]
            element_value_list[i, j, :] = tmp_value
            j += 1

    # perform interpolation
    ele_coord_array = np.array(coord_list)
    x1 = ele_coord_array[:, 0, 0]
    x2 = ele_coord_array[:, 0, 1] - x1
    x3 = ele_coord_array[:, 0, 2] - x1
    x4 = ele_coord_array[:, 0, 3] - x1
    y1 = ele_coord_array[:, 1, 0]
    y2 = ele_coord_array[:, 1, 1] - y1
    y3 = ele_coord_array[:, 1, 2] - y1
    y4 = ele_coord_array[:, 1, 3] - y1

    inv = 1 / (-x2 * x3 * y3 * y4 + x2 * x4 * y4 * y3 + x2 * y2 * x3 * y4 - x2 * y2 * x4 * y3 - x3 * x4 * y4 * y2 + x3 * y3 * x4 * y2)
    inva = np.abs(inv)
    a_1 = (-x3 * y3 * y4 + x4 * y4 * y3) * inv
    a_2 = (x2 * y2 * y4 - x4 * y4 * y2) * inv
    a_3 = (-x2 * y2 * y3 + x3 * y3 * y2) * inv
    b_1 = (-x3 * x4 * y4 + x3 * y3 * x4) * inv
    b_2 = (x2 * x4 * y4 - x2 * y2 * x4) * inv
    b_3 = (-x2 * x3 * y3 + x2 * y2 * x3) * inv
    c_1 = (x3 * y4 - x4 * y3) * inv
    c_2 = (-x2 * y4 + x4 * y2) * inv
    c_3 = (x2 * y3 - x3 * y2) * inv

    for i in tqdm.tqdm(range(concerned_u.shape[0])):
        for j in range(concerned_u.shape[1]):
            tmp_ele_id = int(nearest_ele_id[i, j])
            nodal_coords = coord_list[tmp_ele_id]   # (2, 4)
            tmp_x = concerned_points_x[i, j] - nodal_coords[0, 0]
            tmp_y = concerned_points_y[i, j] - nodal_coords[1, 0]

            nodal_values = element_value_list[tmp_ele_id]   # (4, 8)
            a = a_1[tmp_ele_id] * (nodal_values[1, :] - nodal_values[0, :]) + \
                a_2[tmp_ele_id] * (nodal_values[2, :] - nodal_values[0, :]) + \
                a_3[tmp_ele_id] * (nodal_values[3, :] - nodal_values[0, :])

            b = b_1[tmp_ele_id] * (nodal_values[1, :] - nodal_values[0, :]) + \
                b_2[tmp_ele_id] * (nodal_values[2, :] - nodal_values[0, :]) + \
                b_3[tmp_ele_id] * (nodal_values[3, :] - nodal_values[0, :])

            c = c_1[tmp_ele_id] * (nodal_values[1, :] - nodal_values[0, :]) + \
                c_2[tmp_ele_id] * (nodal_values[2, :] - nodal_values[0, :]) + \
                c_3[tmp_ele_id] * (nodal_values[3, :] - nodal_values[0, :])
            if inva[tmp_ele_id] > 1e6:
                interpolated_values = nodal_values[0, :]
            else:
                interpolated_values = a * tmp_x + b * tmp_y + c * tmp_x*tmp_y + nodal_values[0, :]        # (8,)
            concerned_u[i, j] = interpolated_values[0]
            concerned_v[i, j] = interpolated_values[1]
            concerned_exx[i, j] = interpolated_values[2]
            concerned_eyy[i, j] = interpolated_values[3]
            concerned_exy[i, j] = interpolated_values[4]
    plt.imshow(concerned_v, vmax=2e-2, vmin=-2e-2)
    plt.colorbar()
    plt.show()
    np.save(r'F:\DATASET\GDDIC\Abaqus/Crack/interpolated_values.npy', np.stack([concerned_u, concerned_v, concerned_exx, concerned_eyy, concerned_exy], axis=0))


def warping_img():
    init_results = np.load(r'F:\DATASET\GDDIC\Abaqus/Crack/interpolated_values.npy').astype('float32')
    concerned_points_x, concerned_points_y = torch.meshgrid(torch.linspace(2, 401, 400), torch.linspace(2, 2001, 2000))
    displacements = torch.from_numpy(init_results[:2, :, :] * 10.0)
    my_interpolator = interpolator(device='cpu')
    cur_img = torch.from_numpy(cv2.resize(cv2.imread(r'F:\DATASET\GDDIC\Abaqus/Crack/1.bmp', cv2.IMREAD_GRAYSCALE).astype('float32'), dsize=(2020, 506), interpolation=cv2.INTER_CUBIC))
    img_r = my_interpolator.interpolation(u_pos=concerned_points_x+displacements[1, :, :], v_pos=concerned_points_y+displacements[0, :, :] + 2.0, gray_array=cur_img, img_mode=False)
    cv2.imwrite(r'F:\DATASET\GDDIC\Abaqus/crack0.bmp', img_r.numpy().astype('uint8'))
    cv2.imwrite(r'F:\DATASET\GDDIC\Abaqus/crack1.tif', cur_img[2:402, 2:2002].numpy().astype('uint8'))


def warping_img_crack():
    init_results = np.load(r'F:\DATASET\GDDIC\Abaqus/Crack/interpolated_values.npy').astype('float32')
    concerned_points_x, concerned_points_y = torch.meshgrid(torch.linspace(3, 2003, 2001), torch.linspace(3, 2003, 2001))
    displacements = torch.from_numpy(init_results[:2, :, :] * 10.0)
    my_interpolator = interpolator(device='cpu')
    cur_img = torch.from_numpy(cv2.resize(cv2.imread(r'F:\DATASET\GDDIC\Abaqus/Crack/1.bmp', cv2.IMREAD_GRAYSCALE).astype('float32'), dsize=(2010, 2010), interpolation=cv2.INTER_CUBIC))
    cur_img[3+999:3+1002, 3+1000:] = 0
    displacements[1, 999:1002, 1000:] = 0
    displacements[0, 999:1002, 1000:] = 0
    img_r = my_interpolator.interpolation(u_pos=concerned_points_x-displacements[1, :, :], v_pos=concerned_points_y-displacements[0, :, :], gray_array=cur_img, img_mode=True)
    cv2.imwrite(r'F:\DATASET\GDDIC\Abaqus/Crack/crack0.bmp', img_r.numpy().astype('uint8'))
    cur_img = (cur_img - 255) * (cur_img < 255) + 255
    cv2.imwrite(r'F:\DATASET\GDDIC\Abaqus/Crack/crack1.bmp', cur_img[3:2004, 3:2004].numpy().astype('uint8'))


if __name__ == '__main__':
    # interpolation_results_sort()
    warping_img_crack()