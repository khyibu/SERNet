# -*- coding: utf-8 -*-
import os
import time
import matplotlib.image
import numpy as np
import cv2
from torch.nn import functional as F
import torch

def extract_boundary(mask):
    edges = cv2.Canny((mask).astype(np.uint8), 100, 200)
    return edges

def laplacian_edge(label):
    label = torch.tensor(label, dtype=torch.float32).type(torch.cuda.FloatTensor)
    laplacian_kernel = torch.tensor(
    [-1, -1, -1, -1, 8, -1, -1, -1, -1],
    dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
    boundary_targets = F.conv2d(label.unsqueeze(0).unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = np.squeeze(boundary_targets)
    boundary_tar = boundary_targets
    boundary_tar[boundary_tar > 0.1] = 1
    boundary_tar[boundary_tar <= 0.1] = 0
    boundary_tar = boundary_tar.squeeze(0).squeeze(0).detach().cpu().numpy()

    return boundary_tar


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def compute_iou(box1, box2):
    # 计算两个边界框的IoU
    intersection = np.logical_and(box1, box2)
    union = np.logical_or(box1, box2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def save_result(dir, dicts, note):
    default_dir = os.path.join(os.path.split(dir)[0], 'All_result.txt')
    file_handle = open(default_dir, mode='a', encoding='utf-8')
    file_handle.write('*******--- ' + note + ' ---*******' + '\n')
    for res_items in dicts:
        file_handle.write(res_items + ': ' + '{:.4f}'.format(dicts.get(res_items)) + ' || ')
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    file_handle.write('####' + now_time + '\n')
    file_handle.writelines('\n')
    file_handle.flush()
    file_handle.close()

    file_handle = open(dir, mode='a', encoding='utf-8')
    file_handle.write('*******--- ' + note + ' ---*******' + '\n')
    for res_items in dicts:
        file_handle.write(res_items + ': ' + '{:.4f}'.format(dicts.get(res_items)) + '\n')
        print(res_items + ': ' + '{:.4f}'.format(dicts.get(res_items)))
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    file_handle.write('####' + now_time + '\n' + '+++++++++++++++++++++++++++++++++++++++++\n')
    file_handle.writelines('\n')
    file_handle.flush()
    file_handle.close()

def save_result_single(dir, img_dir, dicts1, dicts2):
    file_handle = open(dir, mode='a', encoding='utf-8')
    file_handle.write('*******' + img_dir + ':\t')
    for i in range(len(dicts1)):
        file_handle.write(dicts1[i] + ':' + '{:.6f}'.format(dicts2[i]) + ' // ')

    file_handle.writelines('\n')
    file_handle.flush()
    file_handle.close()


def calculate_pixel(ori_dir, pre_dir, colors, mode_type, is_gray=False):
    ori_imgs = sorted(os.listdir(ori_dir), key=lambda x: os.path.splitext(x)[0])
    pre_imgs = sorted(os.listdir(pre_dir), key=lambda x: os.path.splitext(x)[0])

    num_imgs = len(ori_imgs)

    checkPointName = pre_imgs[0][-7:-4]

    num_list = np.zeros((len(colors), len(colors), num_imgs), dtype=np.uint32)  # 计算所的结果保存在数组中
    BIOUlist = []
    for i in range(num_imgs):
        ori_img_path = os.path.join(ori_dir, ori_imgs[i])
        pre_img_path = os.path.join(pre_dir, pre_imgs[i])
        ori_img = cv2.imdecode(np.fromfile(ori_img_path, dtype=np.uint8), mode_type)
        if is_gray:
            pre_img = cv2.imdecode(np.fromfile(pre_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            pre_img = pre_img * 255
        else:
            pre_img = cv2.imdecode(np.fromfile(pre_img_path, dtype=np.uint8), mode_type)
        # 提取边界
        if np.all(ori_img == 0):
            BIOUlist.append(0)
        else:
            pred_boundary = mask_to_boundary(pre_img)  # laplacian_edge  extract_boundary mask_to_boundary
            true_boundary = mask_to_boundary(ori_img)
            BIOU=compute_iou(pred_boundary, true_boundary)
            BIOUlist.append(BIOU)

        ori_width = ori_img.shape[1]
        ori_height = ori_img.shape[0]
        pre_width = pre_img.shape[1]
        pre_height = pre_img.shape[0]
        if not (ori_width, ori_height) == (pre_width, pre_height):
            print('影像与标签大小不一致，请检查后重试！')
            break

        # 计算结果数组数据
        for j in range(len(colors)):  # 第J个类别
            for l in range(len(colors)):  # 第L个类别

                calcu_img1 = np.zeros((ori_height, ori_width), dtype=np.uint8)  # 存放标准图中的第J个类别
                calcu_img2 = np.zeros((ori_height, ori_width), dtype=np.uint8)  # 存放预测图中的第L个类别
                calcu_img = np.zeros((ori_height, ori_width), dtype=np.uint8)  # 存放标准图和预测图对比结果
                if is_gray:
                    calcu_img1[(ori_img == colors[j])] = 1  # 标准图中类别为J的像元标注为1
                    calcu_img2[(pre_img == colors[l])] = 1  # 预测图中类别为L的像元标注为1
                else:
                    calcu_img1[(ori_img == colors[j]).all(axis=2)] = 1  # 标准图中类别为J的像元标注为1
                    calcu_img2[(pre_img == colors[l]).all(axis=2)] = 1  # 预测图中类别为L的像元标注为1
                calcu_img[(calcu_img1 + calcu_img2) == 2] = 1  # 选中像元相加为2，提取值为2的像元，标注为1

                num_pixels = np.sum(calcu_img)  # 计算所得个数
                num_list[j][l][i] = num_pixels  # 加入数组中，对应第J行，第L列，第I个图像（通道）

    num_matrix = num_list.sum(axis=2)  # 计算各个图像的总和

    return num_matrix, num_list, pre_imgs, checkPointName, BIOUlist


def calculate_BIOU(ori_dir, pre_dir, txt_dir, note='', colors=None, is_gray=False):
    if is_gray:
        mode_type = cv2.IMREAD_GRAYSCALE  #
        colors = [255, 0]
    else:
        mode_type = cv2.IMREAD_COLOR
        colors = [255, 0]

    matrix_num, list_num, list_img, checkPointName, BIOUlist = calculate_pixel(ori_dir, pre_dir, colors, mode_type, is_gray)

    num = len(colors)  # 类别的个数

    F1 = 0.0
    IoU = 0.0
    IoU1 = 0.0
    IoU2 = 0.0
    IoU3 = 0.0
    TP_FP_TN_FN = np.sum(matrix_num)
    TP_list = np.zeros((num,), dtype=np.uint64)
    TN_list = np.zeros((num,), dtype=np.uint64)
    FN_list = np.zeros((num,), dtype=np.uint64)
    FP_list = np.zeros((num,), dtype=np.uint64)
    Prc_list = np.zeros((num,), dtype=np.float64)
    Rec_list = np.zeros((num,), dtype=np.float64)
    IoU_list = np.zeros((num,), dtype=np.float64)
    Macro_F1_list = np.zeros((num,), dtype=np.float64)

    if is_gray:
        TP = matrix_num[0][0]  # 真实正类预测为正类positive，正确true 真正例
        TN = matrix_num[1][1]  # 真实负类预测为负类negative，正确true 真反例
        FP = matrix_num[1][0]  # 真实反类预测为正类positive，错误false 假正例
        FN = matrix_num[0][1]  # 真实正类预测为反类negative，错误false 假反例

        TP_TN = TP + TN
        TP_FP = TP + FP
        TP_FN = TP + FN
        Prc = TP / TP_FP
        Rec = TP / TP_FN
        IoU_list[0] = TP / (TP_FP + FN)
        IoU_list[1] = TN / (TN + FP + FN)
        IoU = IoU_list[0]
        OA = TP_TN / TP_FP_TN_FN
        F1 = 2 * Prc * Rec / (Prc + Rec)
        Micro_F1 = 0.0
        Macro_F1 = 0.0
    else:
        for i in range(num):
            TP_list[i] = matrix_num[i][i]
            FN_list[i] = np.sum(matrix_num[i, :]) - TP_list[i]
            FP_list[i] = np.sum(matrix_num[:, i]) - TP_list[i]
            TN_list[i] = TP_FP_TN_FN - TP_list[i] - FP_list[i] - FN_list[i]
            Prc_list[i] = TP_list[i] / (TP_list[i] + FP_list[i])
            Rec_list[i] = TP_list[i] / (TP_list[i] + FN_list[i])
            Macro_F1_list[i] = 2 * Prc_list[i] * Rec_list[i] / (Prc_list[i] + Rec_list[i])
            IoU_list[i] = TP_list[i] / (TP_list[i] + FP_list[i] + FN_list[i])
        TP = np.sum(TP_list)
        TN = np.sum(TN_list)
        FP = np.sum(FP_list)
        FN = np.sum(FN_list)
        Prc = TP / (TP + FP)
        Rec = TP / (TP + FN)
        Micro_F1 = 2 * Prc * Rec / (Prc + Rec)
        Macro_F1 = np.average(Macro_F1_list)
        IoU1 = IoU_list[0]
        IoU2 = IoU_list[1]
        IoU3 = IoU_list[2]
        OA = np.sum(np.diagonal(matrix_num)) / TP_FP_TN_FN

    BIOU= np.average(BIOUlist)
    MIoU = np.average(IoU_list)
    Pe = np.dot(matrix_num.sum(axis=0), matrix_num.sum(axis=1) / 1e10)
    Pe = Pe / TP_FP_TN_FN / TP_FP_TN_FN * 1e10
    Po = OA
    Kappa = (Po - Pe) / (1 - Pe)

    if is_gray:
        dicts = {
            '  OA   ': OA,
            '  MIoU ': MIoU,
            '  IoU  ': IoU,
            '  Kappa': Kappa,
            '  F1   ': F1,
            '  P    ': Prc,
            '  R    ': Rec,
            ' BIou  ': BIOU,
            ' TP  ': matrix_num[0][0],  # 真实正类预测为正类positive，正确true 真正例
            ' TN  ': matrix_num[1][1],  # 真实负类预测为负类negative，正确true 真反例
            ' FP  ': matrix_num[1][0],  # 真实反类预测为正类positive，错误false 假正例
            ' FN  ': matrix_num[0][1],  # 真实正类预测为反类negative，错误false 假反例
            # ' matrix_num  ': {matrix_num}
        }
    else:
        dicts = {
            ' OA   ': OA,
            ' MIoU ': MIoU,
            ' IoU1 ': IoU1,
            ' IoU2 ': IoU2,
            ' IoU3 ': IoU3,
            ' Kappa': Kappa,
            ' Mi_F1': Micro_F1,
            ' Ma_F1': Macro_F1,
            ' P    ': Prc,
            ' R    ': Rec,
            ' BIou  ': BIOU,
        }

    note = os.path.split(pre_dir)[1] + ' && ' + checkPointName
    txt_dir = txt_dir + '（' + checkPointName + '）' + '.txt'
    save_result(txt_dir, dicts, note)
    for i in range(list_num.shape[2]):
        TP_FP_TN_FN = np.sum(list_num[:, :, i])
        new_list_num = list_num[:, :, i]
        if is_gray:
            TP = list_num[0][0][i]  # 真实正类预测为正类positive，正确true 真正例
            TN = list_num[1][1][i]  # 真实负类预测为负类negative，正确true 真反例
            FP = list_num[1][0][i]  # 真实反类预测为正类positive，错误false 假正例
            FN = list_num[0][1][i]  # 真实正类预测为反类negative，错误false 假反例
            TP_TN = TP + TN
            TP_FP = TP + FP
            TP_FN = TP + FN
            Prc = TP / TP_FP
            Rec = TP / TP_FN
            IoU_list[0] = TP / (TP_FP + FN)
            IoU_list[1] = TN / (TN + FP + FN)
            IoU = IoU_list[0]
            OA = TP_TN / TP_FP_TN_FN
            F1 = 2 * Prc * Rec / (Prc + Rec)

        else:
            for j in range(num):
                TP_list[j] = new_list_num[j][j]
                FP_list[j] = np.sum(new_list_num[:, j]) - TP_list[j]
                FN_list[j] = np.sum(new_list_num[j, :]) - TP_list[j]
                TN_list[j] = TP_FP_TN_FN - TP_list[j] - FP_list[j] - FN_list[j]
                Prc_list[j] = TP_list[j] / (TP_list[j] + FP_list[j])
                Rec_list[j] = TP_list[j] / (TP_list[j] + FN_list[j])
                Macro_F1_list[j] = 2 * Prc_list[j] * Rec_list[j] / (Prc_list[j] + Rec_list[j])
                IoU_list[j] = TP_list[j] / (TP_list[j] + FP_list[j] + FN_list[j])
            TP = np.sum(TP_list)
            TN = np.sum(TN_list)
            FP = np.sum(FP_list)
            FN = np.sum(FN_list)
            Prc = TP / (TP + FP)
            Rec = TP / (TP + FN)
            Micro_F1 = 2 * Prc * Rec / (Prc + Rec)
            Macro_F1 = np.average(Macro_F1_list)
            IoU1 = IoU_list[0]
            IoU2 = IoU_list[1]
            IoU3 = IoU_list[2]
            OA = np.sum(np.diagonal(new_list_num)) / TP_FP_TN_FN  # 对角线上的元素/总元素

        MIoU = np.average(IoU_list)
        BIOU = np.average(BIOUlist)
        Pe = np.dot(new_list_num.sum(axis=0), new_list_num.sum(axis=1) / 1e10)
        Pe = Pe / TP_FP_TN_FN / TP_FP_TN_FN * 1e10  # 用于计算kappa
        Po = OA  # 用于计算Kappa
        Kappa = (Po - Pe) / (1 - Pe)
        img_dir = list_img[i]

        if is_gray:
            lists1 = ['OA', 'IoU', 'MIoU', 'Kappa', 'F1', 'Prec', 'Reca', ' BIou']
            lists2 = [OA, IoU, MIoU, Kappa, F1, Prc, Rec, BIOU]
        else:
            lists1 = ['OA', 'IoU1', 'IoU2', 'IoU2', 'MIoU', 'Kappa', 'Micro_F1', 'Macro_F1', 'Prec', 'Reca']
            lists2 = [OA, IoU1, IoU2, IoU3, MIoU, Kappa, Micro_F1, Macro_F1, Prc, Rec]

        save_result_single(txt_dir, img_dir, lists1, lists2)

    print('计算完成！')

def check_size():
    pass

if __name__ == '__main__':
    ori_dir = r''
    pre_dir = r''
    now_time = time.strftime('%Y%m%d%H%M%S')
    txt_name = os.path.split(pre_dir)[1] + now_time
    txt_dir = os.path.join(r'', txt_name)
    calculate_BIOU(ori_dir, pre_dir, txt_dir, is_gray=True)





