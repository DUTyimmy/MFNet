import cv2
import os


def pgt_visual():
    # fusion of PAMR and Pgt
    img_root = r'C:\Users\oip\Desktop\wsod\sc\pamr'
    new_pgt_root = r'C:\Users\oip\Desktop\wsod\sc\new_pgt'
    pgt_root = r'C:\Users\oip\Desktop\wsod\sc\pgt\sun_aidmyhpxgsherdax.png'
    pgt = cv2.imread(pgt_root)
    epoch = 8
    max_epoch = 8
    delta = (epoch / (max_epoch + 1e-5))**0.5

    file_list = os.listdir(img_root)

    for file in file_list:
        sal = cv2.imread(img_root+'/'+file)
        new_pgt = delta*sal + (1-delta)*pgt
        cv2.imwrite(new_pgt_root + '/' + file[:-4] + '.jpg', new_pgt)
    print('over')


def heatmap():
    # convert CAMs to heatmaps
    cam_root = r'C:\Users\oip\Desktop\wsod\dataset\cam_d'
    img_root = r'C:\Users\oip\Desktop\wsod\dataset\img'
    save_root = r'C:\Users\oip\Desktop\wsod\dataset'
    cam_list = os.listdir(cam_root)
    for cam in cam_list:

        gray_img = cv2.imread(cam_root +'/' + cam)
        ori_img = cv2.imread(img_root +'/' + cam[:-4] + '.jpg')
        heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        img_add = cv2.addWeighted(ori_img, 0.6, heat_img, 0.4, 0)

        cv2.imwrite(save_root + '/' + cam[:-4] + '_heatmap.png', img_add)

    print('over')


def convert_name():
    # Modify file name
    img_root = r'G:\Saliency maps of full\DGRL\ECSSD\NLDF'

    file_list = os.listdir(img_root)

    for file in file_list:
        img = cv2.imread(img_root+'/'+file)
        new_file = file[:-9]+'.png'
        cv2.imwrite(img_root+'/'+new_file, img)
        os.remove(img_root+'/'+file)
    print('over')
