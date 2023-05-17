#!/usr/bin/env python3
"""
Copyright 2023, Gideon Billings.
GL3D SIFT keypoint descriptor generation util.
"""
import os
import sys
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
from utils.io import read_kpt, read_cams
from utils.geom import undist_points
from tqdm import tqdm

def load_img(img_path):
    img = cv2.imread(img_path)
    img = img[..., ::-1]
    return img

def affine_to_cv(kpts, image_size):
    cv_kpts = []
    W = image_size[0]
    H = image_size[1]
    short_side = (float) (min(W, H));
    for i in range(kpts.shape[0]):
        (x, y) = (kpts[i, 2] * W/2 + W/2, kpts[i, 5] * H/2 + H/2)
        m_cos = kpts[i,0]*short_side
        m_sin = kpts[i,1]*short_side
        rad = np.sqrt(m_cos**2 + m_sin**2)
        patch_scale = 6.
        size = rad/patch_scale
        ori = 360. - (np.arccos(m_cos/rad) * 180./np.pi)
        # octave = 1 << 8
        octave = 0
        cv_kpts.append(cv2.KeyPoint(x, y, size, ori, 1, octave))
    return cv_kpts

def process_sequence(n, d):
    opencv_major =  int(cv2.__version__.split('.')[0])
    opencv_minor =  int(cv2.__version__.split('.')[1])

    if opencv_major == 4 and opencv_minor >= 5: 
        sift = cv2.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SIFT_create()

    img_list = glob.glob(os.path.join(d, 'undist_images/*.jpg'))

    if not os.path.exists(os.path.join(d, 'img_desc')):
        os.makedirs(os.path.join(d, 'img_desc'))

    cam_path = os.path.join(d, 'geolabel', 'cameras.txt')
    if os.path.exists(cam_path):
        cam_dict = read_cams(cam_path)
    else:
        return
        # continue

    for i, img_path in enumerate(img_list):
        name = os.path.splitext(os.path.basename(img_path))[0]
        kpt_path  = os.path.join(d, 'img_kpts', name+'.bin')
        if not os.path.exists(kpt_path):
            return
            # continue
        save_path = os.path.join(d, 'img_desc', name+'.bin')
        # if os.path.exists(save_path):
            # continue

        print('Processing image {} of {} in folder {}: {}\r'.format(i, len(img_list), n, save_path), flush=True)

        rgb = load_img(img_path)

        cam = cam_dict[float(name)]
        K = cam[0]
        t = cam[1]
        R = cam[2]
        dist = cam[3]
        ori_img_size = cam[4]

        kpts = read_kpt(kpt_path)
        kpts = undist_points(kpts, K, dist, ori_img_size)
        cv_kpts = affine_to_cv(kpts, rgb.shape)
        (cv_kpts, des) = sift.compute(rgb, cv_kpts)
        des = np.asarray(des)
        des.astype(np.float32).tofile(save_path)

def main(argv=None):  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(
                    prog='gen_cv_descs.py',
                    description='Generate SIFT descriptors for GL3D dataset')
    parser.add_argument('data_path')
    args = parser.parse_args()

    data_dirs = glob.glob(args.data_path + "/*")

    # if opencv_major == 4 and opencv_minor >= 5: 
    #     sift = cv2.SIFT_create()
    # else:
    #     sift = cv2.xfeatures2d.SIFT_create()

    pbar = tqdm(total=len(data_dirs), desc="total")

    executor = concurrent.futures.ProcessPoolExecutor(10)
    m = multiprocessing.Manager()
    futures = [executor.submit(process_sequence, n, d) for n, d in enumerate(data_dirs)]
    for _ in concurrent.futures.as_completed(futures):
        pbar.update(1)  # Increments counter
    concurrent.futures.wait(futures)

    pbar.close()

    # for n, d in enumerate(data_dirs):
    #     img_list = glob.glob(os.path.join(d, 'undist_images/*.jpg'))

    #     if not os.path.exists(os.path.join(d, 'img_desc')):
    #         os.makedirs(os.path.join(d, 'img_desc'))

    #     cam_path = os.path.join(d, 'geolabel', 'cameras.txt')
    #     if os.path.exists(cam_path):
    #         cam_dict = read_cams(cam_path)
    #     else:
    #         continue
        
    #     for i, img_path in enumerate(img_list):
    #         name = os.path.splitext(os.path.basename(img_path))[0]
    #         kpt_path  = os.path.join(d, 'img_kpts', name+'.bin')
    #         if not os.path.exists(kpt_path):
    #             continue
    #         save_path = os.path.join(d, 'img_desc', name+'.bin')
    #         # if os.path.exists(save_path):
    #             # continue

    #         print('Processing image {} of {}, in folder {} of {}: {}'.format(i, len(img_list), n, len(data_dirs), save_path))

    #         rgb = load_img(img_path)

    #         cam = cam_dict[float(name)]
    #         K = cam[0]
    #         t = cam[1]
    #         R = cam[2]
    #         dist = cam[3]
    #         ori_img_size = cam[4]

    #         kpts = read_kpt(kpt_path)
    #         kpts = undist_points(kpts, K, dist, ori_img_size)
    #         cv_kpts = affine_to_cv(kpts, rgb.shape)
    #         (cv_kpts, des) = sift.compute(rgb, cv_kpts)
    #         des = np.asarray(des)
    #         des.astype(np.float32).tofile(save_path)

if __name__ == '__main__':
    main()