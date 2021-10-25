import os
import sys
from multiprocessing import Pool, Manager
import xlsxwriter
from aa import *
from PIL import Image
import numpy as np
import glob
import copy
import skimage
np.seterr(divide='ignore', invalid='ignore')
import cv2
import time
import yaml
import warnings
warnings.filterwarnings("ignore")

def getOneImageInfo(im_path, ext, src_mag, tar_mag, P, D):
    name = im_path.split('/')[-1].replace(f'.{ext}', '')
    tmpD = {}
    img = Image.open(im_path)
    src_img = copy.deepcopy(img)
    scale = src_mag/tar_mag
    scale = round(scale)
    w,h = img.size
    img = img.resize((w//scale, h//scale), Image.BICUBIC)
    img = np.array(img).astype(np.uint8)
    tmask = getTissueMask(img)
    tmask_ratio = np.sum(tmask)/(tmask.shape[0]*tmask.shape[1])
    dark_tmask = getDarkTissueMask(img)
    dark_tmask_ratio = np.sum(dark_tmask)/(dark_tmask.shape[0]*dark_tmask.shape[1])
    scale2 = round(src_mag/2.5)
    img2 = src_img.resize((w//scale, h//scale), Image.BICUBIC)
    img2 = np.array(img2).astype(np.uint8)
    tmask2 = tmask.astype(np.uint8)
    tmask2 = cv2.resize(tmask2, (img2.shape[1], img2.shape[0]), cv2.INTER_LINEAR)
    tmask2 = tmask2.astype(np.bool)
    pen_model = trainModelPen('artifacts/pen_green.png', 'artifacts/pen_green_mask.png')
    pen_ratio = getPenMarking(img, tmask, pen_model)
    coverslip_edge_model = trainCoverslipEdge('artifacts/coverslip_edge.png', 'artifacts/coverslip_edge_mask.png')
    coverslip_edge_ratio = getCoverslipEdge(img, tmask, pen_model)
    d = {'Tissue Ratio': tmask_ratio, 'Dark Tissue Ratio': dark_tmask_ratio, 'Pen Ratio': pen_ratio, 'Coverslip Edge Ratio': coverslip_edge_ratio}
    tmpD = {**tmpD, **d}
    for p in P:
        if p.__name__ == 'getBlurryRegion':
            d = p(img2, tmask2) # 2.5x
        else:
            d = p(img, tmask)
            tmpD = {**tmpD, **d}
    D[name]=tmpD

if __name__ == '__main__':
    # workflows
    P = [
            getHE,
            getHueSaturation,
            getContrast,
            getSmoothness,
            getFatlikeTissue,
            getBlurryRegion,
            getBrightness,
        ]

    start = time.time()
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)
    for set_name, param in param_sets.items():
        print(set_name)
        print(param)
        image_dir = param.get('image_dir')
        image_ext = param.get('image_ext')
        src_mag = param.get('image_magnification', 40)
        tar_mag = param.get('processing_magnification', 1.25)
        save_xls_path = param.get('save_xls_path')
        ncpus = param.get('ncpus')
        manager = Manager()
        D = manager.dict()
        im_paths = glob.glob(f'{image_dir}/*.{image_ext}')
        pool = Pool(ncpus)
        for im_path in im_paths:
            pool.apply_async(getOneImageInfo, (im_path, image_ext, src_mag, tar_mag, P, D))
        pool.close()
        pool.join()

        os.makedirs(os.path.dirname(save_xls_path), exist_ok=True)
        workbook = xlsxwriter.Workbook(save_xls_path)
        worksheet = workbook.add_worksheet()
        mylist = []
        mylist.append(['name', *list(list(D.items())[0][1])])
        for k,v in D.items():
            tlist = [k, *list(v.values())]
            mylist.append(tlist)
        for i in range(len(mylist)):
            worksheet.write_row(i, 0, mylist[i])
        workbook.close()
    end = time.time()
    print('done! time:',(end-start)/60)
