import os
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

def one(im_path, ext, src_mag, tar_mag, P, D):
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
    #Image.fromarray(dark_tmask*255).save('/tmp/test.png')
    #return
    #pen_red_model = trainModelPen('pen_red.png', 'pen_red_mask.png')
    #pen_green_model = trainModelPen('pen_green.png', 'pen_green_mask.png')
    pen_model = trainModelPen('pen_green.png', 'pen_green_mask.png')
    #red_ratio = getPenMarking(img, tmask, pen_red_model)
    #green_ratio = getPenMarking(img, tmask, pen_green_model)
    pen_ratio = getPenMarking(img, tmask, pen_model)
    coverslip_edge_model = trainCoverslipEdge('coverslip_edge.png', 'coverslip_edge_mask.png')
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
    base_dir = '/mnt/md0/_datasets/OralCavity/im_state'
    P = [
            getHE,
            #getHueSaturation,
            #getContrast,
            #getSmoothness,
            #getFatlikeTissue,
            #getBlurryRegion,
            #getBrightness,
        ]

    #test
    #manager = Manager()
    #D = manager.dict()
    #ext = 'png'
    #im_paths = glob.glob(f'{base_dir}/test/*.{ext}')
    #pool = Pool(20)
    #for im_path in im_paths:
    #    #pool.apply_async(one, (im_path, ext, 1.25, 1.25, P, D))
    #    one(im_path, ext, 1.5, 1.5, P, D)
    #pool.close()
    #pool.join()

    #workbook = xlsxwriter.Workbook(f'{base_dir}/test.xlsx')
    #worksheet = workbook.add_worksheet('SheetName')
    #mylist = []
    #mylist.append(['name', *list(list(D.items())[0][1])])
    #for k,v in D.items():
    #    tlist = [k, *list(v.values())]
    #    mylist.append(tlist)
    #for i in range(len(mylist)):
    #    worksheet.write_row(i, 0, mylist[i])
    #workbook.close()

    start = time.time()
    #D1
    manager = Manager()
    D = manager.dict()
    ext = 'jpg'
    cohort = 'D1'
    im_paths = glob.glob(f'{base_dir}/{cohort}/*.{ext}')
    pool = Pool(20)
    for im_path in im_paths[:1]:
        #pool.apply_async(one, (im_path, ext, 40, 1.25, P, D))
        one(im_path, ext, 40, 1.5, P, D)
    pool.close()
    pool.join()

    workbook = xlsxwriter.Workbook(f'{base_dir}/{cohort}.xlsx')
    worksheet = workbook.add_worksheet(cohort)
    mylist = []
    mylist.append(['name', *list(list(D.items())[0][1])])
    for k,v in D.items():
        tlist = [k, *list(v.values())]
        mylist.append(tlist)
    for i in range(len(mylist)):
        worksheet.write_row(i, 0, mylist[i])
    workbook.close()
    #D2
    manager = Manager()
    D = manager.dict()
    ext = 'tif'
    cohort = 'D2'
    im_paths = glob.glob(f'{base_dir}/{cohort}/*.{ext}')
    pool = Pool(20)
    for im_path in im_paths[:1]:
        #pool.apply_async(one, (im_path, ext, 40, 1.25, P, D))
        one(im_path, ext, 40, 1.5, P, D)
    pool.close()
    pool.join()

    #workbook = xlsxwriter.Workbook(f'{base_dir}/{cohort}.xlsx')
    worksheet = workbook.add_worksheet(cohort)
    mylist = []
    mylist.append(['name', *list(list(D.items())[0][1])])
    for k,v in D.items():
        tlist = [k, *list(v.values())]
        mylist.append(tlist)
    for i in range(len(mylist)):
        worksheet.write_row(i, 0, mylist[i])
    #workbook.close()
    #D3
    manager = Manager()
    D = manager.dict()
    ext = 'png'
    cohort = 'D3'
    im_paths = glob.glob(f'{base_dir}/{cohort}/*.{ext}')
    pool = Pool(20)
    for im_path in im_paths[:1]:
        #pool.apply_async(one, (im_path, ext, 2.5, 1.25, P, D))
        one(im_path, ext, 40, 1.5, P, D)
    pool.close()
    pool.join()

    #workbook = xlsxwriter.Workbook(f'{base_dir}/{cohort}.xlsx')
    worksheet = workbook.add_worksheet(cohort)
    mylist = []
    mylist.append(['name', *list(list(D.items())[0][1])])
    for k,v in D.items():
        tlist = [k, *list(v.values())]
        mylist.append(tlist)
    for i in range(len(mylist)):
        worksheet.write_row(i, 0, mylist[i])
    #workbook.close()
    #D4
    manager = Manager()
    D = manager.dict()
    ext = 'png'
    cohort = 'D4'
    im_paths = glob.glob(f'{base_dir}/{cohort}/*.{ext}')
    pool = Pool(20)
    for im_path in im_paths[:1]:
        #pool.apply_async(one, (im_path, ext, 2.5, 1.25, P, D))
        one(im_path, ext, 40, 1.5, P, D)
    pool.close()
    pool.join()

    #workbook = xlsxwriter.Workbook(f'{base_dir}/{cohort}.xlsx')
    worksheet = workbook.add_worksheet(cohort)
    mylist = []
    mylist.append(['name', *list(list(D.items())[0][1])])
    for k,v in D.items():
        tlist = [k, *list(v.values())]
        mylist.append(tlist)
    for i in range(len(mylist)):
        worksheet.write_row(i, 0, mylist[i])
    #workbook.close()
    #D5
    manager = Manager()
    D = manager.dict()
    ext = 'png'
    cohort = 'D5'
    im_paths = glob.glob(f'{base_dir}/{cohort}/*.{ext}')
    pool = Pool(20)
    for im_path in im_paths[:1]:
        #pool.apply_async(one, (im_path, ext, 2.5, 1.25, P, D))
        one(im_path, ext, 40, 1.5, P, D)
    pool.close()
    pool.join()

    #workbook = xlsxwriter.Workbook(f'{base_dir}/{cohort}.xlsx')
    worksheet = workbook.add_worksheet(cohort)
    mylist = []
    mylist.append(['name', *list(list(D.items())[0][1])])
    for k,v in D.items():
        tlist = [k, *list(v.values())]
        mylist.append(tlist)
    for i in range(len(mylist)):
        worksheet.write_row(i, 0, mylist[i])
    workbook.close()
    end = time.time()
    print('time:',(end-start)/60)
