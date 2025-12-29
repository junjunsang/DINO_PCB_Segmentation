#!/usr/bin/python
#

# python imports
from __future__ import print_function
import os
import glob
import sys
from multiprocessing import Pool # 멀티프로세싱 모듈

# [수정 1] 모듈 경로 추가를 최상단으로 이동 (Import 에러 방지)
sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'helpers')))

# 라이브러리 import
from imageio import imread, imsave
import numpy as np
from numpngw import write_png
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import shutil

# 로컬 모듈 import (경로 추가 후 실행)
from json2labelImg import json2labelImg
from json2instanceImg import json2instanceImg

# 전역 변수
args = None

# [수정 2] Windows 멀티프로세싱을 위한 초기화 함수 추가
def init_worker(shared_args):
    global args
    args = shared_args

def process_folder(fn):
    global args

    # args가 None일 경우를 대비한 방어 코드 (혹시 모를 오류 방지)
    if args is None:
        return

    dst = fn.replace("_polygons.json", "_label_new{}.png".format(args.id_type))

    # do the conversion
    try:
        json2labelImg(fn, dst, args.id_type)
    except:
        tqdm.write("Failed to convert: {}".format(fn))
        raise

    if args.instance:
        dst = fn.replace("_polygons.json",
                         "_instance{}s.png".format(args.id_type))

        # do the conversion
        # try:
        json2instanceImg(fn, dst, args.id_type)
        # except:
        #     tqdm.write("Failed to convert: {}".format(f))
        #     raise

    if args.color:
        # create the output filename
        dst = fn.replace("_polygons.json", "_labelColors.png")

        # do the conversion
        try:
            json2labelImg(fn, dst, 'color')
        except:
            tqdm.write("Failed to convert: {}".format(fn)) # 변수명 f -> fn으로 수정 (원본 코드 버그 수정)
            raise

    # if args.panoptic and args.instance:
        # panoptic_converter(f, out_folder, out_file)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--datadir', default="")
    parser.add_argument('--id-type', default='level3Id')
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--instance', type=bool, default=False)
    parser.add_argument('--panoptic', type=bool, default=False)
    parser.add_argument('--semisup_da', type=bool, default=False)
    parser.add_argument('--unsup_da', type=bool, default=False)
    parser.add_argument('--weaksup_da', type=bool, default=False)
    parser.add_argument('--num-workers', type=int, default=10)

    args = parser.parse_args()

    return args

# The main method
def main(args):
    
    if args.panoptic:
        args.instance = True
    
    # (이미 상단에서 처리했으므로 여기서는 생략 가능하지만, 원본 유지를 위해 남겨둠)
    sys.path.append(os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'helpers')))
        
    # how to search for all ground truth
    searchFine = os.path.join(args.datadir, "gtFine",
                              "*", "*", "*_gt*_polygons.json")

    # search files
    filesFine = glob.glob(searchFine)
    filesFine.sort()

    files = []#filesFine

    #for semi supervised domain adaptation, convert only selected images
    filesnew_semisup = []
    filesnewunsup = []
    if args.semisup_da:
        d_strat = list(pd.read_csv('./domain_adaptation/target/semi-supervised/selected_samples.csv',header=None)[0])
        d_strat = ["/".join(filenew.replace("_labellevel3Ids.png", "").split("/")[-3:]) for filenew in d_strat]
        print(d_strat)
        for fileold in filesFine:
            if "val/" not in fileold:
                searchstr = "/".join(fileold.replace("_polygons.json", "").split("/")[-3:])
                if searchstr in d_strat:
                    print(searchstr)
                    filesnew_semisup.append(fileold)
            else: filesnew_semisup.append(fileold)
        files = filesnew_semisup
    elif args.unsup_da or args.weaksup_da:    #for unsupervised domain adaptation, convert only val images
        for fileold in filesFine:
            if "val/" in fileold:
                filesnewunsup.append(fileold)
        files = filesnewunsup
    else: files = filesFine

    #print('args.semisup_da', args.semisup_da, len(files))
    if not files:
        tqdm.write(
            "Did not find any files. Please consult the README.")

    # a bit verbose
    tqdm.write(
        "Processing {} annotation files for Sematic/Instance Segmentation".format(len(files)))

    # iterate through files
    progress = 0
    tqdm.write("Progress: {:>3} %".format(
        progress * 100 / len(files)), end=' ')

    # [수정 3] Pool 초기화 시 initializer 사용 (args 전달)
    pool = Pool(args.num_workers, initializer=init_worker, initargs=(args,))
    
    # results = pool.map(process_pred_gt_pair, pairs)
    results = list(
        tqdm(pool.imap(process_folder, files), total=len(files)))
    pool.close()
    pool.join()

    if args.panoptic:
        from cityscape_panoptic_gt import panoptic_converter
        for split in ['train', 'val']:

            tqdm.write("Panoptic Segmentation {} split".format(split))
            folder_name = os.path.join(args.datadir, 'gtFine')
            output_folder = os.path.join(folder_name, split + "_panoptic")
            os.makedirs(output_folder, exist_ok=True)
            out_file = os.path.join(folder_name, split + "_panoptic.json")
            panoptic_converter(args.num_workers, os.path.join(
                folder_name, split), output_folder, out_file)


if __name__ == "__main__":
    args = get_args()
    main(args)