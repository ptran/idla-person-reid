#!/usr/bin/env python

import argparse
import os
import h5py
from PIL import Image
import numpy as np


def write_dataset(mat_file, dataset_name, output_dir):
    """ Write dataset to file

        Parameters
        ----------
        mat_file : h5py.File
            input mat file object
        dataset_name : str
            dataset name
        output_dir : str
            base output directory
    """
    poffset = 0
    #create output dataset directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.mkdir(dataset_dir)
    #iterate through dataset image subsets
    dataset = mat_file[dataset_name][0]
    for objref in dataset:
        for v, view in enumerate(mat_file[objref]):
            for p, imgref in enumerate(view):
                #load image
                arr = np.array(mat_file[imgref])
                if len(arr.shape) is not 3:
                    continue
                img = Image.fromarray(arr.transpose(2, 1, 0))
                #save image with format '[person_id]_[view_id].jpg'
                image_path = os.path.join(dataset_dir, "{:04d}_{:02}.jpg".format(p+poffset, v))
                img.save(image_path)
        #keep track of person ID offset
        poffset += mat_file[objref].shape[-1]


def write_testsets(mat_file, output_dir):
    """ Write testsets to file

        Parameters
        ----------
        mat_file : h5py.File
            input mat file object
        output_dir : str
            base output directory
    """
    poffset = 0
    dataset = mat_file['detected'][0]
    mapping = []
    for ref in dataset:
        num_pids = mat_file[ref].shape[-1]
        subset = [i for i in range(poffset, num_pids+poffset)]
        poffset += num_pids
        mapping.append(subset)

    output_file = os.path.join(output_dir, 'testsets.csv')
    with open(output_file, 'w') as fid:
        testsets = mat_file['testsets'][0]
        for ref in testsets:
            testset = mat_file[ref]
            #map testset indices to person IDs
            mapped_testset = []
            for i, j in zip(testset[0], testset[1]):
                i = int(i)-1
                j = int(j)-1
                mapped_testset.append(mapping[i][j])
            #write person IDs to file
            for pid in mapped_testset[:-1]:
                fid.write("{},".format(pid))
            fid.write("{}\n".format(mapped_testset[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unpack CUHK03 mat file into directory structure.')
    parser.add_argument('mat_path', help='path to cuhk-03.mat file')
    args = parser.parse_args()

    mat_path = args.mat_path
    if not os.path.exists(mat_path):
        print("'{}' could not be found".format(mat_path))
        exit()

    output_dir = os.path.join('data', 'cuhk03')
    if os.path.exists(output_dir):
        print("'{}' already exists".format(output_dir))
        exit()
    os.makedirs(output_dir)

    #open mat file
    print("Loading data from '{}'...".format(mat_path))
    mat_file = h5py.File(mat_path, 'r')

    #write datasets
    print("Unpacking data to '{}'...".format(output_dir))
    write_dataset(mat_file, 'detected', output_dir)
    write_dataset(mat_file, 'labeled', output_dir)

    #write test indices to file
    write_testsets(mat_file, output_dir)
    print("Done.")
