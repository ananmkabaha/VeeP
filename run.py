
import os
import sys
from pickle import NONE
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, './util')
import matplotlib.pyplot as plt
import math
import numpy as np
import torchvision.transforms as transforms
import onnx
import onnxruntime
import pickle
from decimal import Decimal
import torch
import csv
import time
import argparse
from config import config
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
from onnx_translator import *
from optimizer import *
from datasets import *
from VeeP import *
from attack import *
from perturbations import *
from socket_interface import *
import warnings
import signal
from functools import partial
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .onnx')
    parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, or imagenet')
    parser.add_argument('--max_itr', type=str, default=config.max_itr, help='maximal number of iterations')
    parser.add_argument('--timeout', type=str, default=config.TimeOut, help='timeout in seconds')
    parser.add_argument('--M', type=str, default=config.M, help='history length')
    parser.add_argument('--pertubration_type', type=str, default=config.pertubration_type, help='perturbation type can be either brightness, saturation, hue, lightness, or brightness_and_contrast.')
    parser.add_argument('--gpus', type=str, default=config.gpus, help='list of gpus to use')
    parser.add_argument('--output_file', type=str, default=config.output_file, help='output file')
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.json = vars(args)
    
    netname = config.netname
    dataset = config.dataset
    M = config.M
    TimeOut = config.TimeOut
    max_itr = config.max_itr
    save_figures = config.save_figures
    output_file = config.output_file
    pertubration_type = config.pertubration_type
    gpus = ([int(i) for i in config.gpus.split(",")])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    number_of_workers = len(gpus)
    assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
    assert is_perturbation(pertubration_type), "perturbation type is not supported." 
    Dataset = Datasets(dataset)
    attack = Attack(pertubration_type)
    os.sched_setaffinity(0,cpu_affinity)
    veep = VeeP( netname, M, config.default_steps, pertubration_type, Dataset, gpus[0], TimeOut)
    network = veep.network
    socket_interface = Socket_interface(number_of_workers)
    signal.signal(signal.SIGINT, partial(signal_handler, socket_interface))
    if len(gpus)>1:
        socket_interface.initalize_servers(netname, dataset, max_itr, M, pertubration_type, gpus, TimeOut)
        socket_interface.initalize_clients(gpus)        
    [samples,labels] = Dataset.get_samples()
    
    results = ""
    for i, sample in enumerate(samples):
        start_time = time.time()
        image = np.float64(sample)/np.float64(255)
        label = int(labels[i])
        specLB = Dataset.normalize(image)
        specUB = Dataset.normalize(image)
        sensitivity,elapsed_time = network.test(specLB, specUB, int(label), True)
        if  sensitivity>0:
            target_delta = attack.attack(image,label,veep,Dataset)
            target_delta_split = target_delta/number_of_workers
            if len(gpus)>1:
                socket_interface.send_to_workers(i, number_of_workers, target_delta,target_delta_split)
                certified = socket_interface.listen_to_workers(number_of_workers,i,M,veep.is_2d)
                certified_delta = veep.compute_certified_delta(certified)
                socket_interface.reset()
                veep.reset()
            else:
                certified_delta,target_delta = veep.certify(image, label, Dataset, 0.0, target_delta, target_delta_split, max_itr,i)
                veep.reset()
                
            results += "i:"+str(i)+" ,certified:"+str(certified_delta)+",target:"+str(target_delta)+" ,time:"+str(time.time()-start_time)+"\n" 
            f = open(output_file, "w")
            f.write(results)
            f.close()    
            if save_figures:
                if Dataset.c == 1:
                    plt.imshow(image.reshape(Dataset.w,Dataset.h))
                    plt.savefig("./figures/"+str(i)+".png", bbox = 'tight', dpi = 800)
                    plt.clf()
                    [_, image_attacked] = veep.perturbations.Perturbe(image, certified_delta)
                    plt.imshow(image_attacked.reshape(Dataset.w,Dataset.h),vmin=0,vmax=1)
                    plt.savefig("./figures/"+str(i)+"_certified.png", bbox = 'tight', dpi = 800)
                    plt.clf()
                else:
                    plt.imshow(image.reshape(Dataset.w,Dataset.h,Dataset.c))
                    plt.savefig("./figures/"+str(i)+".png", bbox = 'tight', dpi = 800)
                    plt.clf()
                    [_, image_attacked] = veep.perturbations.Perturbe(image, certified_delta)
                    plt.imshow(image_attacked.reshape(Dataset.w,Dataset.h,Dataset.c),vmin=0,vmax=1)
                    plt.savefig("./figures/"+str(i)+"_certified.png", bbox = 'tight', dpi = 800)
                    plt.clf()

    print("Results:")
    print(results)   
    socket_interface.clean()
        
        
