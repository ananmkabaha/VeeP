import socket
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
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .onnx')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, or imagenet')
parser.add_argument('--max_itr', type=str, default=config.max_itr, help=' maximal number of iterations')
parser.add_argument('--timeout', type=str, default=config.TimeOut, help='timeout in seconds')
parser.add_argument('--M', type=str, default=config.M, help='history length')
parser.add_argument('--pertubration_type', type=str, default=config.pertubration_type, help='perturbation type can be either brightness, saturation, hue, lightness, or brightness_and_contrast.')
parser.add_argument('--gpus', type=str, default=config.gpus, help='list of gpus to use')
args = parser.parse_args()
for k, v in vars(args).items():
    setattr(config, k, v)
config.json = vars(args)

netname = config.netname
dataset = config.dataset
M = int(config.M)
max_itr = int(config.max_itr)
pertubration_type = config.pertubration_type
gpus = ([int(i) for i in config.gpus.split(",")])
TimeOut = config.TimeOut
assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
assert is_perturbation(pertubration_type), "perturbation type is not supported." 

Dataset = Datasets(dataset)
attack = Attack(pertubration_type)
os.sched_setaffinity(0,cpu_affinity)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
veep = VeeP( netname, M, config.default_steps, pertubration_type, Dataset, gpus[0],TimeOut)
network = veep.network
[samples,labels] = Dataset.get_samples()
host = socket.gethostname()
port = 5000+int(gpus[0])
server_socket = socket.socket()
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host, port))
server_socket.listen(2)
conn, address = server_socket.accept()
print("Connection from: " + str(address))
data = ""
while data != "finish":
    data = conn.recv(1024).decode()
    if "data" in str(data):
        i = np.int(data.split(",")[1])
        delta = np.float64(data.split(",")[2])
        delta_t = np.float64(data.split(",")[3])
        delta_t_1d = np.float64(data.split(",")[4])
        image = np.float64(samples[i])/np.float64(255)
        label = int(labels[i])
        specLB = Dataset.normalize(image)
        specUB = Dataset.normalize(image)
        sensitivity,elapsed_time = network.test(specLB, specUB, int(label), True)
        print("Reading from:",str(address),"sample:",i,"label:",label,"start:",delta,"target:",delta_t,"gpu:",gpus[0])
        start_time = time.time()
        delta_p = 0
        if sensitivity>0:
            if veep.is_2d:
                delta_p, delta_t = veep.certify(image, label, Dataset, delta, delta_t_1d, delta_t, max_itr,i,conn)
            else:
                delta_p, delta_t = veep.certify(image, label, Dataset, delta, delta_t, 0, max_itr,i,conn)
                
        if veep.is_2d:
            conn.send(str("finished,"+str(delta)+"," +str(delta_p[0])+"," +str(delta_p[1])+\
                          "," +str(delta_t[0])+"," +str(delta_t[1])).encode())
        else:
            conn.send(str("finished delta_s:"+str(delta)+" ,delta_p:" +str(delta_p)+" ,delta_t:" +str(delta_t)).encode())
        veep.reset()
conn.close()

        
        
        
        
