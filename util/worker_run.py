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

parser = argparse.ArgumentParser(description='',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .onnx')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, or imagenet')
parser.add_argument('--max_itr', type=str, default=config.max_itr, help=' maximal number of iterations')
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

assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
assert is_perturbation(pertubration_type), "perturbation type is not supported." 

Dataset = Datasets(dataset)
attack = Attack()
os.sched_setaffinity(0,cpu_affinity)
veep = VeeP( netname, M, config.default_steps, pertubration_type, Dataset, gpus[0])
network = veep.network
[samples,labels] = Dataset.get_samples()

host = socket.gethostname()
port = 5000+int(gpus[0])
server_socket = socket.socket()
server_socket.bind((host, port))
server_socket.listen(2)
conn, address = server_socket.accept()
print("Connection from: " + str(address))
while True:
    data = conn.recv(1024).decode()
    if not data:
        break
    print("from connected user: " + str(data))
    

conn.close()







for i, sample in enumerate(samples):
    image = np.float64(sample)/np.float64(255)
    label = int(labels[i])
    specLB = Dataset.normalize(image)
    specUB = Dataset.normalize(image)
    sensitivity,elapsed_time = network.test(specLB, specUB, int(label), True)
    delta = 0
    if  sensitivity>0:
        target_delta = attack.attack(image,label,veep,Dataset) 
        veep.certify(image, label, Dataset, delta, target_delta, max_itr)
        
        
        
        