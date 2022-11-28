
import csv
import numpy as np
import os.path
import pickle

class Datasets:
  def __init__(self, dataset,means=[],stds=[]):
    assert os.path.exists("./data/"+dataset+"_test.p"), dataset+" is not suppourted" 
    self.dataset =  dataset

    [self.images,self.labels] = pickle.load(open("./data/"+dataset+"_test.p", "rb"))

    if self.dataset == 'mnist':
        self.w, self.h, self.c = 28, 28, 1
    elif self.dataset == 'cifar10':
        self.w, self.h, self.c = 32, 32, 3
    elif self.dataset == 'imagenet':
        self.w, self.h, self.c = 224, 224, 3

    if len(means)==0:
        if dataset == 'mnist':
            means = [0]
            stds = [1]
        elif dataset == "cifar10":
            means = [0.4914, 0.4822, 0.4465]
            stds = [0.2023, 0.1994, 0.2010]
        elif dataset == 'imagenet':
            means=[0.485, 0.456, 0.406]
            stds=[0.229, 0.224, 0.225]
    self.means = means
    self.stds = stds

  def get_samples(self):
    return [self.images,self.labels]

  def normalize(self,image_in):
    image = np.copy(image_in)
    if self.dataset == 'mnist':
        for i in range(len(image)):
            image[i] = (image[i] - self.means[0])/self.stds[0]
    elif self.dataset=='cifar10' or self.dataset=='imagenet':
        count = 0
        tmp = np.zeros(self.w*self.h*self.c)
        for i in range(self.w*self.h):
            tmp[count] = (image[count] - self.means[0])/self.stds[0]
            count = count + 1
            tmp[count] = (image[count] - self.means[1])/self.stds[1]
            count = count + 1
            tmp[count] = (image[count] - self.means[2])/self.stds[2]
            count = count + 1  
        count = 0
        for i in range(self.w*self.h):
            image[i] = tmp[count]
            count = count+1
            image[i+self.w*self.h] = tmp[count]
            count = count+1
            image[i+2*self.w*self.h] = tmp[count]
            count = count+1
    return image




