
import csv
import math
import numpy as np
import os.path
import pickle

class Attack:
  def __init__(self, perturbation_type):
    self.precision_step = 0.02 if perturbation_type == "brightness_and_contrast" else 0.01
    self.is_slow_scan = True
  def attack(self,sample,label,veep,Dataset):
        print("Computing target diamater...")
        precision = np.copy(self.precision_step)
        if veep.is_2d and self.is_slow_scan:
            delta_2d = 0
            delta_t = veep.perturbations.max_perturbation
            itr = 0
            while delta_2d<veep.perturbations.max_perturbation:
                delta_1d = 0
                delta_t_ = veep.perturbations.max_perturbation
                while delta_1d <veep.perturbations.max_perturbation:
                    senstivity,_ = veep.get_senstivity(sample, label, [delta_1d, delta_2d],[0, 0] ,Dataset)
                    itr += 1
                    if senstivity<0:
                        if delta_1d == 0 and delta_2d == 0:
                           precision/=10
                           continue
                        break
                    else: 
                        delta_t_ = delta_1d
                    delta_1d += precision 
                delta_t = min(delta_t,delta_t_)
                if delta_2d> delta_t:
                    break                
                delta_2d += precision
        else:
            delta = 0
            delta_t = 0
            itr = 0
            while delta<veep.perturbations.max_perturbation:
               if veep.is_2d:
                   senstivity,_ = veep.get_senstivity(sample, label, [delta, delta],[0, 0] ,Dataset)
               else:
                   senstivity,_ = veep.get_senstivity(sample, label, delta, 0, Dataset)
               if senstivity<0:
                    if delta == 0:
                       precision/=10
                       continue
                    break
               else: 
                   delta_t = delta
               delta += precision
               itr += 1
        print("The target diamater is:",round(delta_t,2))  
        return round(delta_t,2)
              




