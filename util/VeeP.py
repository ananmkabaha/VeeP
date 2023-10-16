
import csv
import numpy as np
import os.path
import pickle
import math
from onnx_translator import *
from optimizer import *
from perturbations import *
import select 
import time 
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class VeeP:
    def __init__(self, netname, M, default_steps, perturbation_type, Dataset,gpu_to_use, TimeOut):
        model, _ = read_onnx_net(netname)
        operations, resources = ONNXTranslator(model, True).translate()
        self.network, _, _, _, _ = Optimizer(operations, resources).get_gpupoly(layers(),gpu_to_use)
        self.sensitivity = np.zeros(M)
        self.sensitivity_zero = np.zeros(M)
        self.sensitivity_deltas = np.zeros(M)      
        self.velocity = np.zeros(M)
        self.velocity_deltas = np.zeros(M) 
        self.default_steps  = default_steps
        self.perturbations = Perturbations(Dataset.w, Dataset.h, Dataset.c, perturbation_type)
        self.M = M
        self.TimeOut = TimeOut
        self.approx_constant = 1000
        self.delta_min = 1e-4
        self.prm_vec = np.zeros(M)
        self.toggle = False 
        self.power_low = 200
        self.power_low_min = 200
        self.power_high = 6000
        self.power_high_max = 6000
        self.prm_band = 400
        self.increase_backoff = 0
        self.FA = 0
        self.gpu_to_use = gpu_to_use
        self.is_2d = True if perturbation_type == "brightness_and_contrast" else False
        self.last_sweep_history = []
        self.current_sweep_history = []
        self.last_velocity = np.inf
        self.last_expected = 0
        self.velocity_protection= 0.25
        self.alpha_decrese_after_failure = 0.1
        self.cnt = 0
        self.use_history_for_next_sweep = True
        self.use_history_for_offset_refinements = True
        self.vertices = [[],[]]
        self.delta_min_counter = 0
        
    def pre_proof(self, sample,label,Dataset,delta_in):
        print("Initializing history...")
        delta = delta_in.copy()
        for i,delta_step in enumerate(self.default_steps):
            if self.is_2d:
                [specLB, specUB] = self.perturbations.Perturbe(sample, [delta[0],delta[1]], [delta_step,delta_step])
            else:
                [specLB, specUB] = self.perturbations.Perturbe(sample, delta[0], delta_step)
            sensitivity_zero,_ = self.network.test(Dataset.normalize(sample), Dataset.normalize(sample), label, False)
            sensitivity,elapsed_time = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, (delta_step > 0))
            self.sensitivity_zero[i] = sensitivity_zero
            self.sensitivity[i] = sensitivity
            self.sensitivity_deltas[i] = delta_step
            if sensitivity>0:
                delta[0] = delta[0] + delta_step
                self.velocity_deltas[i] = delta_step
                self.velocity[i] = delta_step/elapsed_time
        return delta[0]
    
    def initialize_new_sweep_via_history(self, sample,label,Dataset,delta_in):
        delta = delta_in.copy()
        M_missing = int(0)
        M_histroy = len(self.last_sweep_history)
        if M_histroy<len(self.default_steps):
            M_missing =  int(len(self.default_steps)-M_histroy)
     
        for i in range(M_missing):
            delta_step = self.default_steps[i]
            if self.is_2d:
                [specLB, specUB] = self.perturbations.Perturbe(sample, [delta[0],delta[1]], [delta_step,delta_step])
            else:
                [specLB, specUB] = self.perturbations.Perturbe(sample, delta[0], delta_step)
            sensitivity_zero,_ = self.network.test(Dataset.normalize(sample), Dataset.normalize(sample), label, False)
            sensitivity,_ = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, (delta_step > 0))
            self.sensitivity_zero[i] = sensitivity_zero
            self.sensitivity[i] = sensitivity
            self.sensitivity_deltas[i] = delta_step
            
        for i in range(int(len(self.default_steps)-M_missing)):
            data = self.last_sweep_history[i]
            self.sensitivity_deltas[i+M_missing] = data[1]
            self.sensitivity[i+M_missing] = data[2]
            self.sensitivity_zero[i+M_missing] =  data[3]
        
    def next_step(self, sample,label,Dataset,delta):
        [specLB, specUB] = self.perturbations.Perturbe(sample, delta, 0)  
        sensitivity_zero,_ = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, False)
        self.sensitivity[self.M-1]= sensitivity_zero
        self.sensitivity_zero[self.M-1]= sensitivity_zero
        self.sensitivity_deltas[self.M-1]= 0                                             
        delta_step_expected_failure, prm_best = self.approximate_certification_failure_(self.sensitivity_deltas, self.sensitivity,self.sensitivity_zero)
        delta_step_max_velocity = self.approximate_max_velocity(self.velocity_deltas[0:self.M-1],self.velocity[0:self.M-1])
        expected_delta_step = min(delta_step_expected_failure,delta_step_max_velocity)
        expected_delta_step = self.refine_prediction(expected_delta_step)
        expected_delta_step = max(expected_delta_step,self.delta_min)
        [specLB, specUB] = self.perturbations.Perturbe(sample, delta, expected_delta_step)  
        sensitivity,elapsed_time  = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, True)        
        self.update_sensitivity_vec(expected_delta_step, sensitivity, sensitivity_zero)
        if sensitivity>0:
            delta += expected_delta_step
            self.update_prm_vec(prm_best)
            self.update_velocity_vec(expected_delta_step, elapsed_time)
        else:
            self.update_failure()
        return delta, expected_delta_step
    
    def next_step_2d(self, sample,label,Dataset,offsets):
        delta = offsets[0]
        [specLB, specUB] = self.perturbations.Perturbe(sample, offsets, [0,0])  
        sensitivity_zero,_ = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, False)
        self.sensitivity[self.M-1]= sensitivity_zero
        self.sensitivity_zero[self.M-1]= sensitivity_zero
        self.sensitivity_deltas[self.M-1]= 0  
        [cls_delta, cls_sensitivity, cls_sensitivity_zero] = self.find_closest(delta)  
        delta_step_expected_failure, prm_best = self.approximate_certification_failure_(\
                                                np.append(self.sensitivity_deltas,cls_delta),\
                                                np.append(self.sensitivity,cls_sensitivity),
                                                np.append(self.sensitivity_zero,cls_sensitivity_zero) )
        delta_step_max_velocity = self.approximate_max_velocity(self.velocity_deltas[0:self.M-1],self.velocity[0:self.M-1])  
        expected_delta_step = min(delta_step_expected_failure,delta_step_max_velocity)
        expected_delta_step = self.refine_prediction(expected_delta_step)
        expected_delta_step = max(expected_delta_step,self.delta_min)
        offsets = self.refine_due_to_offsets(offsets,expected_delta_step)
        
        [specLB, specUB] = self.perturbations.Perturbe(sample, offsets, [expected_delta_step,expected_delta_step])  
        sensitivity,elapsed_time  = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, True)        
        self.update_sensitivity_vec(expected_delta_step, sensitivity, sensitivity_zero)
        if sensitivity>0:
            delta += expected_delta_step
            self.update_prm_vec(prm_best)
            self.update_velocity_vec(expected_delta_step, elapsed_time)
            self.current_sweep_history.append([offsets.copy(),expected_delta_step,sensitivity,sensitivity_zero, self.increase_backoff]) 
        else:
            self.update_failure()
        self.last_expected = expected_delta_step
        return delta        
    
    def refine_due_to_offsets(self, offsets, delta):        
        min_offset = np.inf
        for i in self.last_sweep_history:
            if (i[0][0]>=offsets[0] and i[0][0]<= offsets[0]+delta) or\
             offsets[0]>=i[0][0] and offsets[0]<=i[0][0]+i[1]:
                min_offset = min(min_offset,i[0][1]+i[1])
                if self.use_history_for_offset_refinements:
                    self.increase_backoff = i[4]
        if min_offset<np.inf:
            offsets[1] = min_offset
        return offsets
    
    def find_closest(self,delta):
        for i in self.last_sweep_history:
            if delta>= i[0][0] and delta<= i[0][0]+i[1]:
                return [i[1], i[2], i[3]]
        return [[],[],[]]
    
    def certify(self,sample,label,Dataset,delta_in,delta_t,delta_split,max_itr,img_index,conn=None):
        start_time = time.time()
        if self.is_2d:
            self.pre_proof(sample,label,Dataset,[0,delta_in])
            offsets = [0,delta_in]
            sweeps_cnt = 0
            save_data = []
            delta_proved = [0,0]
            delta_target = [delta_t,delta_split]  
            for i in range(max_itr): 
                delta = self.next_step_2d(sample,label,Dataset,offsets)
                offsets[0] = delta
                has_failed_over_delta_min = self.has_failed_over_delta_min()
                if delta >= delta_t or self.delta_min_counter>0:    
                    self.update_vertices()                     
                    delta_proved_current = [np.min(self.vertices[0]),np.min(self.vertices[1])]
                    if delta_proved_current[0]*delta_proved_current[1]>=delta_proved[0]*delta_proved[1]:
                        delta_proved = np.copy(delta_proved_current)
                    save_data.append(self.current_sweep_history)
                    self.last_sweep_history = np.copy(self.current_sweep_history)
                    self.current_sweep_history = []
                    for j in self.last_sweep_history:
                        if j[0][1]+j[1] < delta_split:
                            offsets = [j[0][0],j[0][1]+j[1]]
                            break
                    if (delta_proved[1]>=delta_split) or has_failed_over_delta_min or\
                     time.time()-start_time>self.TimeOut:
                        if (delta_proved[1]>=delta_split):
                            delta_proved = [delta_t,delta_split]
                        break
                         
                    if self.use_history_for_next_sweep:
                        self.initialize_new_sweep_via_history(sample,label,Dataset,offsets)
                    else: 
                        self.pre_proof(sample,label,Dataset,offsets) 
                    self.increase_backoff = 0
                    self.FA = 0
                    sweeps_cnt += 1    
                print("s:",img_index,"step:",i,"sweep:",sweeps_cnt,"current:",delta_proved,"target:",delta_target,"time[s]:",time.time()-start_time,"GPU:",self.gpu_to_use)             
            return delta_proved,delta_target 
        else:
            delta_x = np.copy(delta_in)
            delta_x = self.pre_proof(sample,label,Dataset,np.array(delta_x).reshape(1))
            for i in range(max_itr): 
                delta_x, delta_tag = self.next_step(sample,label,Dataset,delta_x)
                if delta_x >= delta_t or self.has_failed_over_delta_min() or time.time()-start_time>self.TimeOut:
                    delta_x = delta_t if delta_x >= delta_t else delta_x
                    break
                if conn is not None:
                    ready = select.select([conn], [], [], 0.001)
                    if ready[0]:
                        data = str(conn.recv(1024)).replace("\'","")
                        if "new_target" in data:
                            delta_t = np.float(data.split(":")[-1])   
                    conn.send(str("delta:"+str(delta_x)+" ,delta_t:" +str(delta_t)+" ,delta_tag:" +str(delta_tag)).encode())
                print("s:",img_index,"step:",i,"current:",delta_x,"target:",delta_t,"time[s]:",time.time()-start_time,"GPU:",self.gpu_to_use)             
            return delta_x,delta_t
        
    def has_failed_over_delta_min(self):
        if self.is_2d:
            if self.sensitivity[self.M-2]<=0 and self.sensitivity_deltas[self.M-2]<=self.delta_min:
                self.delta_min_counter += 1
            else:
                self.delta_min_counter = 0
            if self.delta_min_counter>=2:
                return True
        else:
            for i in range(self.M):
                if self.sensitivity[i]<=0 and self.sensitivity_deltas[i]<=self.delta_min:
                    return True
        return False

    def get_senstivity(self,sample,label,delta,delta_step,Dataset):
        [specLB, specUB] = self.perturbations.Perturbe(sample, delta, delta_step)
        if self.is_2d:
            sensitivity,elapsed_time  = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, np.array(delta_step).any()>0)
        else:  
            sensitivity,elapsed_time  = self.network.test(Dataset.normalize(specLB), Dataset.normalize(specUB), label, delta_step>0)
        return sensitivity,elapsed_time
    
    def approximate_max_velocity(self, delta_vec_in, speed_vec_in):
        velocity_vec = self.approx_constant*np.copy(speed_vec_in)
        delta_vec = np.copy(delta_vec_in)
        t = []
        for i in range(len(velocity_vec)):
            t.append(np.log(velocity_vec[i])-np.log(delta_vec[i]))
        delta_step_max_velocity = np.inf
        prediction_failure = False
        try:        
           coeff = np.polyfit(np.array(delta_vec),np.array(t),1)
           coeff[1] = np.exp(coeff[1])
           delta_step_max_velocity = -1/coeff[0]
           if delta_step_max_velocity>0:
               self.last_velocity = delta_step_max_velocity
        except:
            prediction_failure = True
        
        if prediction_failure or delta_step_max_velocity<0 or \
        np.isnan(delta_step_max_velocity) or delta_step_max_velocity/self.last_expected<self.velocity_protection:
            delta_step_max_velocity = np.inf
                   
        return delta_step_max_velocity

    def approximate_certification_failure_(self, delta_vec_in, sensitivity_vec,sensitivity_zero_vec, back_off = 0.0001):
        delta_vec = np.copy(delta_vec_in)
        compensated_senstivity_vec = []
        for i in range(len(delta_vec)):
            compensated_senstivity_vec.append(sensitivity_vec[i]+sensitivity_zero_vec[self.M-1]-sensitivity_zero_vec[i])
        compensated_senstivity_vec = self.approx_constant*np.copy(compensated_senstivity_vec)
        
        if 0 not in self.prm_vec:
            self.power_low =int( np.clip(np.mean(self.prm_vec)-self.prm_band,self.power_low_min,self.power_high_max))
            self.power_high = int(np.clip(np.mean(self.prm_vec)+self.prm_band,self.power_low_min,self.power_high_max))
        prm_best = 0
        min_rms = np.inf
        prediction_failure = False
        try:
            for prm in range(self.power_low,self.power_high):
                coeff = np.polyfit(np.exp(prm*np.array(delta_vec)),np.array(compensated_senstivity_vec),1)
                y = []
                for i in delta_vec:
                    y.append(coeff[0]*np.exp(prm*i)+coeff[1])
                rms_ = np.mean(np.absolute(np.array(y)-np.array(compensated_senstivity_vec)))
                if rms_<min_rms:
                    min_rms = rms_
                    prm_best = prm    
            for i in range(self.M-1):
                back_off = math.pow(10,math.floor(math.log(delta_vec[i],10))-1)
                delta_vec[i] -=  back_off*(1+min(self.increase_backoff,3))
            coeff = np.polyfit(np.exp(prm_best*np.array(delta_vec)),np.array(compensated_senstivity_vec),1)  
            delta_step_expected_failure = (1/prm_best)*np.log(-coeff[1]/coeff[0])
            delta_step_expected_failure -=  math.pow(10,math.floor(math.log(delta_step_expected_failure ,10))-1)*(1+self.increase_backoff)
        except:
            prediction_failure = True
        if prediction_failure or math.isnan(delta_step_expected_failure) or delta_step_expected_failure<0:
            delta_step_expected_failure = np.min(delta_vec[0:self.M-1])
            back_off = math.pow(10,math.floor(math.log(delta_step_expected_failure,10))-1)
            delta_step_expected_failure -= back_off   
        
        return delta_step_expected_failure,prm_best
    
    def update_vertices(self):
        if len(self.current_sweep_history)>0:
            self.vertices[1] = []
            for i in self.current_sweep_history:
                self.vertices[1].append(i[0][1]+i[1])
            self.vertices[0].append(i[0][0]+i[1])
        
    def update_prm_vec(self, prm_best):
        for i in range(self.M-1):
            self.prm_vec[i]=self.prm_vec[i+1]
        self.prm_vec[self.M-1] = prm_best
        
    def update_velocity_vec(self, expected_delta_step, elapsed_time):
        for i in range(self.M-2):
            self.velocity[i]=self.velocity[i+1]
            self.velocity_deltas[i]=self.velocity_deltas[i+1]
        self.velocity[self.M-2]=expected_delta_step/elapsed_time    
        self.velocity_deltas[self.M-2]=expected_delta_step

    def update_sensitivity_vec(self, expected_delta_step, sensitivity, sensitivity_zero):
        for i in range(self.M-2):
            self.sensitivity[i]=self.sensitivity[i+1]
            self.sensitivity_zero[i]=self.sensitivity_zero[i+1]
            self.sensitivity_deltas[i]=self.sensitivity_deltas[i+1]
        self.sensitivity[self.M-2] = sensitivity
        self.sensitivity_zero[self.M-2] = sensitivity_zero
        self.sensitivity_deltas[self.M-2] = expected_delta_step
           
    def refine_prediction(self, expected_delta_step):
        if self.toggle:
            back_off = math.pow(10,math.floor(math.log(expected_delta_step,10))-1)
            if np.absolute(expected_delta_step-self.last_expected)<=back_off:
                expected_delta_step = self.last_expected-back_off
        self.toggle = not self.toggle
        min_fail_delta = np.inf
        for i in range(self.M):
            if self.sensitivity[i]<0 and self.sensitivity_deltas[i]<expected_delta_step:
                if min_fail_delta>self.sensitivity_deltas[i]:
                    min_fail_delta = self.sensitivity_deltas[i]
        if min_fail_delta<np.inf:
            expected_delta_step = min_fail_delta*(1-self.alpha_decrese_after_failure)
        expected_delta_step = max(expected_delta_step, self.delta_min)
        return expected_delta_step
        
    def reset(self):
        self.sensitivity = np.zeros(self.M)
        self.sensitivity_zero = np.zeros(self.M)
        self.sensitivity_deltas = np.zeros(self.M)      
        self.velocity = np.zeros(self.M)
        self.velocity_deltas = np.zeros(self.M) 
        self.prm_vec = np.zeros(self.M)
        self.toggle = False 
        self.power_low = 200
        self.power_high = 6000
        self.prm_band = 400
        self.FA = 0
        self.increase_backoff = 0
        self.last_velocity = np.inf
        self.last_sweep_history = []
        self.current_sweep_history = []
        self.last_expected = 0
        self.vertices = [[],[]]
        self.delta_min_counter = 0
        
    def update_failure(self):
        self.FA += 1
        self.increase_backoff  = min(self.increase_backoff+1,9)

    def compute_certified_delta(self,certified):
        if self.is_2d:
            certified_delta = [np.inf,np.inf]
            for i in certified:
              if certified_delta[0]>i[1][0]:
                certified_delta[0] = i[1][0]
              certified_delta[1] = i[1][1]
              if i[1][1]<i[2][1]:
                break
            return certified_delta
        else:
            certified_delta = 0
            for i in certified:
                certified_delta = i[1]
                if i[1]<i[2]:
                    break
            return certified_delta
