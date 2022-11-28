import socket
import time
import os
import sys
from subprocess import Popen
import signal
from functools import partial
import numpy as np
import select

def signal_handler(socket_interface, signal, frame):
    print("Ctrl-c was pressed. cleaning")
    socket_interface.clean()
    exit(1)

class Socket_interface:
    def __init__(self,number_of_workers):
        self.clients = []
        self.procs = []
        self.timeout = 1
        self.max_iterations = 100
        self.number_of_workers = number_of_workers
        self.is_available = []
        self.workers_info = [[0,0,0,0]]* number_of_workers
        self.certfied = []
        self.targets = [0]*number_of_workers
    def initalize_servers(self, netname, dataset, max_itr, M, pertubration_type, gpus, TimeOut):
        for gpu in gpus:
            print("python3 ./util/worker_run.py --netname "+str(netname)+" --dataset "+str(dataset)+\
                      " --max_itr "+str(max_itr)+" --M "+str(M)+" --pertubration_type "+\
                      str(pertubration_type) +" --gpus "+str(gpu)+"&")
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = Popen(['python3', './worker_run.py', "--netname", netname,\
                           "--dataset", dataset, "--max_itr", str(max_itr),\
                            "--M", str(M), "--pertubration_type", str(pertubration_type), "--gpus", str(gpu), "--timeout", str(TimeOut)]\
                          , stdout=sys.stdout, stderr=sys.stderr,env=my_env)
            self.procs.append(proc)
    
    def initalize_clients(self,gpus):
        for gpu in gpus:
            cnt = 0
            while cnt<self.max_iterations:
                print("Try number",cnt, "to connect to gpu",gpu)
                try:
                    host = socket.gethostname()
                    port = 5000 +int(gpu)
                    client_socket = socket.socket()
                    client_socket.connect((host, port))
                    self.clients.append(client_socket)
                    break
                except:
                    cnt += 1
                    time.sleep(self.timeout)

    def clean(self):
        for p in self.procs:
            p.terminate()
        for c in self.clients:
            c.close()
            
    def send_to_workers(self,image_index,number_of_workers,target_delta,target_delta_split):
        for w in range(0, number_of_workers):
            message = "data," + str(image_index) + "," + str(target_delta_split*w) \
            + "," + str(target_delta_split*(w+1))+","+str(target_delta)
            self.clients[w].send(message.encode())
            self.targets[w] = target_delta_split*(w+1)
    def listen_to_workers(self, number_of_workers,image_index,M,is_2d):
        if is_2d:
            while 1:
                for w in range(0, number_of_workers):
                    ready = select.select([self.clients[w]], [], [], 0.01)
                    if ready[0]:
                        data = str(self.clients[w].recv(1024)).replace("\'","")
                        if "finished" in data:
                            self.is_available.append(w)
                            delta_s = np.float(data.split(",")[1].strip())
                            delta_p = [np.float(data.split(",")[2].strip()),np.float(data.split(",")[3].strip())]
                            delta_t = [np.float(data.split(",")[4].strip()),np.float(data.split(",")[5].strip())]                
                            self.certfied.append([delta_s,delta_p,delta_t])  
                if len(self.is_available) == number_of_workers:
                    self.certfied = sorted(self.certfied, key=lambda x: x[0])
                    break                
        else:               
            i = 0
            while 1:
                i += 1
                for w in range(0, number_of_workers):
                    ready = select.select([self.clients[w]], [], [], 0.01)
                    if ready[0]:
                        data = str(self.clients[w].recv(1024)).replace("\'","")
                        if "finished" in data:
                            self.is_available.append(w)
                            delta_s = np.float(data.split(",")[0].split(":")[1])
                            delta_p = np.float(data.split(",")[1].split(":")[1])
                            delta_t = np.float(data.split(",")[2].split(":")[1])        
                            self.certfied.append([delta_s,delta_p,delta_t])  
                
                        elif "delta" in data:
                            delta = np.float(data.split(",")[0].split(":")[1])
                            delta_t = np.float(data.split(",")[1].split(":")[1])
                            delta_tag = np.float(data.split(",")[2].split(":")[1])
                            self.workers_info[w] = [delta, delta_t, delta_tag, (delta_t-delta)/delta_tag]
                        
                if len(self.is_available)>0:
                    w_best = -1 
                    w_score = -np.inf
                    for ind,w in enumerate(self.workers_info):
                        if (w[3]>w_score) and (self.targets[ind]==w[1]) and (ind not in self.is_available):
                            w_score = w[3]
                            w_best = ind
                    if w_score>M+2:
                        mid_point = (self.workers_info[w_best][0]+self.workers_info[w_best][1])/2
                        message = "data," + str(image_index) + "," + str(mid_point) + "," + str(self.workers_info[w_best][1])+",0"
                        self.clients[self.is_available[0]].send(message.encode())               
                        message = "new_target:" + str(mid_point)
                        self.clients[w_best].send(message.encode())  
                        self.targets[w_best] = mid_point              
                        self.targets[self.is_available[0]] = self.workers_info[w_best][1]              
                        self.is_available = self.is_available[1:]
                        
                if len(self.is_available) == number_of_workers:
                    self.certfied = sorted(self.certfied, key=lambda x: x[0])
                    break
        return self.certfied  
      
    def get_from_workers(self,certified, target_delta_split, number_of_workers):
        for w in range(0, number_of_workers):
            print("Reading data from worker ",w)
            delta = np.float64(self.clients[w].recv(1024).decode()) 
            certified.append([target_delta_split*w,delta,target_delta_split*(w+1)])
        return certified

    def reset(self):
        self.is_available = []
        self.workers_info = [[0,0,0,0]]* self.number_of_workers
        self.certfied = []
        self.targets = [0]*self.number_of_workers
 
