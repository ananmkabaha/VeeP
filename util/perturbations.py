
import csv
import numpy as np
import os.path
import pickle
from colorsys import rgb_to_hls, hls_to_rgb

def is_perturbation(perturbation_type):
    if perturbation_type in ["brightness", "saturation", "hue", "lightness", "brightness_and_contrast"]:
        return True
    else:
        return False
    
class Perturbations:
  def __init__(self, w=28, h=28, c=1, Perturbation_type="brightness"):
    self.Perturbation_type = Perturbation_type
    self.w = w
    self.h = h
    self.c = c
    self.max_perturbation = 1
    if Perturbation_type == "hue":
        self.max_perturbation = 6 
  def Perturbe(self,sample,delta,delta_step=0):
      
      
    if self.Perturbation_type == "brightness":
        specLB = np.clip(sample+delta, 0, 1)
        specUB = np.clip(sample+delta+delta_step, 0, 1)
        return [specLB,specUB]
    elif self.Perturbation_type == "saturation": 
        assert self.c ==3,"Saturation is suppurted only for RGB images."
        image_src = np.copy(sample).reshape(self.w ,self.h ,self.c)
        image_dst = np.copy(sample).reshape(self.w ,self.h ,self.c)
        specLB = np.copy(sample).reshape(self.w ,self.h ,self.c)
        specUB = np.copy(sample).reshape(self.w ,self.h ,self.c)
        eps_l = delta
        eps_h = delta + delta_step
        eps   = delta
        for i in range(0,self.w):
          for j in range(0,self.h):
            r_org = image_src[i,j,0]
            g_org = image_src[i,j,1]
            b_org = image_src[i,j,2]
            (h,l,s) = rgb_to_hls(r_org,g_org,b_org)
            s_ = s + eps
            s_low = s+eps_l
            s_high = s+eps_h
            dr=(r_org-l)/(s+1e-12); dg=(g_org-l)/(s+1e-12); db=(b_org-l)/(s+1e-12);
            phi = s_
            if s_>1:
              phi = 1
            if s_<0:
              phi = 0
            sigma_0_l = s_low
            sigma_0_h = s_high
            if sigma_0_h<0:
              sigma_0_l = 0 
              sigma_0_h = 0
            elif sigma_0_l<0:
              sigma_0_l = 0 
    
            sigma_1_l = s_low - 1
            sigma_1_h = s_high - 1
            if sigma_1_h<0:
              sigma_1_l = 0 
              sigma_1_h = 0
            elif sigma_1_l<0:
              sigma_1_l = 0 
            phi_l = sigma_0_l - sigma_1_h
            phi_h = sigma_0_h - sigma_1_l
            r_  = dr*phi+l;   g_ = dg*phi+l;    b_  = db*phi+l;
            if dr <0:
              r_l = dr*phi_h+l;
              r_h = dr*phi_l+l;
            else:
              r_l = dr*phi_l+l;
              r_h = dr*phi_h+l;         
            if dg <0:
              g_l = dg*phi_h+l;
              g_h = dg*phi_l+l;
            else:
              g_l = dg*phi_l+l;
              g_h = dg*phi_h+l; 
            if db <0:
              b_l = db*phi_h+l;
              b_h = db*phi_l+l;
            else:
              b_l = db*phi_l+l;
              b_h = db*phi_h+l; 
            image_dst[i,j,0]   = r_;   image_dst[i,j,1] = g_;    image_dst[i,j,2] = b_;
            specLB[i,j,0] = r_l;  specLB[i,j,1] = g_l; specLB[i,j,2] = b_l;
            specUB[i,j,0] = r_h;  specUB[i,j,1] = g_h; specUB[i,j,2] = b_h;
        return np.clip(specLB,0,1).reshape(3*self.w*self.h),np.clip(specUB,0,1).reshape(3*self.w*self.h)
    elif self.Perturbation_type == "hue": 
        assert self.c ==3,"Hue is suppurted only for RGB images."
        image_src = np.copy(sample).reshape(self.w,self.h,3)
        specLB = np.copy(sample).reshape(self.w,self.h,3)
        specUB = np.copy(sample).reshape(self.w,self.h,3)
        eps_l = delta
        eps_h = delta+delta_step
        for i in range(0,self.w):
          for j in range(0,self.h):
            r_org = image_src[i,j,0]
            g_org = image_src[i,j,1]
            b_org = image_src[i,j,2]
            (h,l,s) = rgb_to_hls(r_org,g_org,b_org)
            d = (1 -np.absolute(2*l-1))*s;
            m = l-d/2;
            h_ = 6*h + delta        
            h_low  = 6*h + eps_l
            h_high = 6*h + eps_h
            sigma_0 = h_
            if sigma_0<0:
                sigma_0 = 0
            sigma_1 = h_-1
            if sigma_1<0:
                sigma_1 = 0
            sigma_2 = h_-2
            if sigma_2<0:
                sigma_2 = 0
            sigma_3 = h_-3
            if sigma_3<0:
                sigma_3 = 0
            sigma_4 = h_-4
            if sigma_4<0:
                sigma_4 = 0
            sigma_5 = h_-5
            if sigma_5<0:
                sigma_5 = 0
            sigma_6 = h_-6
            if sigma_6<0:
                sigma_6 = 0
            sigma_0_l = h_low
            sigma_0_h = h_high
            if sigma_0_h<0:
              sigma_0_l = 0 
              sigma_0_h = 0
            elif sigma_0_l<0:
              sigma_0_l = 0 
            sigma_1_l = h_low - 1
            sigma_1_h = h_high - 1
            if sigma_1_h<0:
              sigma_1_l = 0 
              sigma_1_h = 0
            elif sigma_1_l<0:
              sigma_1_l = 0 
            sigma_2_l = h_low - 2
            sigma_2_h = h_high - 2
            if sigma_2_h<0:
              sigma_2_l = 0 
              sigma_2_h = 0
            elif sigma_2_l<0:
              sigma_2_l = 0 
            sigma_3_l = h_low - 3
            sigma_3_h = h_high - 3
            if sigma_3_h<0:
              sigma_3_l = 0 
              sigma_3_h = 0
            elif sigma_3_l<0:
              sigma_3_l = 0 
            sigma_4_l = h_low - 4
            sigma_4_h = h_high - 4
            if sigma_4_h<0:
              sigma_4_l = 0 
              sigma_4_h = 0
            elif sigma_4_l<0:
              sigma_4_l = 0 
            sigma_5_l = h_low - 5
            sigma_5_h = h_high - 5
            if sigma_5_h<0:
              sigma_5_l = 0 
              sigma_5_h = 0
            elif sigma_5_l<0:
              sigma_5_l = 0 
            phi_r =  1+sigma_2+sigma_4-sigma_5-sigma_1
            phi_g =  sigma_0+sigma_4-sigma_1-sigma_3
            phi_b =  sigma_2-sigma_5-sigma_3
            phi_r_l = 1+sigma_2_l+sigma_4_l-sigma_5_h-sigma_1_h
            phi_r_h = 1+sigma_2_h+sigma_4_h-sigma_5_l-sigma_1_l
            phi_g_l =  sigma_0_l+sigma_4_l-sigma_1_h-sigma_3_h
            phi_g_h =  sigma_0_h+sigma_4_h-sigma_1_l-sigma_3_l
            phi_b_l =  sigma_2_l-sigma_5_h-sigma_3_h
            phi_b_h =  sigma_2_h-sigma_5_l-sigma_3_l
            if d<0:
              r_l = d*phi_r_h+m
              r_h = d*phi_r_l+m
              g_l = d*phi_g_h+m
              g_h = d*phi_g_l+m
              b_l = d*phi_b_h+m
              b_h = d*phi_b_l+m
            else:
              r_l = d*phi_r_l+m
              r_h = d*phi_r_h+m
              g_l = d*phi_g_l+m
              g_h = d*phi_g_h+m
              b_l = d*phi_b_l+m
              b_h = d*phi_b_h+m         
            r_ = d*phi_r+m; g_ = d*phi_g+m; b_ = d*phi_b+m;
            specLB[i,j,0] = r_l;  specLB[i,j,1] = g_l; specLB[i,j,2] = b_l;
            specUB[i,j,0] = r_h;  specUB[i,j,1] = g_h; specUB[i,j,2] = b_h;
        
        return  np.clip(specLB,0,1).reshape(self.w*self.h*3),np.clip(specUB,0,1).reshape(self.w*self.h*3)

    elif self.Perturbation_type == "lightness": 
        assert self.c ==3,"Lightness is suppurted only for RGB images."
        image_src = np.copy(sample).reshape(self.w,self.h,3)
        specLB = np.copy(sample).reshape(self.w,self.h,3)
        specUB = np.copy(sample).reshape(self.w,self.h,3)
        eps_l = delta
        eps_h = delta+delta_step
        for i in range(0,self.w):
            for j in range(0,self.h):
                r_org = image_src[i,j,0]
                g_org = image_src[i,j,1]
                b_org = image_src[i,j,2]
                (h,l,s) = rgb_to_hls(r_org,g_org,b_org)
                l_low  = l + eps_l
                l_high = l + eps_h
                l_ = l+delta
                r,g,b = hls_to_rgb(h,l_,s)
                dr=(r_org-l)/(1-np.absolute(2*l-1)+1e-12); dg=(g_org-l)/(1-np.absolute(2*l-1)+1e-12); db=(b_org-l)/(1-np.absolute(2*l-1)+1e-12);
                sigma_0 = l_
                if sigma_0<0:
                    sigma_0 = 0
                sigma_1 = l_-1
                if sigma_1<0:
                    sigma_1 = 0
                sigma_0_2 = 2*l_
                if sigma_0_2<0:
                    sigma_0_2 = 0
                sigma_1_2 = 2*l_-1
                if sigma_1_2<0:
                    sigma_1_2 = 0
                sigma_2_2 = 2*l_-2
                if sigma_2_2<0:
                    sigma_2_2 = 0
                phi_1 =  sigma_0_2 + -2*sigma_1_2 + sigma_2_2 
                phi_2 =  sigma_0 - sigma_1
    
                sigma_0_l = l_low
                sigma_0_h = l_high
                if sigma_0_h<0:
                  sigma_0_l = 0 
                  sigma_0_h = 0
                elif sigma_0_l<0:
                  sigma_0_l = 0 
                sigma_1_l = l_low - 1
                sigma_1_h = l_high - 1
                if sigma_1_h<0:
                  sigma_1_l = 0 
                  sigma_1_h = 0
                elif sigma_1_l<0:
                  sigma_1_l = 0 
                sigma_0_2_l = 2*l_low 
                sigma_0_2_h = 2*l_high 
                if sigma_0_2_h<0:
                  sigma_0_2_l = 0 
                  sigma_0_2_h = 0
                elif sigma_0_2_l<0:
                  sigma_0_2_l = 0 
                sigma_1_2_l = 2*l_low-1 
                sigma_1_2_h = 2*l_high-1
                if sigma_1_2_h<0:
                  sigma_1_2_l = 0 
                  sigma_1_2_h = 0
                elif sigma_1_2_l<0:
                  sigma_1_2_l = 0 
                sigma_2_2_l = 2*l_low-2
                sigma_2_2_h = 2*l_high-2
                if sigma_2_2_h<0:
                  sigma_2_2_l = 0 
                  sigma_2_2_h = 0
                elif sigma_2_2_l<0:
                  sigma_2_2_l = 0 
                phi_1_l =  sigma_0_2_l + -2*sigma_1_2_h + sigma_2_2_l 
                phi_1_h =  sigma_0_2_h + -2*sigma_1_2_l + sigma_2_2_h
                phi_2_l =  sigma_0_l - sigma_1_h
                phi_2_h =  sigma_0_h - sigma_1_l
                r_ = dr*phi_1+phi_2; g_ = dg*phi_1+phi_2; b_ = db*phi_1+phi_2;
                if dr <0:
                  r_l = dr*phi_1_h+phi_2_l;
                  r_h = dr*phi_1_l+phi_2_h;
                else:
                  r_l = dr*phi_1_l+phi_2_l;
                  r_h = dr*phi_1_h+phi_2_h;
                if dg <0:
                  g_l = dg*phi_1_h+phi_2_l;
                  g_h = dg*phi_1_l+phi_2_h;
                else:
                  g_l = dg*phi_1_l+phi_2_l;
                  g_h = dg*phi_1_h+phi_2_h;
                if db <0:
                  b_l = db*phi_1_h+phi_2_l;
                  b_h = db*phi_1_l+phi_2_h;
                else:
                  b_l = db*phi_1_l+phi_2_l;
                  b_h = db*phi_1_h+phi_2_h;
                specLB[i,j,0] = r_l;  specLB[i,j,1] = g_l; specLB[i,j,2] = b_l;
                specUB[i,j,0] = r_h;  specUB[i,j,1] = g_h; specUB[i,j,2] = b_h;
        return np.clip(specLB,0,1).reshape(self.w*self.h*3),np.clip(specUB,0,1).reshape(self.w*self.h*3)

    elif self.Perturbation_type == "brightness_and_contrast": 
        if delta_step ==0:
            delta_step =[0,0]
        image_src = np.copy(sample)
        image_dst = np.copy(sample)
        specLB = np.copy(sample)
        specUB = np.copy(sample)
        ec_l = [delta[1],delta[1]+delta_step[1]]
        eb_l = [delta[0],delta[0]+delta_step[0]]
        for ind,_ in enumerate(image_src):
          val_low  = (1+ec_l[0])*image_src[ind]+eb_l[0]
          val_high = (1+ec_l[1])*image_src[ind]+eb_l[1]
          sigma_0_h = val_high
          sigma_0_l = val_low
          if sigma_0_h<0:
            sigma_0_h = 0 
            sigma_0_l = 0
          elif sigma_0_l<0:
            sigma_0_l = 0 
          sigma_1_h = val_high-1
          sigma_1_l = val_low-1
          if sigma_1_h<0:
            sigma_1_l = 0 
            sigma_1_h = 0
          elif sigma_1_l<0:
            sigma_1_l = 0 
          ub = np.clip(sigma_0_h-sigma_1_l,0,1)
          lb = np.clip(sigma_0_l-sigma_1_h,0,1)
          specUB[ind] = ub
          specLB[ind] = lb
        return np.clip(specLB,0,1),np.clip(specUB,0,1)
