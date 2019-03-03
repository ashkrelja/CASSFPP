# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Lab Notes: Tried ARIMA model w/ no constant, used those params then simulated data at log then convertd to level - not much movement
# Lab Notes: Left off at ARIMA model w/ no constant, used those params then simulated data at log and left at log - no

import quantecon
import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
#Import/manipulate data

path = 'C:/Users/ashkrelja/.PyCharmCE2018.3/CASSFPP/Data/'

#df_fer = pd.read_excel(path + 'D_FER_Var.xls',header=None)
#df_ser = pd.read_excel(path + 'D_SER_Var.xls',header=None)
#df_usrgdp = pd.read_excel(path + 'D_USRGDP_Var.xls',header=None)
#df_ukrgdp = pd.read_excel(path + 'D_UKRGDP_Var.xls',header=None)
#df_usfahh = pd.read_excel(path + 'D_USFAHH_Var.xls',header=None)
#df_ukcurhh = pd.read_excel(path + 'D_UKCurHH_Var.xls',header=None)

df_formatted = pd.read_excel(path + 'Updated_Data_Formatted.xlsx')

# Level Data

FER = df_formatted['FER'].as_matrix()
SER = df_formatted['SER'].as_matrix()
XX = df_formatted['US RGDP'].as_matrix()
YY = df_formatted['UK RGDP'].as_matrix()
MM = df_formatted['UK M1'].as_matrix()
NN = df_formatted['USM1'].as_matrix()

# OLS of Level Data

test=FER - SER
test=pd.Series(test)
test=test[1:128]

SER_lag=pd.Series(SER).shift(1)
SER_expost=SER-SER_lag
SER_expost=SER_expost[1:128]

plt.plot(FER)
plt.plot(SER)
plt.plot(FER-SER)
plt.plot(SER_expost)


model = sm.OLS(SER_expost,test)
results = model.fit()
results.params # beta is -1.235885

#Log Data

FER = np.log(FER)
SER = np.log(SER)
XX = np.log(XX)
YY = np.log(YY)
MM = np.log(MM)
NN = np.log(NN)

plt.plot(MM)
plt.plot(NN)
plt.plot(FER-SER)
plt.plot(SER_expost)

# OLS of Logged Data

test=FER - SER
test=pd.Series(test)
test=test[1:128]

SER_lag=pd.Series(SER).shift(1)
SER_expost=SER-SER_lag
SER_expost=SER_expost[1:128]

model = sm.OLS(SER_expost,test)
results = model.fit()
results.params # beta is -1.058728

# HP Filter Data


FER_cycle, FER_trend = sm.tsa.filters.hpfilter(FER)
SER_cycle, SER_trend = sm.tsa.filters.hpfilter(SER)
XX_cycle, XX_trend = sm.tsa.filters.hpfilter(XX)
YY_cycle, YY_trend = sm.tsa.filters.hpfilter(YY)
MM_cycle, MM_trend = sm.tsa.filters.hpfilter(MM)
NN_cycle, NN_trend = sm.tsa.filters.hpfilter(NN)

plt.plot(SER_cycle)
plt.plot(FER_cycle)
plt.plot(MM_cycle)
plt.plot(NN_cycle)

# OLS of HP Filter Data

test_c=FER_cycle - SER_cycle
test_c=pd.Series(test_c)
test_c=test_c[1:128]

SER_lagc=pd.Series(SER_cycle).shift(1)
SER_expostc=SER_cycle-SER_lagc
SER_expostc=SER_expostc[1:128]

model = sm.OLS(SER_expostc,test_c)
results = model.fit()
results.params # beta is -1.759092


# fit normal distribution to logged data

mu_mm, std_mm = norm.fit(MM)
mu_nn, std_nn = norm.fit(NN)
plt.hist(MM, bins=25, density=True, alpha=0.6, color='g')
plt.hist(NN, bins=25, density=True, alpha=0.6, color='g')


#Fit ARIMA system

model = ARIMA(XX, order=(1,1,0))
model_fit = model.fit(disp=0 , trend='nc')
print(model_fit.summary())

model = ARIMA(YY, order=(1,1,0))
model_fit = model.fit(disp=0, trend='nc')
print(model_fit.summary())

model = ARIMA(MM, order=(1,1,0))
model_fit = model.fit(disp=0, trend='nc')
print(model_fit.summary())

model = ARIMA(NN, order=(1,1,0))
model_fit = model.fit(disp=0, trend='nc')
print(model_fit.summary())

#Model Parameters

beta = 0.99
rho1 = 0.7152 #0.3736  #0.8816 #0.8927 
rho2 = 0.7731 #0.6016  #0.9248 #0.8261 
rho3 = 0.7821 #0.4407  #0.8751 #0.6224 
rho4 = 0.8270 #0.6880  #0.9285 #0.5376
phi = 0.5
model_theta = 0.5
gamma = 10

# Steady-State Values

ssCx = 0.5
ssCy = 0.5
ssX = 1
ssY = 1
ssM = 1
ssN = 1

alpha = ((ssCx**model_theta)*(ssCy**(1-model_theta))**(1-gamma))/2

r1=(alpha)*(model_theta*ssX/(2*ssCx))
r2=(alpha)*((1-model_theta)*ssY/(2*ssCy))                                      
r3=(alpha)*(ssM/ssCx)*(0.5*(-(1-gamma))*((model_theta)**2)-(model_theta)*0.5)          
r4=(alpha)*(ssN/ssCy)*(0.5*(-(1-gamma))*((1-model_theta)**2)-(1-model_theta)*0.5) 


#Parameters (See MATLAB Program for Formulations)

#rho1=0.896698861048550
#rho2=0.803832231835419
#rho3=0.627841584406173
#rho4=0.542150217857893
#
#r1=4
#r2=4
#r3=158
#r4=158

beta=0.99
theta=1
theta=np.array(range(1,100,10))

for theta in theta:
    
    try:
        
    #theta=-1 #pessimism
    
    #LQ Matrices
       
        A=np.matrix([[rho1,0,0,0],
                     [0,rho2,0,0],
                     [0,0,rho3,0],
                     [0,0,0,rho4]])
        
        B=np.zeros((4,4))
        
        C=np.identity(4)
        
        Q=np.zeros((4,4))
        
        R=np.matrix([[r1,0,0,0],
                     [0,r2,0,0],
                     [0,0,r3,0],
                     [0,0,0,r4]])
            
        R = -R
        
        F, K, P = quantecon.robustlq.RBLQ(Q,R,A,B,C,beta,theta).robust_rule()
    
        
        #State- Evolution Equation Xt+1 = (A + C'K)Xt + et+1
        
        A0=(A+C.T*K)
        
        #Monte - Carlo Simulation of State - Evolution Equation
        
        #Experiment Set - Up
        
        betaols=[]
        
        for i in range(1000):
            
            T=128 #number of observations
            
            Xo_hist=[]
            s_hist=[]
            f_hist=[]

            
            for T in range(T):
            
            
                Xoint=np.matrix([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,mu_mm,0],
                     [0,0,0,mu_nn]]) #initialize state matrix
    
                mu=0 #error mean 0
                sigma=0.0001 #error standard deviation 0.1
                error = np.matrix([np.random.normal(mu,sigma),np.random.normal(mu,sigma),np.random.normal(mu,std_mm),np.random.normal(mu,std_nn)])
                
                Xo=np.diagonal(A0*Xoint + error)
                
                selection_s=np.matmul(np.matrix([[0, 0, 1, 0],[0, 0, 0, 1]]),Xo)
                S=np.matmul(np.matrix([1,-1]),selection_s.T)
                
                #Convert S to level
#                S_interim = np.asscalar(S)
#                S = np.exp(S_interim)
                
                
                selection_f=np.matmul(np.matrix([[0, 0, rho3, 0],[0, 0, 0, rho4]]),Xo)
                F=np.matmul(np.matrix([1,-1]),selection_f.T)
                
                #Convert F to level
#                F_interim = np.asscalar(F)
#                F = np.exp(F_interim)
                
                Xo_hist.append(Xo)
                s_hist.append(S)
                f_hist.append(F)
        
        
            s=pd.Series(s_hist)
            s_lag=pd.Series(s).shift(1)
            s_expost=s-s_lag
            s_expost=s_expost[1:T] #Eliminate NaN
        
            f=pd.Series(f_hist)
            f_premium=f-s
            f_premium=f_premium[1:T]
        
            Beta=np.linalg.inv(f_premium.T.dot(f_premium))*f_premium.T.dot(s_expost)
#            model = sm.OLS(s_expost,f_premium)
#            results = model.fit()
#            Beta = results.params # beta is -1.35687

            
            Beta=np.asscalar(Beta)
            betaols.append(Beta)
            
        plt.figure("Distribution")        
        sns.distplot(betaols, label=str(theta))
        plt.legend()
        plt.show()
            
        
        
    except Exception:
        print("Unable to iterate for theta {}".format(str(theta)))
        pass
    
    plt.figure("Distribution")        
    sns.distplot(betaols, label=str(theta))
    plt.legend()
    plt.show()
    

plt.plot(s_expost) #simulated
plt.plot(SER_expost) #actual

plt.plot(f_premium) #simulated
plt.plot(FER-SER) #actual






