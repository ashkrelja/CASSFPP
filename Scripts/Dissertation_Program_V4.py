# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import quantecon
import numpy as np
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
#Import/manipulate data

path = 'C:/Users/ashkrelja/.spyder-py3/CASSFPP/Data/'

df_fer = pd.read_excel(path + 'D_FER_Var.xls',header=None)
df_ser = pd.read_excel(path + 'D_SER_Var.xls',header=None)
df_usrgdp = pd.read_excel(path + 'D_USRGDP_Var.xls',header=None)
df_ukrgdp = pd.read_excel(path + 'D_UKRGDP_Var.xls',header=None)
df_usfahh = pd.read_excel(path + 'D_USFAHH_Var.xls',header=None)
df_ukcurhh = pd.read_excel(path + 'D_UKCurHH_Var.xls',header=None)


FER = df_fer.as_matrix()
SER = df_ser.as_matrix()
XX = df_usrgdp.as_matrix()
YY = df_ukrgdp.as_matrix()
MM = df_usfahh.as_matrix()
NN = df_ukcurhh.as_matrix()

FER = np.log(FER)
SER = np.log(SER)
XX = np.log(XX)
YY = np.log(YY)
MM = np.log(MM)
NN = np.log(NN)

FER_cycle, FER_trend = sm.tsa.filters.hpfilter(FER)
SER_cycle, SER_trend = sm.tsa.filters.hpfilter(SER)
XX_cycle, XX_trend = sm.tsa.filters.hpfilter(XX)
YY_cycle, YY_trend = sm.tsa.filters.hpfilter(YY)
MM_cycle, MM_trend = sm.tsa.filters.hpfilter(MM)
NN_cycle, NN_trend = sm.tsa.filters.hpfilter(NN)

#Fit ARIMA system

model = ARIMA(XX_cycle, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(YY_cycle, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(MM_cycle, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(NN_cycle, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#Model Parameters

beta = 0.99
rho1 = 0.9646 #0.8927 
rho2 = 0.9670 #0.8261 
rho3 = 0.8965#0.6224 
rho4 = 0.7607#0.5376
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

theta=np.array(range(1,5000,1000))

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
        
    #    #Solve P using Schur - Decomposition
    #    
    #    #Define M for Schur - Decomposition
    #    
    #    M11=A+C*np.linalg.inv(beta*theta*A)*C.T*R
    #    M12=C*np.linalg.inv(beta*theta*A)*C.T
    #    M21=np.linalg.inv(A)*R
    #    M22=np.linalg.inv(A)
    #    
    #    M1=np.hstack((M11,M12))
    #    M2=np.hstack((M21,M22))
    #    
    #    M=np.vstack((M1,M2))
    #    
    #    [T,Z,sdim]=scipy.linalg.schur(M,output='real',sort='iuc')
    #    
    #    #Define Z11 & Z21 to solve for matrix P
    #    
    #    Z21=Z[4:8,0:4]
    #    Z11=Z[0:4,0:4]
    #    P1=Z21*np.linalg.inv(Z11)
    #    
    #    #Check for equivalence -(Z22^-1)Z21=Z21*Z11^-1
    #    Zinv=np.linalg.inv(Z)
    #    Z22=Z[4:8,4:8]
    #    Z21=Z[4:8,0:4]
    #    P2=-1*np.linalg.inv(Z22)*Z21
    #    
    #    #Break - Down Point
    #    
    #    w,v=np.linalg.eig(theta*C-C.T*P1*C) # W eigenvalues >0
    #    
    #    
    #    #Derive worst - case shock matrix K
    #    
    #    I=np.identity(4)
    #    K=np.linalg.inv(theta*I-C.T*P1*C)*C.T*P1*A
    
        R = -R
        
        F, P, K = quantecon.robustlq.RBLQ(Q,R,A,B,C,beta,theta).robust_rule()
    
        
        #State- Evolution Equation Xt+1 = (A + C'K)Xt + et+1
        
        A0=(A+C.T*K)
        
        #Monte - Carlo Simulation of State - Evolution Equation
        
        #Experiment Set - Up
        
        betaols=[]
        
        for i in range(1000):
            
            T=109 #number of observations
            Xo=np.ones((4,1)) #initialize state matrix
            mu=0 #error mean 0
            sigma=0.1 #error standard deviation 0.1
            error = np.random.normal(mu,sigma,T)
        
        
            Xo_hist=[]
            s=[]
            f=[]
        
            for error in error:
            
                Xo=A0*Xo + error
                S=np.matrix([1, -1])*np.matrix([[0, 0, 1, 0],[0, 0, 0, 1]])*Xo
                F=np.matrix([1, -1])*np.matrix([[0, 0, rho3, 0], [0, 0, 0, rho4]])*Xo
            
                Xo_hist.append(Xo)
                s.append(S)
                f.append(F)
        
        
            s=pd.Series(s)
            s_lag=pd.Series(s).shift(1)
            s_expost=s-s_lag
            s_expost=s_expost[1:109] #Eliminate NaN
        
            f=pd.Series(f)
            f_premium=f-s
            f_premium=f_premium[1:109]
        
            Beta=np.linalg.inv(f_premium.T.dot(f_premium))*f_premium.T.dot(s_expost)
            
            Beta=np.asscalar(Beta)
            betaols.append(Beta)
        
        #Plot Beta Density 
        
#        density=gaussian_kde(betaols)
#        xs = np.linspace(-5,5)
#        plt.plot(xs,density(xs))
#        plt.show()
        
        sns.distplot(betaols)
        
    except Exception:
        print("Unable to iterate for theta {}".format(str(theta)))
        pass
    

############
#  AD-HOC #
###########    
theta =2
R=-R
F, P, K = quantecon.robustlq.RBLQ(Q,R,A,B,C,beta,theta).robust_rule()

P = quantecon.solve_discrete_riccati(A, B, R, Q)
from quantecon.lqcontrol import LQ
I=np.eye(4)
lq = LQ(-beta*I*theta, R, A, C, beta=beta)

Ptest, ftest, dtest = lq.stationary_values()




theta =901

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

#Solve P using Schur - Decomposition

#Define M for Schur - Decomposition

M11=A+C*np.linalg.inv(beta*theta*A)*C.T*R
M12=C*np.linalg.inv(beta*theta*A)*C.T
M21=np.linalg.inv(A)*R
M22=np.linalg.inv(A)

M1=np.hstack((M11,M12))
M2=np.hstack((M21,M22))

M=np.vstack((M1,M2))

[T,Z,sdim]=scipy.linalg.schur(M,output='real',sort='iuc')

#Define Z11 & Z21 to solve for matrix P

Z21=Z[4:8,0:4]
Z11=Z[0:4,0:4]
P1=Z21*np.linalg.inv(Z11)

#Check for equivalence -(Z22^-1)Z21=Z21*Z11^-1
Zinv=np.linalg.inv(Z)
Z22=Z[4:8,4:8]
Z21=Z[4:8,0:4]
P2=-1*np.linalg.inv(Z22)*Z21



np.linalg.eigvals(theta*C-C.T*P1*C) # W eigenvalues >0

np.linalg.eig()