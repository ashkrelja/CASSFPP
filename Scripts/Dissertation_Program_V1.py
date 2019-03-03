# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import quantecon
import numpy as np
import scipy 
from quantecon.lqcontrol import LQ
#Parameters (See MATLAB Program for Formulations)

rho1=0.896698861048550
rho2=0.803832231835419
rho3=0.627841584406173
rho4=0.542150217857893

r1=4
r2=4
r3=158
r4=158

beta=0.99
theta=20 #pessimism
#LQ Matrices

A=np.matrix([[rho1,0,0,0],
   [0,rho2,0,0],
   [0,0,rho3,0],
   [0,0,0,rho4]])

B=np.matrix([[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]])

C=np.matrix([[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]])

Q=np.matrix([[0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]])

R=np.matrix([[r1,0,0,0],
             [0,r2,0,0],
             [0,0,r3,0],
             [0,0,0,r4]])

P=np.ones(4)
I = np.ones(4)
Z = np.zeros(4)
quantecon.robustlq.RBLQ(Q,R,A,B,C,beta,theta).robust_rule()
F,K,P=quantecon.robustlq.RBLQ.robust_rule()

lq = LQ(-beta*I*theta, R, A, C, beta=beta)
P, f, d = lq.stationary_values()

#Define M for Schur-Decomposition

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
#Zinv=np.linalg.inv(Z)
#Z22=Z[4:8,4:8]
#Z21=Z[4:8,0:4]
#P2=-1*np.linalg.inv(Z22)*Z21


























