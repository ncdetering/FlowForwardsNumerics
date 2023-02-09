import numpy as np
import math as mt
import matplotlib.pyplot as plt
import sys

a1=45*np.sqrt(5)
a2=24*np.sqrt(806115)
a3=1560*np.sqrt(49407661)
a4=720*np.sqrt(234602520551435)
a5=65520*np.sqrt(float(577134754173907))
a6=10080*np.sqrt(float(683295409767530975347))

def basis_fct(k,t):
    if k==1:
        return 1+0*t
    if k==2:
        return -1+np.exp(-t)
    if k==3:
        return np.exp(-t)*t
    if k==4:
        return np.exp(-t)*(1/2)*(-2*t+np.power(t,2))
    if k==5:
        return np.exp(-t)*(-6*t-36*np.power(t,2)+np.power(t,3))/a1
    if k==6:
        return np.exp(-t)*(-24*t-192*np.power(t,2)-1440*np.power(t,3)+np.power(t,4))/a2
    if k==7:
        return np.exp(-t)*(-120*t-1200*np.power(t,2)-10800*np.power(t,3)+100800*np.power(t,4)+np.power(t,5))/a3
    if k==8:
        return np.exp(-t)*(-720*t-8640*np.power(t,2)-90720*np.power(t,3)-967680*np.power(t,4)-10886400*np.power(t,5)+np.power(t,6))/a4
    if k==9:
        return np.exp(-t)*(-5040*t-70560*np.power(t,2)-846720*np.power(t,3)-10160640*np.power(t,4)-127008000*np.power(t,5)-127008000*np.power(t,6)+np.power(t,7))/a5
    if k==10:
        return np.exp(-t)*(-40320*t-645120*np.power(t,2)-8709120*np.power(t,3)-116121600*np.power(t,4)-1596672000*np.power(t,5)-1596672000*np.power(t,6)-1596671999*np.power(t,7)+np.power(t,8))/a6
    else:
        return 0
    
    
def evalCurve0(x,tau):
    xSize=x.size
    Y0=np.zeros(tau.size)
    for k in range(1,xSize+1,1):
        Y0+=x[k-1]*basis_fct(k,tau)
    return np.exp(Y0)


# Simulate one Brownian motion from 0 to T. We discretize daily 
T=1/12#one month simulation time horizon 
L=30#Discretization of BM
s = np.arange(0.0, 1/12, 1/(12*L))
Tminuss=T-s
print(Tminuss)
deltaS=1/(12*L)
Ssize=s.size

def evalCurve(x,BMInc,tau):
    sqrtdeltaS=mt.sqrt(deltaS)
    xSize=x.size
    TauSize=tau.size
    Y0=np.zeros(TauSize)
    drift=np.zeros(TauSize)
    noise=np.zeros(TauSize)
    for k in range(1,xSize+1,1):
        Y0+=x[k-1]*basis_fct(k,tau+T)
        for j in range(0,L,1):
            tmp=basis_fct(k,tau+Tminuss[j])
            drift+=tmp*tmp*deltaS
            noise+=tmp*BMInc[k-1,j]*sqrtdeltaS
        #tmp+=x[k]*np.power(t+T,k-1)*mt.exp(-(t+T))
    return np.exp(Y0-(1/2)*drift+noise)#mt.exp(-(1/2)*T+mt.sqrt(T)*BMInc +tmp)


from numpy.random import SeedSequence, default_rng
ss = SeedSequence(12345)

# Spawn off 32 child SeedSequences to pass to child processes and ensure that random variables are independent acros the nodes.
child_seeds = ss.spawn(32)
streams = [default_rng(s) for s in child_seeds]


taugrid=np.arange(0.0, 1/12, 1/(12*L))
TrainSizePerNode = 156250
def magic_function(f):
    Strike = 1.0
    a=0.8    
    maxKoefficient=f[1].poisson(lam=3.0, size=TrainSizePerNode)+1.
    StartingValuesLocal=[f[1].uniform(low=[-1,-a,-a**2,-a**3,-a**4,-a**5,-a**6,-a**7,-a**8,-a**9],high=[1,a,a**2,a**3,a**4,a**5,a**6,a**7,a**8,a**9], size=None) for k in range(0,TrainSizePerNode)]
    StartingValuesLocal=np.array(StartingValuesLocal)
    indexMatrix=maxKoefficient[:,None]<=np.arange(StartingValuesLocal.shape[1])
    StartingValuesLocal[indexMatrix]=0
    
    IncrementsBM = f[1].normal(0,1,size=[TrainSizePerNode,10,L])
    Y=np.zeros(TrainSizePerNode)
    
    for sample in range(0, TrainSizePerNode, 1):
        
        tmpvec=evalCurve(StartingValuesLocal[sample],IncrementsBM[sample],taugrid)
        tmp=np.sum(tmpvec)*(12/365)
        tmp=max(tmp-Strike,0)
        Y[sample]=tmp
    return StartingValuesLocal, Y


def process_frame(f):
    return f[0], magic_function(f)


from tqdm import tqdm

from multiprocess import Pool
#from magic_functions import process_frame
import time
tic = time.perf_counter()

frames_list = [[i,streams[i]] for i in range(0,len(streams),1)]

max_pool = 32

with Pool(max_pool) as p:
    pool_outputs = list(
        tqdm(
            p.imap(process_frame,
                   frames_list),
            total=len(frames_list)
        )
    )    

print(pool_outputs)
new_dict = dict(pool_outputs)
toc = time.perf_counter()
print(f"Calc time: {toc - tic:0.4f} seconds")



TotalTrainSize=TrainSizePerNode*len(streams)
startingValues=np.zeros((TotalTrainSize,10))
Yvalues=np.zeros(TotalTrainSize)
for x in range(0, len(streams), 1):
    startingValues[x*TrainSizePerNode:(x+1)*TrainSizePerNode]=pool_outputs[x][1][0]
    Yvalues[x*TrainSizePerNode:(x+1)*TrainSizePerNode]=pool_outputs[x][1][1]
np.save('startingvaluesMultiVariatePoisson', startingValues)
np.save('outputsMultiVariatePoisson',Yvalues)
