import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

T_classes = np.load(open('D_train20_Labels.npy', 'rb'))
T_samples = np.load(open('D_train20_Samples.npy', 'rb'))

z_fun = lambda x: np.concatenate(([np.ones(x.shape[0])],[x[:,0]],[x[:,1]],
                        [np.multiply(x[:,0],x[:,0])],[np.multiply(x[:,0],x[:,1])],[np.multiply(x[:,1],x[:,1])]),axis=0).T
h_fun = lambda w,z: 1.0/(1+np.exp(-np.matmul(z,np.array([w]).T))) #because minimize keeps on making w a row vector :(
cost_fun = lambda w,x,y: -np.sum(np.multiply(y,np.log(h_fun(w,z_fun(x))))+np.multiply(1-y,np.log(1-h_fun(w,z_fun(x)))))

#extra constraint to keep weights reasonable, nah
#cons = ({'type': 'eq', 'fun': lambda x:  np.linalg.norm(x,ord=2)-1})
#w = np.array([[-2.66260419 , 1.68913753 , 1.53184803 ,-0.47240616,  0.30101388, -0.37719923]])
w = np.array([[0,0,0,0,0,0]])
res = minimize(cost_fun,w,args=(T_samples,T_classes))

print(res.x)
w=res.x

Decisions = h_fun(w,z_fun(T_samples))

fig,ax = plt.subplots()

samples1 = T_samples[np.argwhere(T_classes)[:,0],:]#wacky syntax. 
s1 = ax.scatter(samples1[:,0],samples1[:,1],color='g',marker='o',alpha=0.5)
samples0 = T_samples[np.argwhere(1-T_classes)[:,0],:]#wacky syntax. 
s2 = ax.scatter(samples0[:,0],samples0[:,1],color='r',marker='o',alpha=0.5)

samples1p = T_samples[np.argwhere(Decisions>0.5)[:,0],:]#wacky syntax. 
s3 = ax.scatter(samples1p[:,0],samples1p[:,1],color='b',marker='+',alpha=0.5)

plt.show()
#print(T_classes)
#cost_fun = lambda x,y,w : 

