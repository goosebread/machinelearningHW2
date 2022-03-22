import numpy as np
import matplotlib.pyplot as plt
from hw2q2 import hw2q2

#generate cubic terms
#vector of basis functions
def z_fun(x):
    x02 = np.multiply(x[:,0],x[:,0])
    x0x1 = np.multiply(x[:,1],x[:,0])
    x12 = np.multiply(x[:,1],x[:,1])
    x03 = np.multiply(x[:,0],x02)
    x02x1 = np.multiply(x[:,1],x02)
    x0x12 = np.multiply(x[:,0],x12)
    x13 = np.multiply(x[:,1],x12)
    return np.concatenate(([np.ones(x.shape[0])],[x[:,0]],[x[:,1]],
                [x02],[x0x1],[x12],
                [x03],[x02x1],[x0x12],[x13]),axis=0).T

#forward pass function to predict y from input x and weights w
def c_func(x,w):
    return np.matmul(z_fun(x),w)

#ML parameter estimation applied to cubic model with additive noise
def run_ML(xTrain,yTrain,xValidate,yValidate):

    #closed form solution for weights for cubic basis functions
    zTrain = z_fun(xTrain)
    w = np.matmul(np.linalg.inv(np.matmul(zTrain.T,zTrain)),np.matmul(zTrain.T,yTrain))

    #evaluate using validate set
    Predictions = c_func(xValidate,w)

    #analyze results
    N = yValidate.shape[0]
    totalSquaredError = np.sum(np.multiply(yValidate-Predictions,yValidate-Predictions))
    print("Average squared error for ML = "+str(totalSquaredError/N))

    #visualize results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    s1 = ax.scatter(xValidate[:,0],xValidate[:,1],yValidate,color='g',alpha=0.5,label="True Y")
    s2 = ax.scatter(xValidate[:,0],xValidate[:,1],Predictions,color='r',alpha=0.5,label="Predicted Y")
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')
    ax.set_title('ML Estimated Cubic Regression Model')
    ax.legend(handles=[s1,s2])
    plt.show()

    #return avg squared error for further analysis
    return (totalSquaredError/N)

def get_MAP_error(xTrain,yTrain,xValidate,yValidate,tau):

    #closed form solution for weights for cubic basis functions
    #tau = sigma*sigma/(N*gamma)
    zTrain = z_fun(xTrain)
    w = np.matmul(np.linalg.inv(np.matmul(zTrain.T,zTrain)+tau*np.eye(zTrain.shape[1])),np.matmul(zTrain.T,yTrain))
    #evaluate using validate set
    Predictions = c_func(xValidate,w)

    #analyze results
    N = yValidate.shape[0]
    totalSquaredError = np.sum(np.multiply(yValidate-Predictions,yValidate-Predictions))
    return totalSquaredError/N

#MAP parameter estimation applied to cubic model with additive noise
#Priors of the weights vector is assumed to be a zero-mean gaussian with hyperparameter for covariance
#The hyperparameter tau is defined as sigma*sigma/(N*gamma)
#where sigma = known standard deviation of additive noise model
#and gamma = hyperparameter for the variance of the distribution of priors defined in the problem description
def run_MAP(xTrain,yTrain,xValidate,yValidate):

    #test across many tau values (1e-4 to 1e4)
    Ntaus = 8000
    t1 = np.logspace(-1,-1,num=int(Ntaus/4),endpoint=False)
    t2 = np.linspace(0.1,10,num=int(Ntaus/2),endpoint=False)
    t3 = np.logspace(1,12,num=int(Ntaus/4))
    taus = np.concatenate((t1,t2,t3))
    tauErrors = np.zeros(Ntaus)
    for i in range(Ntaus):
        tauErrors[i] = get_MAP_error(xTrain,yTrain,xValidate,yValidate,taus[i])
        #print(str(taus[i])+' '+str(tauErrors[i]))

    tau_min = taus[np.argmin(tauErrors)]

    #apply classifier using optimal tau
    zTrain = z_fun(xTrain)
    w = np.matmul(np.linalg.inv(np.matmul(zTrain.T,zTrain)-tau_min*np.eye(zTrain.shape[1])),np.matmul(zTrain.T,yTrain))

    #evaluate using validate set
    Predictions = c_func(xValidate,w)

    #analyze results
    N = yValidate.shape[0]
    totalSquaredError = np.sum(np.multiply(yValidate-Predictions,yValidate-Predictions))

    print("Minimum Average squared error for MAP= "+str(totalSquaredError/N))
    print("Threshold tau for min error = "+str(tau_min))

    #visualize results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    s1 = ax.scatter(xValidate[:,0],xValidate[:,1],yValidate,color='g',alpha=0.5,label="True Y")
    s2 = ax.scatter(xValidate[:,0],xValidate[:,1],Predictions,color='r',alpha=0.5,label="Predicted Y")
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')
    ax.set_title('Optimal MAP Estimated Cubic Regression Model')
    ax.legend(handles=[s1,s2])
    plt.show()

    v = np.zeros((Ntaus,2))
    v[:,0] = taus
    v[:,1] = tauErrors
    return v

def run_q2():
    #get samples
    [xTrain_t,yTrain,xValidate_t,yValidate]=hw2q2()
    ML_Error = run_ML(xTrain_t.T,yTrain,xValidate_t.T,yValidate)
    v = run_MAP(xTrain_t.T,yTrain,xValidate_t.T,yValidate)
    taus=v[:,0]
    MAP_Errors=v[:,1]

    #plot tau vs average squared error
    fig,ax = plt.subplots()
    l1,=ax.semilogx(taus,MAP_Errors,color='b',label="MAP with specified prior")
    l2,=ax.semilogx([taus[0],taus[-1]],[ML_Error,ML_Error],color='g',linestyle='dashed',label="ML Average Squared Error")
    ax.set_xlabel('tau = sigma^2/(N*gamma)')
    ax.set_ylabel('Average Squared Error')
    ax.set_title('Average Squared Error on Validation Set vs Tau')
    ax.legend(handles=[l1,l2])
    plt.show()

run_q2()


