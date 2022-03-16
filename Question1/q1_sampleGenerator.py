#Alex Yeh
#Question 1 Sample Generator

import numpy as np

def makeSamples(N,name):
    #provided data: 
    #N = 10000 #number of Samples
    n=2        #number of dimensions
    p0 = 0.65
    p1=0.35
    w1=0.5
    w2=0.5

    #using row vectors
    m01 = np.array([3,0])
    C01 = np.array([[2,0],
                [0,1]])

    m02 = np.array([0,3])
    C02 = np.array([[1,0],
                [0,2]])

    m1 = np.array([2,2])
    C1 = np.array([[1,0],
                [0,1]])


    #generate true labels and samples
    A = np.random.rand(N,1)
    class1 = A<=p1 #0.35
    class0a = (A<=p1+p0*w1) & (A>p1) #0.325
    class0b = A>p1+p0*w1 #0.325

    trueClassLabels = class1
    print("Class Priors")
    print("p(L=0) = "+str(np.sum(trueClassLabels==0)/N))
    print("p(L=1) = "+str(np.sum(trueClassLabels==1)/N))

    x0a = np.random.multivariate_normal(m01, C01, N)
    x0b = np.random.multivariate_normal(m02, C02, N)
    x1 = np.random.multivariate_normal(m1, C1, N)

    #class0 and class1 are mutually exclusive and collectively exhaustive
    samples = class1*x1 + class0a*x0a + class0b*x0b

    #store true labels and samples
    with open(name+str(N)+'_Labels.npy', 'wb') as f1:
        np.save(f1, trueClassLabels)

    with open(name+str(N)+'_Samples.npy', 'wb') as f2:
        np.save(f2, samples)

makeSamples(20,"D_train")
makeSamples(200,"D_train")
makeSamples(2000,"D_train")
makeSamples(10000,"D_validate")
