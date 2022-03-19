import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def runLogisticQuadratic(name):
    T_classes = np.load(open(name+'_Labels.npy', 'rb'))
    T_samples = np.load(open(name+'_Samples.npy', 'rb'))

    #quadratic basis vector
    z_fun = lambda x: np.concatenate(([np.ones(x.shape[0])],[x[:,0]],[x[:,1]],
                            [np.multiply(x[:,0],x[:,0])],[np.multiply(x[:,0],x[:,1])],[np.multiply(x[:,1],x[:,1])]),axis=0).T
    h_fun = lambda w,z: 1.0/(1+np.exp(-np.matmul(z,np.array([w]).T))) #because minimize keeps on making w a row vector :(
    cost_fun = lambda w,x,y: -np.sum(np.multiply(y,np.log(h_fun(w,z_fun(x))))+np.multiply(1-y,np.log(1-h_fun(w,z_fun(x)))))

    #train the logistic quadratic model
    w = np.array([[0,0,0,0,0,0]])

    #the 2000 sample training set is ill conditioned for the quadratic model
    #scipy's minimize function has a hard time converging so we help it out by supplying 
    #the weights of the 200 sample training set as a starting point
    if name=='D_train2000':
        w = np.array([[-2.66260412,1.68913743,1.53184807,-0.47240614,0.30101389,-0.37719926]])

    res = minimize(cost_fun,w,args=(T_samples,T_classes))
    w = res.x
    print("Optimal weights:")
    print(w)

    #evaluate on validation samples
    #h_fun is the logistic linear model that approximates class posteiors
    V_samples = np.load(open('D_validate10000_Samples.npy', 'rb'))
    pL1givenx = h_fun(res.x,z_fun(V_samples))
    pL0givenx = 1-pL1givenx #1-h_fun(w,x)
    ratio = pL1givenx/pL0givenx

    trueLabels = np.load(open('D_validate10000_Labels.npy', 'rb'))
    N = trueLabels.shape[0]
    totalPos = np.sum(trueLabels==1)
    totalNeg = N - totalPos

    #Generate ROC Curve to compare against other classifiers
    Ngammas = 8000
    g1 = np.logspace(-20,-1,num=int(Ngammas/4),endpoint=False)
    g2 = np.linspace(0.1,10,num=int(Ngammas/2),endpoint=False)
    g3 = np.logspace(1,20,num=int(Ngammas/4))
    gammas = np.concatenate((g1,g2,g3))

    gammaResults = np.zeros((Ngammas,4))#col = {gamma, false negative, false positive, true positive}
    gammaResults[:,0] = gammas

    #future improvement: these could be done in parallel
    for i in range(Ngammas):
        gamma = gammas[i]
        Decisions = ratio>gamma

        truePos = (Decisions==1) & (trueLabels==1)
        falsePos = (Decisions==1) & (trueLabels==0)
        ntp = np.sum(truePos)
        nfp = np.sum(falsePos)

        ptp = ntp/totalPos #probability of true positive 
        pfp = nfp/totalNeg #probability of false positive (type 1 error)

        falseNeg = (Decisions==0) & (trueLabels==1)
        pfn = np.sum(falseNeg)/totalPos #probability of false negative (type 2 error)

        gammaResults[i,1] = pfn #kept track of since section 2 specifically asks for it
        gammaResults[i,2] = pfp
        gammaResults[i,3] = ptp

    #save ROC data for part B
    with open('Q1_ROC_'+name+'_quadratic.npy', 'wb') as f1:
        np.save(f1, gammaResults[:,2:4])

    #Calculate min error from theoretically optimal gamma value
    #Gamma is just a threshold. For gamma = 1, we are simply choosing the class with greater posterior probability
    gamma = 1 #MAP optimal gamma (since h models the class posteriors, not the class conditionals)
    Decisions = ratio>gamma
    #evaluate error
    falsePos = (Decisions==1) & (trueLabels==0)
    pfp = np.sum(falsePos)/totalNeg #probability of false positive (type 1 error)
    falseNeg = (Decisions==0) & (trueLabels==1)
    pfn = np.sum(falseNeg)/totalPos #probability of false negative (type 2 error)
    truePos = (Decisions==1) & (trueLabels==1)
    ptp = np.sum(truePos)/totalPos
    trueNeg = (Decisions==0) & (trueLabels==0)
    ptn = np.sum(trueNeg)/totalNeg

    gammaError = (pfn * totalPos + pfp * totalNeg)/N

    #Output select stats to console
    print("N = "+str(N))
    print("P(L=0) = "+str(totalNeg/N))
    print("Minimum P(error) = "+str(gammaError))
    print("Gamma for min P(error) = "+str(gamma))

    #decision boundary plot
    #make decision boundary
    NP = 500
    x0 = np.linspace(min(V_samples[:,0]),max(V_samples[:,0]),NP)
    x1 = np.linspace(min(V_samples[:,1]),max(V_samples[:,1]),NP)
    grid_samples=np.reshape(np.array(np.meshgrid(x0,x1)),(2,NP*NP))

    g_pL0givenx = 1-h_fun(w,z_fun(grid_samples.T))
    g_pL1givenx = h_fun(w,z_fun(grid_samples.T))
    g_ratio = g_pL1givenx/g_pL0givenx
    ratio_grid = g_ratio.reshape(NP,NP)

    fig,ax = plt.subplots()
    c1 = ax.contour(x0,x1,ratio_grid,levels=[gamma])

    #plot sample data
    samples_tp = V_samples[np.argwhere(truePos)[:,0],:]#wacky syntax. 
    s1 = ax.scatter(samples_tp[:,0],samples_tp[:,1],color='g',marker='o',alpha=0.5)
    samples_fn = V_samples[np.argwhere(falseNeg)[:,0],:]
    s2 = ax.scatter(samples_fn[:,0],samples_fn[:,1],color='r',marker='o',alpha=0.5)
    samples_tn = V_samples[np.argwhere(trueNeg)[:,0],:]
    s3 = ax.scatter(samples_tn[:,0],samples_tn[:,1],color='g',marker='+',alpha=0.5)
    samples_fp = V_samples[np.argwhere(falsePos)[:,0],:]
    s4 = ax.scatter(samples_fp[:,0],samples_fp[:,1],color='r',marker='+',alpha=0.5)

    #plot settings
    margin = 1.03
    ax.set(xlim=(min(V_samples[:,0])*margin, max(V_samples[:,0])*margin),
            ylim=(min(V_samples[:,1])*margin, max(V_samples[:,1])*margin)) 
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_title('Classifier Visualization for '+name)
    ax.legend(handles=[s1,s2,s3,s4],labels=['Correct Decision Class 1',
                    'Wrong Decision Class 1','Correct Decision Class 0','Wrong Decision Class 0'])

    plt.show()

def plotROC_logisticQuadratic():
    #load ROC data from part A,B
    ROCtruePdf = np.load(open('Q1_ROC1.npy', 'rb'))
    ROC20 = np.load(open('Q1_ROC_D_train20_quadratic.npy', 'rb'))
    ROC200 = np.load(open('Q1_ROC_D_train200_quadratic.npy', 'rb'))
    ROC2000 = np.load(open('Q1_ROC_D_train2000_quadratic.npy', 'rb'))

    # plot the ROC curve
    fig,ax = plt.subplots()
    l0,=ax.plot(ROCtruePdf[:,0], ROCtruePdf[:,1], color='tab:green',zorder=0,label='True PDF used')
    l1,=ax.plot(ROC20[:,0], ROC20[:,1], color='tab:pink',zorder=1,label='Logistic Quadratic trained on 20 Samples')
    l2,=ax.plot(ROC200[:,0], ROC200[:,1], color='tab:orange',zorder=2,label='Logistic Quadratic trained on 200 Samples')
    l3,=ax.plot(ROC2000[:,0], ROC2000[:,1], color='tab:blue',zorder=3,label='Logistic Quadratic trained on 2000 Samples')

    margin = 0.01
    ax.set(xlim=(-margin, 1+margin), ylim=(-margin, 1+margin)) #display [0,1] on both axes

    ax.set_xlabel('P(False Positive)')
    ax.set_ylabel('P(True Positive)')
    ax.set_title('Question 1 Part 2 ROC comparisons with Logistic Quadratic models')
    ax.legend(handles=[l0,l1,l2,l3])
    ax.axis('equal')

    plt.show()


runLogisticQuadratic('D_train20')
runLogisticQuadratic('D_train200')
runLogisticQuadratic('D_train2000')
plotROC_logisticQuadratic()
