# Alex Yeh
# Question 1 Part 1
# some code is reused from HW1 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

N = 10000 #number of samples
n=2        #number of dimensions

#true distribution knowledge is known
p0 = 0.65
p1=0.35
w1=0.5
w2=0.5
m01 = np.array([3,0])
C01 = np.array([[2,0],
            [0,1]])
m02 = np.array([0,3])
C02 = np.array([[1,0],
            [0,2]])
m1 = np.array([2,2])
C1 = np.array([[1,0],
            [0,1]])

mvn0a = multivariate_normal(m01,C01)
mvn0b = multivariate_normal(m02,C02)
mvn1 = multivariate_normal(m1,C1)

#classify samples
samples = np.load(open('D_validate10000_Samples.npy', 'rb'))

pxgivenL0a = np.array([mvn0a.pdf(samples)]).T
pxgivenL0b = np.array([mvn0b.pdf(samples)]).T
pxgivenL1 = np.array([mvn1.pdf(samples)]).T

#likelihood ratio to be compared against gamma
ratio = pxgivenL1/(0.5*pxgivenL0a+0.5*pxgivenL0b) 

trueLabels = np.load(open('D_validate10000_Labels.npy', 'rb'))
totalPos = np.sum(trueLabels==1)
totalNeg = N - totalPos

#Generate ROC Curve
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

#Section 3 - Calculate min error from theoretically optimal gamma value
gamma = p0/p1 #MAP optimal gamma
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


# plot the ROC curve
fig,ax = plt.subplots()
l1=ax.plot(gammaResults[:,2], gammaResults[:,3], color='tab:orange',zorder=1)
l2=ax.scatter(pfp, ptp, color='tab:blue',marker='x',label="minimum P(error)",zorder=2)
margin = 0.01
ax.set(xlim=(-margin, 1+margin), ylim=(-margin, 1+margin)) #display [0,1] on both axes

ax.set_xlabel('P(False Positive)')
ax.set_ylabel('P(True Positive)')
ax.set_title('Question 1 Part 1 ROC curve Approximation')
ax.legend(handles=[l2])
ax.axis('equal')

#save ROC data for part B
with open('Q1_ROC1.npy', 'wb') as f1:
    np.save(f1, gammaResults[:,2:4])

#decision boundary plot
#make decision boundary
NP = 500
x0 = np.linspace(min(samples[:,0]),max(samples[:,0]),NP)
x1 = np.linspace(min(samples[:,1]),max(samples[:,1]),NP)
grid_samples=np.reshape(np.array(np.meshgrid(x0,x1)),(2,NP*NP))

g_pxgivenL0a = np.array([mvn0a.pdf(grid_samples.T)]).T
g_pxgivenL0b = np.array([mvn0b.pdf(grid_samples.T)]).T
g_pxgivenL1 = np.array([mvn1.pdf(grid_samples.T)]).T
g_ratio = g_pxgivenL1/(0.5*g_pxgivenL0a+0.5*g_pxgivenL0b) 
ratio_grid = g_ratio.reshape(NP,NP)

gamma = 65.0/35
fig,ax = plt.subplots()
c1 = ax.contour(x0,x1,ratio_grid,levels=[gamma])

#plot sample data
samples_tp = samples[np.argwhere(truePos)[:,0],:]#wacky syntax. 
s1 = ax.scatter(samples_tp[:,0],samples_tp[:,1],color='g',marker='o',alpha=0.5)
samples_fn = samples[np.argwhere(falseNeg)[:,0],:]
s2 = ax.scatter(samples_fn[:,0],samples_fn[:,1],color='r',marker='o',alpha=0.5)
samples_tn = samples[np.argwhere(trueNeg)[:,0],:]
s3 = ax.scatter(samples_tn[:,0],samples_tn[:,1],color='g',marker='+',alpha=0.5)
samples_fp = samples[np.argwhere(falsePos)[:,0],:]
s4 = ax.scatter(samples_fp[:,0],samples_fp[:,1],color='r',marker='+',alpha=0.5)

#plot settings
margin = 1.03
ax.set(xlim=(min(samples[:,0])*margin, max(samples[:,0])*margin),
        ylim=(min(samples[:,1])*margin, max(samples[:,1])*margin)) 
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_title('Classifier Visualization')
ax.legend(handles=[s1,s2,s3,s4],labels=['Correct Decision Class 1',
                'Wrong Decision Class 1','Correct Decision Class 0','Wrong Decision Class 0'])

plt.show()

