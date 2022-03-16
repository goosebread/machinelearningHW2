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

x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
samples = np.array(np.meshgrid(x,y))
samplesr=samples.reshape(2,10000)

pxgivenL0a = np.array([mvn0a.pdf(samplesr.T)]).T
pxgivenL0b = np.array([mvn0b.pdf(samplesr.T)]).T
pxgivenL1 = np.array([mvn1.pdf(samplesr.T)]).T
ratio = pxgivenL1/(0.5*pxgivenL0a+0.5*pxgivenL0b) 

vals = ratio.reshape(100,100)

gamma = 65.0/35
fig,ax = plt.subplots()
cs = ax.contour(x,y,vals,levels=[0.5*gamma,gamma,2*gamma])
cs.changed()

plt.show()
"""

discriminantScoreGridValues =...
log(evalGaussian([h(:)';v(:)'],mu1,Sigma1))-log(evalGMM([h(:)';v(:)'],...
alpha0,mu0,Sigma0)) - logGamma_ideal;
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
contour(horizontalGrid,verticalGrid,...
discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); %
plot equilevel contours of the discriminant function
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0',...
'Wrong decisions for data from Class 0',...
'Wrong decisions for data from Class 1',...
'Correct decisions for data from Class 1',...
'Equilevel contours of the discriminant function' ),
end

"""