import matplotlib.pyplot as plt
import sampler
import numpy as np
import math


I = np.array([[1,0], [0,1]])
obj = sampler.MixtureModel([0.25, 0.25, 0.25, 0.25], [sampler.MultiVariateNormal([1,1], I), sampler.MultiVariateNormal([-1,1], I), sampler.MultiVariateNormal([1,-1], I), sampler.MultiVariateNormal([-1,-1], I)])

xlist = []
ylist = []

xlistC=[]
ylistC=[]

circleCounter = 0
for i in range(10000):
    x, y = (obj.sample())
    xlist.append(x)
    ylist.append(y)
    if math.sqrt((x-0.1)**2 + (y-0.2)**2) <= 1:
        circleCounter +=1
        xlistC.append(x)
        ylistC.append(y)

total = len(xlist)
prob = float(circleCounter)/float(total)

print prob


plt.plot(xlist, ylist, 'bo')
plt.plot(xlistC, ylistC, 'ro')
plt.savefig('sample.png')
