---
title: k-means
date: 2020-02-16 00:18:05
tags: Machinelearning
---

The main code part

```python

import numpy as np
import matplotlib.pyplot as plt
import random

class kmeans:
    def __init__(self, data, k):
        self.x = data
        self.k = k
        self.y = self.forward(self.x, self.k)

    def distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def rand_cent(self, data, k):
        n, dim = data.shape
        cent = np.zeros((k, dim))
        for i in range(k):
            cent[i,:] = data[random.randint(0, n),:]
        return cent

    def forward(self, data, k):
        n, dim = data.shape
        cent = self.rand_cent(data, k)
        y = np.zeros(n)
        dis = np.zeros(k)
        clu = dict()
        for j in range(k):
            clu[j] = [cent[j,:]]
        for i in range(n):
            for j in range(k):
                dis[j] = self.distance(cent[j,:], data[i,:])
            c = np.argmin(dis)
            clu[c].append(data[i,:])
            cent[c,:] = np.mean(np.array(clu[c]))
            y[i] = c
        return y

    def res(self):
        return self.y
```

Test on the iris dataset
```python

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
data = X[:,[0,2]]
data = X[:,[1,2]]
plt.scatter(data[:,0],data[:,1])

```
![0001.png](https://upload-images.jianshu.io/upload_images/18864424-d79a75bdf6d70076.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
k = 4
#k is the number of cluster
km = kmeans(data, k)
y_p=km.res()

for i in range(k):
    y0 = list(np.argwhere(y_p==i).reshape(1,-1)[0])
    plt.scatter(data[y0,0],data[y0,1])
```
![0002.png](https://upload-images.jianshu.io/upload_images/18864424-f17935d7eb574948.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)