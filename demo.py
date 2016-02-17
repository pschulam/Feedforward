import imp
import random

import numpy as np
import matplotlib.pyplot as plt

import nn
imp.reload(nn)

np.random.seed(0)

x = 2 * np.pi * np.sort(np.random.uniform(size=300))
X = nn.add_bias(x[:, None])
sinx = nn.rescale(np.sin(x), -1, 1) + np.random.normal(scale=0.1, size=300)
w = nn.learn_neuron(X, sinx)

xgrid = np.linspace(0, 2 * np.pi, 200)
Xgrid = nn.add_bias(xgrid[:, None])

plt.plot(x, sinx, 'o', label='Observations')
plt.plot(xgrid, [nn.neuron(w, x) for x in Xgrid], label='Single Neuron')
plt.xlim(0, 2*np.pi)

model = nn.FeedforwardNN([1, 4, 1])
plt.plot(xgrid, np.array([model.predict(x) for x in xgrid]), label='Initial NN')

train = list(zip(x, sinx))
rate = 5.0
batch_size = 10

random.seed(0)

model = nn.learn(model, train, rate, batch_size, 1000)
plt.plot(xgrid, np.array([model.predict(x) for x in xgrid]), label='Trained NN')

plt.legend()

plt.show()
