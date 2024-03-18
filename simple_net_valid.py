import numpy as np
import matplotlib.pyplot as plt
from simple_net import SimpleNet
import torch


func = lambda x: 2 * np.cos(0.5 * x) + 1.5 * np.sin(3 * x)
x = np.linspace(-5, 5, 900)
x_train, x_test = x[::2], x[1::2]
y_train, y_test = func(x_train), func(x_test)

x_test = torch.from_numpy(x_test).to(dtype = torch.float)
model = torch.load('model.pth')
model.eval()
model.to(device = 'cpu')


predictions = model(torch.unsqueeze(x_test, dim = 1)).squeeze(dim = 1).detach().numpy()
plt.plot(x_test, y_test, x_test, predictions)
plt.legend(['y(x)', 'model(x)'])
plt.show()