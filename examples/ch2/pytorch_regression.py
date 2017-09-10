import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random, IPython

import torch
from torch import nn, optim
from torch.autograd import Variable

url = "https://raw.githubusercontent.com/nikcheerla/deeplearningschool/master/data/housing.csv"
data = pd.read_csv(url)

area = Variable(torch.Tensor(np.array(data["Square Feet (Millions)"])))
price =  Variable(torch.Tensor(np.array(data["Price ($, Millions)"])))

M = Variable(torch.Tensor([100]), requires_grad=True)
loss = nn.MSELoss()
optimizer = optim.Adam([M], lr=0.1)

def predict(M, input_area):
    return M.expand_as(input_area)*input_area**(0.8)

# Evaluates MSE of model y = Mx
def evaluate(M):
    price_predicted = predict(M, area)
    MSE = loss(price_predicted, price).data.cpu().numpy().mean()
    return MSE

def learn(M):
    j = random.randint(0, len(area) - 1)
    price_predicted = predict(M, area[j])
    error = loss(price_predicted, price[j])

    error.backward()
    optimizer.step()
    optimizer.zero_grad()
    return M

print ("Initial value of M: ", M)
for i in range(0, 50000):
	M = learn(M)
	if i % 100 == 0:
		print ("Loss: ", evaluate(M), "(M =", M.data.cpu().numpy().mean(), ")")

