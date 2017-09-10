import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

url = "https://raw.githubusercontent.com/nikcheerla/deeplearningschool/master/data/housing.csv"
data = pd.read_csv(url)
area = data["Square Feet (Millions)"]
price = data["Price ($, Millions)"]

sns.jointplot(area, price-24.0*area)
plt.show()

# Takes in M + numpy array of areas, returns price predictions
M = 1

def predict(M, input_area):
    return M*input_area

# Evaluates MSE of model y = Mx
def evaluate(M):
    price_predicted = [predict(M, x) for x in area]
    MSE = ((price - price_predicted)**2).mean()
    return MSE

def learn(M):
    j = random.randint(0, len(area) - 1) # choosing a random sample
    deriv = 2*(M*area[j]- price[j])*M # derivative calculation
    M = M - 0.005*deriv # SGD update step
    return M

print ("Initial value of M: ", M)
for i in range(0, 2000):
	M = learn(M)
	if i % 100 == 0:
		print ("Loss: ", evaluate(M), "(M =", M, ")")

