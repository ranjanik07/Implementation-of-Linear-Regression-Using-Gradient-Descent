# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset (population, profit).
2. Initialize θ = 0, set learning rate α, iterations.
3. Add bias term to X.
4. Repeat until convergence / iterations:

     Predict ℎ𝜃(𝑥)=𝑋⋅𝜃
	
     ​Compute error = prediction – y
  
     Update 𝜃=𝜃−𝛼𝑚(𝑋𝑇⋅𝑒𝑟𝑟𝑜𝑟)

6. Output final θ.
7. Predict profit for given population.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: RANJANI K
RegisterNumber: 212224230220
```
```
import numpy as np

data = np.array([
    [6.1101, 17.592],
    [5.5277, 9.1302],
    [8.5186, 13.662],
    [7.0032, 11.854],
    [5.8598, 6.8233],
    [8.3829, 11.886],
    [7.4764, 4.3483],
    [8.5781, 12.000],
    [6.4862, 6.5987],
    [5.0546, 3.8166]
])

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
m = len(y)
X = np.hstack((np.ones((m, 1)), X))
theta = np.zeros((2, 1))
alpha = 0.01
iterations = 1500

def compute_cost(X, y, theta):
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        theta -= (alpha / m) * (X.T @ errors)
    return theta

print("Initial cost:", compute_cost(X, y, theta))
theta = gradient_descent(X, y, theta, alpha, iterations)
print("Theta after training:", theta.ravel())
print("Final cost:", compute_cost(X, y, theta))

population = 7
prediction = np.array([1, population]) @ theta
print(f"Predicted profit for city with population {population*10000}: {prediction[0]*10000:.2f}")
```
## Output:
<img width="653" height="112" alt="Screenshot 2025-09-03 153609" src="https://github.com/user-attachments/assets/599572aa-28e6-42a6-8c0d-9781ea45b6e9" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
