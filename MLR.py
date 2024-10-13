import numpy as np
import pandas as pd

# Generate some sample data
np.random.seed(0)
n = 50
X1 = np.random.rand(n)
X2 = np.random.rand(n)
# Create a dependent variable Y with some noise
Y = 3 + 2 * X1 + 5 * X2 + np.random.normal(0, 0.1, n)

# Create a DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Define the independent variables and dependent variable
X = data[['X1', 'X2']].values
Y = data['Y'].values

# Add a column of ones to include the intercept (β0)
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

# Calculate the estimated parameters (β̂) using the normal equation
# β̂ = (X^TX)^{-1}X^TY
X_transpose = X_with_intercept.T
beta_hat = np.linalg.inv(X_transpose @ X_with_intercept) @ (X_transpose @ Y)

# Display the estimated parameters
print("Estimated parameters (β̂):", beta_hat)
