import numpy as np
import matplotlib.pyplot as plt

def parse_data(data_str):
    """Parse the input data string into  features and targets"""
    lines = data_str.strip().split('\n')
    sizes = []
    prices = []

    for line in lines[1:]:
        values = line.split(',')
        sizes.append(float(values[-2]))
        prices.append(float(values[-1]))

    return np.array(sizes), np.array(prices)

def mean_squared_error(y_true, y_pred):
    """Compute the mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent_step(X, y, m, c, learning_rate):
    """
    Perform one step of gradient descent
    Returns new m, c and current error
    """
    y_pred = m * X + c
    error = mean_squared_error(y, y_pred)

    #Compute gradients
    m_gradient = -2 * np.mean(X * (y - y_pred))
    c_gradient = -2 * np.mean((y - y_pred))

    # Update parameters
    m = m - learning_rate * m_gradient
    c = c - learning_rate * c_gradient

    return m, c, error

# Parse the input data
data = """LOC,FUR,AMB,PROX_SCH,PROX_ROAD,PROX_MALL,WATER,HK_SER,SIZE,PRICE
karen,yes,serene,no,yes,yes,yes,yes,32.50234527,31.70700585
madaraka,yes,semi_serene,yes,yes,yes,no,no,53.42680403,68.77759598
karen,no,noisy,no,yes,yes,yes,yes,61.53035803,62.5623823
karen,yes,semi_serene,no,no,no,yes,yes,47.47563963,71.54663223
buruburu,no,semi_serene,no,yes,yes,yes,yes,59.81320787,87.23092513
donholm,no,serene,no,no,yes,no,yes,55.14218841,78.21151827
langata,no,very_noisy,yes,yes,no,no,yes,52.21179669,79.64197305
langata,yes,serene,no,no,yes,yes,no,39.29956669,59.17148932
donholm,yes,semi_serene,yes,no,no,yes,no,48.10504169,75.3312423
karen,yes,serene,no,no,no,no,no,52.55001444,71.30087989
madaraka,yes,noisy,yes,yes,no,yes,yes,45.41973014,55.16567715
langata,no,semi_serene,yes,no,yes,yes,yes,54.35163488,82.47884676
buruburu,yes,semi_serene,yes,yes,no,no,no,44.1640495,62.00892325
karen,yes,semi_serene,yes,yes,yes,yes,yes,58.16847072,75.39287043"""

X, y = parse_data(data)

# Normalize features for better training
X_mean = np.mean(X)
X_std = np.std(X)
X_norm = (X - X_mean) / X_std

# Initialize parameters
m = 0.0
c = np.mean(y)
learning_rate = 0.1
epochs = 10

# Training Loop
print("Training the model...")
print("Initial parameters: m =", m, "c =", c)
print("\nTraining Progress:")
print("Epoch | Error")
print("-" * 20)

for epoch in range(epochs):
    m, c, error = gradient_descent_step(X_norm, y, m, c, learning_rate)
    print(f" {epoch+1:5d} | {error:.4f}")

# Convert the learned parameters back to original scale
m_original = m / X_std
c_original = c - m * X_mean / X_std

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
y_pred = m_original * X + c_original
plt.plot(X, y_pred, color='red', label='Line of best fit')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Real Estate Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

print("\nFinal parameters (in original scale):")
print(f"Slope (m): {m_original:.4f}")
print(f"Intercept (c): {c_original:.4f}")

