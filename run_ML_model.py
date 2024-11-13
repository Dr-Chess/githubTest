import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error 
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.decomposition import PCA

#%% Set random seeds for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

#%% Import data
X = np.loadtxt('X_data.dat')
y = np.loadtxt('y_data.dat')

#%% Feature Importance Analysis 
chord_mean = X[:, :51].mean(axis=1)
twist_mean = X[:, 51:102].mean(axis=1)
qc_y_mean = X[:, 102:153].mean(axis=1)
qc_z_mean = X[:, 153:-1].mean(axis=1)

feature_df = pd.DataFrame({
    'chord_mean': chord_mean,
    'twist_mean': twist_mean,
    'qc_y_mean': qc_y_mean,
    'qc_z_mean': qc_z_mean,
    'target': y
})

# Correlation with target
correlation = feature_df.corr()['target'].sort_values(ascending=False)
print('CORRELATION:')
print(correlation)

#%% Perform PCA
n_components = 20
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
X = X_pca

#%% Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#%% Build model
# params = {'n1': 128, 'n2': 64, 'n3': 32, 'n4': 16, 'learning_rate': 0.001, 'batch_size': 64,
#           'patience': 10, 'factor': 0.1, 'epochs': 100}  # Non-optimized parameters
params = {'n1': 4, 'n2': 4, 'n3': 4, 'n4': 4, 'learning_rate': 0.0034802456735404112,
          'batch_size': 51, 'patience': 60, 'factor': 0.6619610363440357, 'epochs': 106}
          # Optuna parameters

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(params['n1'], activation='relu'),
    Dense(params['n2'], activation='relu'),
    Dense(params['n3'], activation='relu'),
    Dense(params['n4'], activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')

#%% Train model
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=params['factor'], patience=params['patience'], min_lr=1e-6)
history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                    validation_split=0.2, verbose=0, callbacks=[lr_scheduler])

#%% Evaluate model performance

# Generate predictions
y_pred = model.predict(X_test).reshape(-1)
mse_test = mean_squared_error(y_test, y_pred)
print("Test Loss (MSE):", mse_test)

# Look at first few values
print("Predicted FM values:\n", np.round(y_pred[:5], 3))
print("Actual FM values:\n", y_test[:5])

#%% Plotting
plt.figure()
plt.semilogy(history.history['loss'], label='Train Loss', color='blue', linestyle='-', marker='o')
plt.semilogy(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--', marker='x')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Model Training and Validation Loss', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid()
plt.tight_layout()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', color='dodgerblue', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Parity Plot', fontsize=14)
r2 = r2_score(y_test, y_pred)
plt.text(0.05, 0.95, f'R² = {r2:.2f}', fontsize=12, color='darkblue',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', edgecolor='darkblue', facecolor='azure'))
plt.grid(ls='--', lw=0.5)
plt.tight_layout()