import time
import random
import optuna
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau


#%% Set random seeds for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

#%% Import data
X = np.loadtxt('X_data.dat')
y = np.loadtxt('y_data.dat')

#%% Preprocess data
n_components = 20  # Set the number of components you want to reduce to
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
X = X_pca
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters to tune
    n1 = trial.suggest_int('n1', 4, 128)
    n2 = trial.suggest_int('n2', 4, n1)
    n3 = trial.suggest_int('n3', 4, n2)
    n4 = trial.suggest_int('n4', 4, n3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    patience = trial.suggest_int('patience', 4, 64)
    factor = trial.suggest_float('factor', 0.1, 1)
    epochs = trial.suggest_int('epochs', 100, 200)
    
    # Build the model using the hyperparameters
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(n1, activation='relu'), 
        Dense(n2, activation='relu'),
        Dense(n3, activation='relu'),
        Dense(n4, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Train the model
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=1e-6)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=0, callbacks=[lr_scheduler])
    # history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size,
    #                     validation_split=0.2, verbose=0)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test).reshape(-1)
    mse_test = mean_squared_error(y_test, y_pred)
    
    # Return the test MSE (Optuna minimizes this)
    return mse_test

# Run the Optuna optimization
start = time.time()
study = optuna.create_study(direction='minimize')  # We want to minimize MSE
study.optimize(objective, n_trials=50)  # Run 50 trials
stop = time.time()
runtime = stop - start
print(f"Execution time: {round(runtime,5)} seconds")

# Get the best hyperparameters
best_params = study.best_trial.params 
print(f"Best trial: {best_params}")
