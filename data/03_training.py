import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import pandas as pd
import matplotlib.pyplot as plt # Optional, but good for plotting history if needed

# --- Custom Metric: R-Squared (Coefficient of Determination) ---
def r_squared(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# Load Data
print("--- Loading Processed Data ---")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')

print(f"Data Shape: {X_train.shape}")
print(f"Target Mean: {np.mean(y_train):.4f} (If this is high, model will be biased towards accidents)")

# --- Build Neural Network ---
# Simplified architecture for small dataset (~900 rows)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    
    # Reduced complexity to prevent overfitting on small data
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(16, activation='relu'),
    
    # Output Layer 
    layers.Dense(1, activation='softplus') 
])

# Optimizer with lower learning rate for stability
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=opt,
    loss='mse',
    metrics=['mse', 'mae', r_squared]
)

# Early Stopping to prevent overfitting (and negative R2 due to diverging validation loss)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

print("--- Training Model ---")
history = model.fit(
    X_train, y_train,
    epochs=100, # Increased epochs, but early stopping handles cutoff
    batch_size=16, # Smaller batch size for small dataset updates
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --- Evaluate on Validation Data (Last Split) ---
print("\n--- Final Model Evaluation ---")
# Taking the last 20% manually just to print a clean report (mimicking validation split)
split_idx = int(len(X_train) * 0.8)
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]

results = model.evaluate(X_val, y_val, verbose=0)
print(f"Mean Squared Error (MSE): {results[1]:.4f}")
print(f"Mean Absolute Error (MAE): {results[2]:.4f}")
print(f"R-Squared (R2): {results[3]:.4f}")

# Save
model.save('speed_limit_model.keras')
print("--- Model Saved ---")

# ==========================================
# LOGIC FOR PROPOSING SPEED LIMIT
# ==========================================

def propose_speed_limit(current_weather_row_df):
    
    possible_speeds = [130, 120, 110, 100, 90, 80, 70, 60, 50]
    
    try:
        speed_col_idx = feature_cols.index('SPEED_LIMIT')
    except ValueError:
        print("Error: SPEED_LIMIT column not found in training features.")
        return 80

    best_safe_speed = 30 # Fallback
    
    # Base vector from current sensors
    input_vector = current_weather_row_df[feature_cols].values[0].copy()
    
    print("\n--- Calculating Proposed Speed ---")
    
    for speed in possible_speeds:
        # 1. Update speed in raw vector
        raw_vector = input_vector.copy()
        raw_vector[speed_col_idx] = speed
        
        # 2. Scale the vector
        # Create a DataFrame with column names to avoid UserWarning
        df_for_scale = pd.DataFrame([raw_vector], columns=feature_cols)
        scaled_vector = scaler.transform(df_for_scale)
        
        # 3. Predict risk
        predicted_accidents = model.predict(scaled_vector, verbose=0)[0][0]
        
        print(f"Speed: {speed} km/h -> Predicted Accidents: {predicted_accidents:.4f}")
        
        # REQ1: Average at most one near-accident per hour
        if predicted_accidents < 1.0:
            best_safe_speed = speed
            break 
            
    return best_safe_speed

# --- Demo Test ---
print("\n--- DEMO: Proposing Speed for a fake scenario ---")
mean_values = scaler.inverse_transform([np.mean(X_train, axis=0)])
demo_df = pd.DataFrame(mean_values, columns=feature_cols)

# Modify specific conditions to simulate bad weather
if 'W' in demo_df.columns: demo_df['W'] = 1200 
if 'T' in demo_df.columns: demo_df['T'] = -2   
if 'L' in demo_df.columns: demo_df['L'] = 200  

recommended_speed = propose_speed_limit(demo_df)
print(f"\nFinal Proposed Speed Limit: {recommended_speed} km/h")