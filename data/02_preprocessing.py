import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import joblib

def preprocess_and_balance():
    print("--- Starting Preprocessing (Hybrid Boosting + Balanced) ---")

    # 1. Load Data
    try:
        df = pd.read_csv('aggregated_data.csv')
    except FileNotFoundError:
        print("Error: aggregated_data.csv not found. Run Step 1 first.")
        return

    print(f"DEBUG: Columns found: {df.columns.tolist()}")

    # 2. Clean Column Names
    df.columns = [c.strip() for c in df.columns]
    
    # 3. Identify Key Sensors
    col_water = 'W'
    col_temp = 'T'
    col_speed = 'SPEED_LIMIT'
    
    if col_water not in df.columns or col_temp not in df.columns:
        print(f"CRITICAL: '{col_water}' or '{col_temp}' column missing. Cannot detect Black Ice.")
        return

    # --- 4. Flag Conditions First ---
    # Black Ice Condition: Water > 1000 AND Temp < 0
    df['IsBlackIce'] = ((df[col_water] > 1000) & (df[col_temp] < 0)).astype(int)
    
    # Define Classes
    # Danger = Any Accident (>0) OR Black Ice presence
    # Safe = No Accident (0) AND No Black Ice
    df_danger = df[(df['NearAccidentCount'] > 0) | (df['IsBlackIce'] == 1)].copy()
    df_safe = df[(df['NearAccidentCount'] == 0) & (df['IsBlackIce'] == 0)].copy()
    
    print(f"Original Distribution: {len(df_danger)} Danger, {len(df_safe)} Safe")

    # --- 5. Hybrid Boosting (Black Ice Only) ---
    # We specifically boost the Black Ice subset of the Danger class
    black_ice_real = df_danger[df_danger['IsBlackIce'] == 1]
    
    if len(black_ice_real) > 0:
        print(f">>> STRATEGY: Boosting {len(black_ice_real)} REAL Black Ice rows (x50) <<<")
        # Oversample Black Ice
        black_ice_boosted = resample(black_ice_real, 
                                     replace=True, 
                                     n_samples=len(black_ice_real) * 50, 
                                     random_state=42)
        # Add boosted rows to the Danger pile
        df_danger = pd.concat([df_danger, black_ice_boosted])
    else:
        print(">>> STRATEGY: Injecting SYNTHETIC Black Ice Data (Fallback) <<<")
        n_synthetic = 200
        synthetic_rows = []
        template_row = df.mean(numeric_only=True).to_dict()
        
        for _ in range(n_synthetic):
            new_row = template_row.copy()
            new_row[col_water] = np.random.uniform(1100, 2500)
            new_row[col_temp] = np.random.uniform(-10, -1)
            new_row[col_speed] = np.random.choice([100, 110, 120, 130])
            new_row['NearAccidentCount'] = np.random.uniform(2.0, 5.0)
            new_row['IsBlackIce'] = 1 
            synthetic_rows.append(new_row)
            
        df_synthetic = pd.DataFrame(synthetic_rows)
        # Add synthetic to Danger pile
        df_danger = pd.concat([df_danger, df_synthetic])

    # --- 6. BALANCING (The Critical Fix) ---
    # Now that Danger is huge (due to boosting), we must balance Safe to match it.
    count_danger = len(df_danger)
    count_safe = len(df_safe)
    
    print(f"Post-Boosting Distribution: {count_danger} Danger vs {count_safe} Safe")
    
    if count_safe < count_danger:
        print(f">>> Balancing: Oversampling Safe rows to match {count_danger}...")
        df_safe_balanced = resample(df_safe, 
                                    replace=True, 
                                    n_samples=count_danger, 
                                    random_state=42)
        df_final = pd.concat([df_danger, df_safe_balanced])
    else:
        # Rare case where Safe is still larger, we assume we keep all or undersample
        # Usually boosting makes Danger larger, so we just concat
        df_final = pd.concat([df_danger, df_safe])

    # 7. Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("--- Final Dataset Size ---")
    print(f"Total Rows: {len(df_final)}")
    
    # 8. Prepare Features and Target
    drop_cols = ['Month', 'Day', 'Hour', 'NearAccidentCount', 'IsBlackIce']
    feature_cols = [c for c in df_final.columns if c not in drop_cols]
    
    print(f"Training on Features: {feature_cols}")

    X = df_final[feature_cols]
    y = df_final['NearAccidentCount']
    
    # 9. Handling Missing Values (Imputation)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # 10. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 11. Save Artifacts
    np.save('X_train.npy', X_scaled)
    np.save('y_train.npy', y.values)
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_cols, 'feature_cols.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    
    print("--- Preprocessing Complete. Files saved. ---")

if __name__ == "__main__":
    preprocess_and_balance()