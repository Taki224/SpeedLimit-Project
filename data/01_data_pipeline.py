import pandas as pd
import numpy as np
import os

def load_and_clean_data():
    print("--- Loading Data ---")
    
    # Load metadata mappings
    # Use ; as delimiter based on file snippets
    sensors_meta = pd.read_csv('Case_Study_Speed_Limit_Sensors_V100.csv', sep=';')
    sensor_types = pd.read_csv('Case_Study_Speed_Limit_SensorTypes_V100.csv', sep=';')
    
    # Load Readings (Use decimal=',' because snippet showed "0,00")
    readings = pd.read_csv('Case_Study_Speed_Limit_SensorReadingsYearN_V100.csv', sep=';', decimal=',')
    
    # Load Accidents
    accidents = pd.read_csv('Case_Study_Speed_Limit_AccidentsYearN_V100.csv', sep=';', decimal=',')

    print("--- Mapping Sensors ---")
    # Create a map of SensorID -> SensorTypeCode (e.g., 1 -> 'W')
    id_to_type = dict(zip(sensors_meta['SensorID'], sensors_meta['SensorTypeCode']))
    
    # Add Type to readings
    # Note: SensorID -1 appears in snippets (likely Current Speed Limit), we handle it manually
    readings['Type'] = readings['Sensor'].map(id_to_type)
    readings.loc[readings['Sensor'] == -1, 'Type'] = 'SPEED_LIMIT'

    # Filter out unknown types if any remain (except -1)
    readings = readings.dropna(subset=['Type'])

    print("--- Aggregating by Hour ---")
    # We aggregate by Hour to match general weather patterns and accident rates
    # Group by [Month, Day, Hour] and Pivot based on Sensor Type
    
    # 1. Pivot the sensor table
    # We take the mean value if multiple readings occur within the hour
    pivot_sensors = pd.pivot_table(
        readings, 
        values='Value', 
        index=['Month', 'Day', 'Hour'], 
        columns='Type', 
        aggfunc='mean'
    ).reset_index()
    
    # Fill NaNs: Forward fill first (weather persists), then 0
    pivot_sensors = pivot_sensors.ffill().fillna(0)

    # 2. Aggregating Accidents
    # We need to count how many accidents happened in that specific hour
    accident_counts = accidents.groupby(['Month', 'Day', 'Hour']).size().reset_index(name='NearAccidentCount')
    
    print("--- Merging Data ---")
    # Merge sensors with accidents
    # usage of 'left' ensures we keep hours where 0 accidents happened
    final_df = pd.merge(pivot_sensors, accident_counts, on=['Month', 'Day', 'Hour'], how='left')
    
    # Fill missing accident counts with 0 (since no record means no accident)
    final_df['NearAccidentCount'] = final_df['NearAccidentCount'].fillna(0)
    
    # Rename columns for clarity based on PDF codes
    # T=Temp, W=Water, L=Light, SPEED_LIMIT=CurrentLimit
    # If duplicates exist (like multiple T sensors), pivot_table averaged them automatically
    
    print(f"Total Rows Generated: {len(final_df)}")
    print("Columns:", final_df.columns.tolist())
    
    # Save
    final_df.to_csv('aggregated_data.csv', index=False)
    print("--- Saved to aggregated_data.csv ---")

if __name__ == "__main__":
    load_and_clean_data()