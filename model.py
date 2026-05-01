import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# --- Configuration ---
DATA_PATH = r"House Price Prediction Dataset.csv" # Ensure this path matches your setup

def load_data(file_path):
    """Loads dataset and performs basic validation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Dataset not found at {file_path}")
    
    print(f"[*] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    return df

def build_pipeline(X):
    """Builds a scikit-learn processing and modeling pipeline."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline, categorical_cols, numerical_cols

def show_feature_importance(pipeline, X_train, categorical_cols, numerical_cols):
    """Extracts and displays the most important features driving the price."""
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get the names of the one-hot encoded columns
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    
    # Combine with numerical columns
    all_feature_names = list(numerical_cols) + list(cat_feature_names)
    
    # Map importances to feature names
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance (%)': importances * 100
    }).sort_values(by='Importance (%)', ascending=False).head(5)
    
    print("\n" + "="*40)
    print(" 🏆 TOP 5 PRICE DRIVERS (Feature Importance)")
    print("="*40)
    for index, row in feature_importance_df.iterrows():
        print(f" - {row['Feature']:<20}: {row['Importance (%)']:.2f}%")
    print("="*40 + "\n")

def interactive_prediction(pipeline, X_train, categorical_cols, numerical_cols):
    """Robust terminal UI that dynamically restricts user inputs."""
    print("\n" + "="*50)
    print("   🏠 INTERACTIVE HOUSE PRICE PREDICTOR 🏠")
    print("="*50)
    print("Type 'quit' at any prompt to exit.\n")

    # Extract valid options and ranges directly from the training data
    valid_categories = {col: X_train[col].dropna().unique().tolist() for col in categorical_cols}
    valid_ranges = {col: (X_train[col].min(), X_train[col].max()) for col in numerical_cols}

    while True:
        user_input_data = {}
        
        for col in X_train.columns:
            while True:
                # 1. Handle Categorical Inputs (Text)
                if col in categorical_cols:
                    options = valid_categories[col]
                    options_str = ", ".join([f"'{opt}'" for opt in options])
                    val = input(f"Enter {col} [{options_str}]: ").strip()
                    
                    if val.lower() == 'quit': return
                    
                    # Case-insensitive matching
                    match = next((opt for opt in options if opt.lower() == val.lower()), None)
                    if match:
                        user_input_data[col] = [match]
                        break
                    else:
                        print(f"  ❌ Invalid input. You must type one of: {options_str}")

                # 2. Handle Numerical Inputs
                elif col in numerical_cols:
                    min_val, max_val = valid_ranges[col]
                    dtype = X_train[col].dtype
                    val = input(f"Enter {col} (Range: {min_val} to {max_val}): ").strip()
                    
                    if val.lower() == 'quit': return
                    
                    try:
                        if pd.api.types.is_integer_dtype(dtype):
                            num_val = int(val)
                        else:
                            num_val = float(val)
                            
                        # Optional: Warn the user if they input something totally unrealistic based on training data
                        if num_val < min_val or num_val > max_val:
                            print(f"  ⚠️ Warning: {num_val} is outside the training data range. Prediction may be inaccurate.")
                            
                        user_input_data[col] = [num_val]
                        break 
                    except ValueError:
                        print(f"  ❌ Invalid input. Please enter a number for {col}.")
        
        # Make the prediction
        input_df = pd.DataFrame(user_input_data)
        predicted_price = pipeline.predict(input_df)[0]
        
        print("\n" + "-"*50)
        print(f"🎯 ESTIMATED HOUSE PRICE: ${predicted_price:,.2f}")
        print("-" * 50 + "\n")
        
        again = input("Predict another house? (y/n): ")
        if again.lower() != 'y':
            print("Exiting predictor. Goodbye!")
            break
        print("\n")

def main():
    try:
        df = load_data(DATA_PATH)
        
        target_col = 'Price' if 'Price' in df.columns else 'price'
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        print("[*] Building and training model pipeline...")
        pipeline, cat_cols, num_cols = build_pipeline(X)
        pipeline.fit(X_train, y_train)
        print("[*] Training complete.\n")
        
        # Display model accuracy
        r2 = r2_score(y_test, pipeline.predict(X_test))
        print(f"[*] Model Accuracy (R-squared on unseen test data): {r2:.4f}\n")

        # Display what factors matter most
        show_feature_importance(pipeline, X_train, cat_cols, num_cols)

        # Launch UI
        interactive_prediction(pipeline, X_train, cat_cols, num_cols)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()