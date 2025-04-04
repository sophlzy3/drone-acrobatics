import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split(csv_path, target_column, test_size=0.2, val_size=0.25, random_state=42, normalize=True):
    """
    Load a CSV file containing numerical data and split it into train, validation, and test sets.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    target_column : str
        Name of the column to use as target/label
    test_size : float, default=0.2
        Proportion of the data to include in the test split
    val_size : float, default=0.25
        Proportion of the training data to include in the validation split
    random_state : int, default=42
        Random seed for reproducibility
    normalize : bool, default=True
        Whether to normalize the features using StandardScaler
        
    Returns:
    --------
    train_x, val_x, test_x, train_y, val_y, test_y
    scaler : sklearn.preprocessing.StandardScaler or None
        The fitted scaler if normalize=True, None otherwise
    feature_names : list
        Names of the feature columns
    """
    # Load the dataset
    try:
        data = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        # Try with different encodings if the default fails
        data = pd.read_csv(csv_path, encoding='latin-1')
    
    # Check if target_column exists in the data
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the CSV file.")
    
    # Extract features and labels
    features = data.drop(columns=[target_column])
    labels = data[target_column].values
    
    # Store feature names for later reference
    feature_names = features.columns.tolist()
    
    # Convert features to numpy array
    features = features.values
    
    # Split into train, test sets
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    # Split train into train, validation sets
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=val_size, random_state=random_state
    )
    
    # Normalize features if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)
    
    return train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names


# # Example usage:
# if __name__ == "__main__":
#     # Replace with your actual CSV file and target column name
#     csv_path = 'your_numerical_data.csv'
#     target_column = 'target'  # Change this to your actual target column name
    
#     # Load and split the data
#     train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names = load_and_split(
#         csv_path=csv_path, 
#         target_column=target_column
#     )
    
#     # Print dataset info
#     print(f"Number of features: {len(feature_names)}")
#     print(f"Feature names: {feature_names}")
#     print(f"Training data shape: {train_x.shape}")
#     print(f"Validation data shape: {val_x.shape}")
#     print(f"Test data shape: {test_x.shape}")
    
#     # Check the class distribution if it's a classification problem
#     if len(np.unique(train_y)) < 10:  # Heuristic for classification vs regression
#         unique_classes, train_counts = np.unique(train_y, return_counts=True)
#         print("\nClass distribution in training set:")
#         for cls, count in zip(unique_classes, train_counts):
#             print(f"  Class {cls}: {count} samples ({count/len(train_y):.2%})")