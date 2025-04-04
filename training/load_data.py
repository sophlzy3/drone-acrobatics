import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split(features_csv_path, labels_csv_path, test_size=0.2, val_size=0.25, random_state=42, normalize=True):
    """
    Load features and labels from separate CSV files and split them into train, validation, and test sets.
    
    Parameters:
    -----------
    features_csv_path : str
        Path to the CSV file containing feature data
    labels_csv_path : str
        Path to the CSV file containing label data
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    val_size : float, default=0.25
        Proportion of the training set to include in the validation split
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split
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
    # Load the datasets
    try:
        features_data = pd.read_csv(features_csv_path)
        labels_data = pd.read_csv(labels_csv_path)
    except UnicodeDecodeError:
        # Try with different encodings if the default fails
        features_data = pd.read_csv(features_csv_path, encoding='latin-1')
        labels_data = pd.read_csv(labels_csv_path, encoding='latin-1')
    
    # Check if the number of rows match
    if len(features_data) != len(labels_data):
        raise ValueError(f"Number of rows in features file ({len(features_data)}) does not match number of rows in labels file ({len(labels_data)})")
    
    # Extract features and labels
    features = features_data.values
    labels = labels_data.values
    
    # Store feature names for later reference
    feature_names = features_data.columns.tolist()
    
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


# Example usage:
# if __name__ == "__main__":
#     # Replace with your actual CSV files
#     features_csv_path = 'your_features_data.csv'
#     labels_csv_path = 'your_labels_data.csv'
    
#     # Load and split the data
#     train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names = load_and_split(
#         features_csv_path=features_csv_path, 
#         labels_csv_path=labels_csv_path
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