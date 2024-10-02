
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import mapping
from sklearn.model_selection import train_test_split
import warnings

# Ignore shapely deprecation warnings
warnings.filterwarnings('ignore')

# Function to load and process the geospatial training data
def load_geospatial_data(train_data_path, raster_path):
    # Load the GeoDataFrame (training points with labels)
    import geopandas as gpd
    train_data = gpd.read_file(train_data_path)

    # Get the unique class labels from the training data
    class_labels = np.unique(train_data['Class'])
    
    # Load the raster image (Sentinel-2 or other imagery)
    with rasterio.open(raster_path) as src:
        s2_data = src.read()
        s2_meta = src.meta
        crs = src.crs
        print(f'Coordinate Reference System: {crs}')
    
    # Rasterize the training points to create a mask
    train_mask = rasterize(
        [(mapping(point), class_label) for point, class_label in zip(train_data.geometry, train_data['Class'])],
        out_shape=s2_data.shape[-2:], 
        transform=src.transform, 
        fill=-1, 
        dtype='int16'
    )
    
    # Flatten the Sentinel-2 image into a 2D array
    s2_data_2d = s2_data.reshape(s2_data.shape[0], -1).T
    
    # Extract the training samples and labels
    train_samples = s2_data_2d[train_mask.flatten() != -1]
    train_labels = train_mask[train_mask != -1]

    return train_samples, train_labels, s2_meta

# Split the data into training and testing sets
def split_train_test(train_samples, train_labels, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        train_samples, train_labels, test_size=test_size, random_state=42, stratify=train_labels
    )
    return X_train, X_test, y_train, y_test

# Main function to run the data loading and splitting process
def main():
    # File paths
    train_data_path = 'path_to_training_points.shp'  # Replace with your path
    raster_path = 'LillyBandsOBIADSM1.tif'           # Replace with your raster image path

    # Load and process the geospatial data
    train_samples, train_labels, s2_meta = load_geospatial_data(train_data_path, raster_path)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(train_samples, train_labels)
    
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Testing set size: {X_test.shape[0]}')

if __name__ == '__main__':
    main()
    
    
