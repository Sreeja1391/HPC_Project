import pandas as pd
import pickle
import os
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

# Argument parsing
parser = argparse.ArgumentParser(description='Data processing script.')
parser.add_argument('main_dataset_url', type=str, help='URL of the main dataset')
args = parser.parse_args()

main_dataset_url = args.main_dataset_url

# Create output directory
output_dir = f"{main_dataset_url.split('/')[-1].split('.')[0]}_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to save plots
def save_plot(plt, filename):
    plt.savefig(os.path.join(output_dir, filename))

# Redirect print statements to a file
sys.stdout = open(os.path.join(output_dir, 'output.txt'), 'w')


# Function to load a pickle file from a URL
def load_pickle_from_url(url):
    # Convert GitHub URL to raw content URL
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
    response = requests.get(raw_url, stream=True)
    if response.status_code == 200:
        file = io.BytesIO(response.content)
        data = pickle.load(file)
        file.close()
        return data
    else:
        print(f"Error downloading file: {raw_url}")
        return None

# Function to process track data from the pickle file
def process_track_data(track_data, track_uri):
    # Extracting fields from the track_data dictionary
    track_features = {
        'track_uri': track_uri,
        'duration': track_data['track'].get('duration', None),
        'loudness': track_data['track'].get('loudness', None),
        'tempo': track_data['track'].get('tempo', None),
        'key': track_data['track'].get('key', None),
        'mode': track_data['track'].get('mode', None),
        'time_signature': track_data['track'].get('time_signature', None),
        'tempo_confidence': track_data['track'].get('tempo_confidence', None),
        'key_confidence': track_data['track'].get('key_confidence', None),
        'mode_confidence': track_data['track'].get('mode_confidence', None),
        'time_signature_confidence': track_data['track'].get('time_signature_confidence', None),
    }
    return track_features

# Read the main dataset
#main_dataset_url = "https://media.githubusercontent.com/media/NiharikaCNR/STAT605-Project/main/data/main_dataset.csv"

#Test Run
#main_dataset = pd.read_csv(main_dataset_url, nrows=100)
main_dataset = pd.read_csv(main_dataset_url)

main_dataset['track_uri'] = main_dataset['track_uri'].str.replace('spotify:track:', '')

# Read the genre dataset and perform a left join
genre_dataset_url = "https://media.githubusercontent.com/media/NiharikaCNR/STAT605-Project/main/data/top_tracks_and_playlists.csv"
genre_data = pd.read_csv(genre_dataset_url)[['track_uri', 'genre_query_tag']]
merged_data = pd.merge(main_dataset, genre_data, on='track_uri', how='left')

# Process and merge track data from pickle files
genres = merged_data['genre_query_tag'].unique()
for genre in genres:
    genre_dir = f"https://github.com/NiharikaCNR/STAT605-Project/tree/main/data/tracks/{genre}"
    for track_uri in merged_data[merged_data['genre_query_tag'] == genre]['track_uri']:
        pickle_path = f"{genre_dir}/{track_uri}.pickle"
        track_data = load_pickle_from_url(pickle_path)
        if track_data is not None:
            processed_data = process_track_data(track_data, track_uri)
            track_df = pd.DataFrame([processed_data])
            merged_data = pd.merge(merged_data, track_df, on='track_uri', how='left')

# Calculate correlation matrix
correlation_matrix = merged_data.corr()

# Select correlations with the 'popularity' column
popularity_correlation = correlation_matrix['popularity']

# Drop the popularity column itself to avoid self-correlation
popularity_correlation = popularity_correlation.drop(labels=['popularity'])

# Sort by absolute value in descending order to get top correlated features
top_correlated_features = popularity_correlation.abs().sort_values(ascending=False).head(5)

# Print the top 5 correlated features
print(top_correlated_features)

# Selecting the top 5 features and the target variable 'popularity'
X = merged_data[top_correlated_features.index]
y = merged_data['popularity']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the linear regression model
model = LinearRegression()

# Fitting the model to the training data
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the model coefficients and performance metrics
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=residuals, color="g")
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='red', linestyle='--')
save_plot(plt, 'residuals_plot.png')  # Save instead of show
plt.close()  # Close the plot

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
save_plot(plt, 'actualvspredicted_plot.png')  # Save instead of show
plt.close()  # Close the plot

# Coefficient Plot
feature_coefficients = pd.Series(model.coef_, index=top_correlated_features.index)
plt.figure(figsize=(10, 6))
feature_coefficients.sort_values().plot(kind='barh')
plt.title('Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
save_plot(plt, 'cieffifient_plot.png')  # Save instead of show
plt.close()  # Close the plot

# Reset stdout to its default value and close the output file
sys.stdout.close()
sys.stdout = sys.__stdout__
