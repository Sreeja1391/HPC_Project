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

## Argument parsing
parser = argparse.ArgumentParser(description='Data processing script.')
parser.add_argument('main_dataset_url', type=str, help='URL of the main dataset')
args = parser.parse_args()
main_dataset_url = args.main_dataset_url

# Function to load a pickle file from a URL
def load_pickle_from_url(raw_url):
    # Convert GitHub URL to raw content URL
    # raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
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
        'loudness': track_data['track'].get('loudness', None),
        'tempo': track_data['track'].get('tempo', None),
    }
    return track_features

# Read the main dataset
main_dataset = pd.read_csv(main_dataset_url)

# Create output directory
genre_name = main_dataset_url.split('/')[-1].split('_')[-1].split('.')[0]

output_dir = f"{genre_name}_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Redirect print statements to a file
sys.stdout = open(os.path.join(output_dir, 'output.txt'), 'w')

# Read the genre dataset and perform a left join
genre_dataset_url = "https://media.githubusercontent.com/media/NiharikaCNR/STAT605-Project/main/data/top_tracks_and_playlists.csv"
genre_data = pd.read_csv(genre_dataset_url)[['track_uri', 'genre_query_tag']]
genre_data = genre_data[genre_data['genre_query_tag'] == genre_name]
merged_data = pd.merge(main_dataset, genre_data, on='track_uri', how='inner')

# Process and merge track data from pickle files
tracks_df = pd.DataFrame(columns=['track_uri','loudness','tempo'])

genre_dir = f"https://github.com/NiharikaCNR/STAT605-Project/raw/main/data/tracks/{genre_name.replace(" ","%20")}"

for track_uri in merged_data[merged_data['genre_query_tag'] == genre]['track_uri']:
    pickle_path = f"{genre_dir}/{track_uri}.pickle"
    print(pickle_path)
    track_data = load_pickle_from_url(pickle_path)
    if track_data is not None:
        processed_data = process_track_data(track_data, track_uri)
        tracks_df = pd.concat([tracks_df, pd.DataFrame([processed_data])], ignore_index=True)



# Selecting the specified columns from processed data frame
final_data = pd.merge(merged_data, tracks_df, on='track_uri', how='left').dropna()
final_data = final_data[['popularity', 'energy', 'acousticness', 'valence', 'loudness', 'tempo']] 

# Compute the correlation matrix for the selected columns
corr = final_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Matrix Visualization for Jazz Genre')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
img_path = os.path.join(output_dir, 'coefficient_plot_Jazz.png')
plt.savefig(img_path, bbox_inches="tight")  # Save instead of show
plt.close()  # Close the plot

# Reset stdout to its default value and close the output file
sys.stdout.close()
sys.stdout = sys.__stdout__
