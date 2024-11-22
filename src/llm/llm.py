#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import ast


# In[4]:


client_id = '75d0ab19dcdc4db7821a27bf07df72a0'  # Replace with your Spotify client ID
client_secret = 'f64897e446834d7cb83b1c90916242df'  # Replace with your Spotify client secret
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to extract song names from Spotify URLs
def get_song_names_from_url(song_uris):
    song_names = []
    for uri in song_uris:
        try:
            track_id = uri.split(":")[-1]  # Extract the track ID from the URI
            track_info = sp.track(track_id)  # Get track information
            song_name = track_info['name']  # Extract song name
            artist_name = track_info['artists'][0]['name']  # Extract artist name
            song_names.append(f"{song_name} by {artist_name}")
        except Exception as e:
            song_names.append(f"Error retrieving song for URI: {uri} ({e})")
    return song_names


# In[11]:


with open("api_key.txt", "r") as f:
    key = f.readline().strip()

from openai import OpenAI
client = OpenAI(api_key=key)


# In[8]:


data = pd.read_csv('../../datasets/seven_day_listening_history.csv')


# In[12]:


def get_prompt_features(prompt):
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Specify the model
            messages=[
                {
                    "role": "system",
                    "content": "You are a music expert.",
                },
                {
                    "role": "user",
                    "content": f"Given the playlist prompt: '{prompt}', assign values between 0 and 1 for these features: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, and tempo. Respond with only a list of values in Python list format.",
                }
            ]
        )

        # Extract the content from the response
        response_message = chat_completion.choices[0].message.content.strip()
        print("Response from LLM:", response_message)  # Debugging step

        # Directly evaluate the list (since the LLM is instructed to return a Python-style list)
        feature_scores = eval(response_message)
        return feature_scores
    except Exception as e:
        print(f"Error fetching prompt features: {e}")
        return None


# In[13]:


features = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "valence", "tempo", "loudness", "speechiness"
]
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
song_vectors = data[features].values
normalized_vectors = song_vectors / np.linalg.norm(song_vectors, axis=1, keepdims=True)
unique_indices = list({tuple(vec): i for i, vec in enumerate(normalized_vectors)}.values())
data = data.iloc[unique_indices].reset_index(drop=True)

# Compute Euclidean distances
def compute_distance(song, prompt_features):
    song_features = song[features].values
    return np.sqrt(np.sum((song_features - np.array(prompt_features)) ** 2))

# Main logic to generate a playlist
def generate_playlist(prompt, user_id=1):
    # Fetch features for the prompt
    prompt_features = get_prompt_features(prompt)
    if prompt_features is None:
        print("Failed to fetch prompt features.")
        return

    # Filter data for the user
    user_data = data[data["user_id"] == user_id]

    # Calculate distances
    user_data["distance"] = user_data.apply(lambda row: compute_distance(row, prompt_features), axis=1)

    # Get top 5 recommended URIs
    recommended_songs = user_data.sort_values(by="distance").head(5)
    recommended_uris = recommended_songs["uri"].tolist()

    # Retrieve song names (replace stub with Spotify API integration)
    song_names = get_song_names_from_url(recommended_uris)

    # Print playlist
    print("Generated Playlist:")
    for song in song_names:
        print(song)

# Example usage
prompt = "dance songs to play at a Friday night disco party"
generate_playlist(prompt)


# For all users(not working , its printing for some but not all)

# In[37]:


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import openai
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# # Initialize OpenAI client
# client = openai.Client(api_key="your_openai_api_key")

# # Set up Spotify API credentials
# sp = spotipy.Spotify(
#     client_credentials_manager=SpotifyClientCredentials(
#         client_id="your_spotify_client_id", client_secret="your_spotify_client_secret"
#     )
# )

# # Load dataset
# data = pd.read_csv('../../datasets/seven_day_listening_history.csv')

# # Normalize features
# features = [
#     "acousticness", "danceability", "energy", "instrumentalness",
#     "liveness", "valence", "tempo", "loudness", "speechiness"
# ]
# scaler = MinMaxScaler()
# data[features] = scaler.fit_transform(data[features])

# # Function to clean and parse LLM response
# def clean_llm_response(response_message):
#     try:
#         # Remove code block formatting
#         if response_message.startswith("```"):
#             response_message = response_message.split("```")[-2].strip()
#         return eval(response_message)
#     except Exception as e:
#         print(f"Error parsing LLM response: {e}")
#         return None

# # Function to get prompt feature scores
# def get_prompt_features(prompt):
#     try:
#         chat_completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a music expert.",
#                 },
#                 {
#                     "role": "user",
#                     "content": f"Given the playlist prompt: '{prompt}', assign values between 0 and 1 for these features: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, and tempo. Respond with only a list of values in Python list format.",
#                 }
#             ]
#         )
#         response_message = chat_completion.choices[0].message.content.strip()
#         print(f"Response from LLM: {response_message}")  # Debugging step
#         return clean_llm_response(response_message)
#     except Exception as e:
#         print(f"Error fetching prompt features: {e}")
#         return None

# # Function to retrieve song names using Spotify API
# def get_song_names_from_url(song_uris):
#     song_names = []
#     for uri in song_uris:
#         try:
#             track_id = uri.split(":")[-1]  # Extract the track ID from the URI
#             track_info = sp.track(track_id)  # Get track information
#             song_name = track_info['name']  # Extract song name
#             artist_name = track_info['artists'][0]['name']  # Extract artist name
#             song_names.append(f"{song_name} by {artist_name}")
#         except Exception as e:
#             song_names.append(f"Error retrieving song for URI: {uri} ({e})")
#     return song_names

# # Function to filter unique songs
# def filter_unique_songs(data, features):
#     song_vectors = data[features].values
#     normalized_vectors = song_vectors / np.linalg.norm(song_vectors, axis=1, keepdims=True)
#     unique_indices = list({tuple(vec): i for i, vec in enumerate(normalized_vectors)}.values())
#     return data.iloc[unique_indices].reset_index(drop=True)

# # Compute Euclidean distances
# def compute_distance(song, prompt_features):
#     song_features = song[features].values
#     return np.sqrt(np.sum((song_features - np.array(prompt_features)) ** 2))

# # Main logic to generate playlists for all users
# def generate_playlists_for_all_users(prompt):
#     # Group by user_id
#     grouped_data = data.groupby("user_id")
    
#     for user_id, user_data in grouped_data:
#         print(f"\nGenerating playlist for User {user_id}...")
        
#         # Filter unique songs for the user
#         user_data = filter_unique_songs(user_data, features)
        
#         # Fetch features for the prompt
#         prompt_features = get_prompt_features(prompt)
#         if prompt_features is None:
#             print("Failed to fetch prompt features.")
#             continue

#         # Calculate distances
#         user_data["distance"] = user_data.apply(lambda row: compute_distance(row, prompt_features), axis=1)

#         # Get top 5 recommended URIs
#         recommended_songs = user_data.sort_values(by="distance").head(5)
#         recommended_uris = recommended_songs["uri"].tolist()

#         # Retrieve song names using Spotify API
#         song_names = get_song_names_from_url(recommended_uris)

#         # Print playlist
#         print(f"Generated Playlist for User {user_id}:")
#         for song in song_names:
#             print(song)

# # Example usage
# prompt = "dance songs to play at a Friday night disco party"
# generate_playlists_for_all_users(prompt)


# In[ ]:




