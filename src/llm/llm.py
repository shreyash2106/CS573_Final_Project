# Normalize features
features = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "valence", "tempo", "loudness", "speechiness"
]
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Function to get prompt feature scores
def get_prompt_features(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a music expert."},
            {"role": "user", "content": f"Given the playlist prompt: '{prompt}', assign values between 0 and 1 for these features: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, and tempo. Respond in JSON format."}
        ],
        max_tokens=100
    )
    feature_scores = eval(response["choices"][0]["message"]["content"].strip())
    return feature_scores

# Example prompt
prompt = "dance songs to play at a Friday night disco party"
prompt_features = get_prompt_features(prompt)

# Filter user data
user_data = data[data["user_id"] == 1]

# Compute Euclidean distances
def compute_distance(song, prompt_features):
    song_features = song[features].values
    prompt_vector = np.array([prompt_features[f] for f in features])
    return np.sqrt(np.sum((song_features - prompt_vector) ** 2))

user_data["distance"] = user_data.apply(lambda row: compute_distance(row, prompt_features), axis=1)

# Get top 5 recommendations
recommended_songs = user_data.sort_values(by="distance").head(5)
recommended_uris = recommended_songs["uri"].tolist()

# Get song names
song_names = get_song_names_from_url(recommended_uris)

# Print the playlist
print("Generated Playlist:")
for song in song_names:
    print(song)