# Simulated User Dataset Schema

This dataset contains a simulated set of users, each assigned to one of seven user groups based on different musical preferences. 
Each user has approximately 1,000 songs sampled from the original dataset, reflecting their group's preferences for certain musical attributes and emotion labels.

## Columns Description

| Column Name           | Data Type | Description                                                                                         |
|-----------------------|-----------|-----------------------------------------------------------------------------------------------------|
| `user_id`             | int       | A unique identifier for each user (ranging from 1 to the number of users).                         |
| `group_no`            | int       | The group number to which the user belongs (1-7), indicating different musical preferences.         |
| `song_id`             | int       | A unique identifier for each song in the dataset.                                                   |
| `duration (ms)`       | float     | The duration of the song in milliseconds.                                                           |
| `danceability`        | float     | A measure of how suitable the track is for dancing (0.0 to 1.0).                                    |
| `energy`              | float     | A measure of intensity and activity (0.0 to 1.0).                                                   |
| `loudness`            | float     | The overall loudness of the track in decibels.                                                      |
| `speechiness`         | float     | A measure of the presence of spoken words in the track (0.0 to 1.0).                                |
| `acousticness`        | float     | A measure of the track’s acoustic qualities (0.0 to 1.0).                                           |
| `instrumentalness`    | float     | A measure of the likelihood that the track contains no vocals (0.0 to 1.0).                         |
| `liveness`            | float     | A measure of the presence of a live audience in the recording (0.0 to 1.0).                         |
| `valence`             | float     | A measure of the musical positiveness conveyed by the track (0.0 to 1.0).                           |
| `tempo`               | float     | The tempo of the song in beats per minute (BPM).                                                    |
| `spotify_uri`         | string    | The Spotify URI for the song, allowing it to be accessed directly on Spotify.                       |
