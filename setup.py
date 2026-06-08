# setup.py
import kagglehub
import pandas as pd
from annoy import AnnoyIndex
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv

load_dotenv()
# Download Dataset
print("Downloading 1M Spotify dataset...")
path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks", 
output_dir="./dataset")

# We need the exact CSV path.
csv_path = os.path.join(path, "spotify_data.csv")

# Load Data into Pandas
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path, usecols=[
    'track_name', 'artist_name', 'valence', 'energy', 
    'danceability', 'loudness', 'acousticness'])

# Drop any rows with missing values
df = df.dropna()

# Setup Annoy
print("Initializing Annoy...")
f = 2
t = AnnoyIndex(f, 'euclidean')

# Setup Postgres Connection
conn = psycopg2.connect(
    dbname=os.getenv("DATABASE_NAME"), 
    user=os.getenv("DATABASE_USER"), 
    password=os.getenv("DATABASE_PASSWORD"), 
    host=os.getenv("DATABASE_HOST"),
    port=os.getenv("DATABASE_PORT")
)
cursor = conn.cursor()

# Create the table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        id SERIAL PRIMARY KEY,
        annoy_id INTEGER UNIQUE NOT NULL,
        title TEXT,
        artist TEXT,
        valence FLOAT,
        arousal FLOAT,
        danceability FLOAT,
        loudness FLOAT,
        acousticness FLOAT
    );
""")
cursor.execute("TRUNCATE TABLE tracks RESTART IDENTITY CASCADE;")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS playback_history (
        id SERIAL PRIMARY KEY,
        track_id INTEGER REFERENCES tracks(id),
        played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        duration_listened_seconds INT NOT NULL,
        total_duration_seconds INT NOT NULL,
        explicit_skip BOOLEAN DEFAULT FALSE
    );
""")
cursor.execute("TRUNCATE TABLE playback_history RESTART IDENTITY CASCADE;")

# Process Data
print("Building Annoy Index and preparing database records. This may take minutes...")
db_records = []

# Need to normalize the Valence Arousal since the model outputs -1 to 1 range, while
# Spotify VA is 0 to 1 range. So need to normalize it
def normalize(num):
    return (num * 2.0) - 1.0
    
# itertuples() is much faster than iterrows() for large datasets supposedly
for i, row in enumerate(df.itertuples()):
    annoy_id = i
    valence = normalize(row.valence) 
    arousal = normalize(row.energy)
    
    # add vector to Annoy
    t.add_item(annoy_id, [valence, arousal])

    # loudness scaling
    db_min, db_max = -60.0, 0.0
    loudness_clean = (row.loudness - db_min) / (db_max - db_min)
    loudness_scaled = max(0.0, min(1.0, loudness_clean))
    
    # Prepare tuple for Postgres bulk insert
    db_records.append((
        annoy_id, row.track_name, row.artist_name, 
        valence, arousal, row.danceability, loudness_scaled, row.acousticness
    ))

# Save Annoy File
print("Building Annoy trees...")
t.build(10)
t.save('music_vectors.ann')
print("music_vectors.ann saved successfully!")

# Bulk Insert into Postgres
length = len(df)
print(f"Inserting {length} million rows into PostgreSQL...")
insert_query = """
    INSERT INTO tracks (annoy_id, title, artist, valence, arousal, danceability, loudness, acousticness)
    VALUES %s
"""
# execute_values is the secret to making this take seconds instead of hours
execute_values(cursor, insert_query, db_records)
conn.commit()

cursor.close()
conn.close()
print("Database populated successfully!")