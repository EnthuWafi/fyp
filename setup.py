# setup.py
import csv
import os
import sqlite3
import sys
from annoy import AnnoyIndex
import kagglehub
from pathlib import Path

def setup_database(db_path, ann_path):
    # Download Dataset
    if hasattr(sys, "frozen") or "NUITKA_PACKAGE_HOME" in os.environ:
        base_dir = Path(sys.argv[0]).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset"
    csv_path = base_dir / "dataset" / "spotify_data.csv"

    # Check if the target CSV file already exists
    if not os.path.exists(csv_path):
        print("Downloading 1M Spotify dataset...")
        path = kagglehub.dataset_download(
            "amitanshjoshi/spotify-1million-tracks", output_dir=str(dataset_path)
        )

    else:
        print("[INFO] Spotify dataset found. Skipping download.")

    # Setup Annoy
    print("Initializing Annoy...")
    f = 2
    t = AnnoyIndex(f, "euclidean")

    # Setup SQLite Connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Create the tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annoy_id INTEGER UNIQUE NOT NULL,
            title TEXT,
            artist TEXT,
            valence REAL,
            arousal REAL,
            danceability REAL,
            loudness REAL,
            acousticness REAL
        );
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_annoy_id ON tracks(annoy_id);"
    )
    cursor.execute("DELETE FROM tracks;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS playback_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
            played_at TEXT DEFAULT CURRENT_TIMESTAMP,
            duration_listened_seconds INTEGER NOT NULL,
            total_duration_seconds INTEGER NOT NULL,
            explicit_skip INTEGER DEFAULT 0
        );
    """)
    cursor.execute("DELETE FROM playback_history;")

    # Helper function for normalization
    def normalize(num):
        return (num * 2.0) - 1.0

    print(
        "Processing CSV, building Annoy Index, and preparing database records..."
    )
    db_records = []
    annoy_id = 0

    # Open and stream the CSV line-by-line using the built-in csv module
    with open(csv_path, mode="r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)

        for row in reader:
            # Drop rows with missing values dynamically
            required_fields = [
                row["track_name"],
                row["artist_name"],
                row["valence"],
                row["energy"],
                row["danceability"],
                row["loudness"],
                row["acousticness"],
            ]
            if any(val is None or val == "" for val in required_fields):
                continue

            try:
                raw_valence = float(row["valence"])
                raw_energy = float(row["energy"])
                danceability = float(row["danceability"])
                loudness = float(row["loudness"])
                acousticness = float(row["acousticness"])
            except ValueError:
                # Skip row if type conversion fails due to corrupted data
                continue

            # Normalize values
            valence = normalize(raw_valence)
            arousal = normalize(raw_energy)

            # Add vector to Annoy
            t.add_item(annoy_id, [valence, arousal])

            # Loudness scaling
            db_min, db_max = -60.0, 0.0
            loudness_clean = (loudness - db_min) / (db_max - db_min)
            loudness_scaled = max(0.0, min(1.0, loudness_clean))

            # Append to database records
            db_records.append(
                (
                    annoy_id,
                    row["track_name"],
                    row["artist_name"],
                    valence,
                    arousal,
                    danceability,
                    loudness_scaled,
                    acousticness,
                )
            )

            annoy_id += 1

    # Save Annoy File
    print("Building Annoy trees...")
    t.build(10)
    t.save(str(ann_path))
    print("music_vectors.ann saved successfully!")

    # Bulk Insert into SQLite
    print(f"Inserting {len(db_records)} rows into SQLite...")
    insert_query = """
        INSERT INTO tracks (annoy_id, title, artist, valence, arousal, danceability, loudness, acousticness)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_query, db_records)
    conn.commit()

    cursor.close()
    conn.close()
    print("Database populated successfully!")


if __name__ == "__main__":
    setup_database()