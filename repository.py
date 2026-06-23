# repository.py
# import psycopg2
import sqlite3
import pandas as pd
import os
# from dotenv import load_dotenv

# load_dotenv()

class MusicRepository:
    def __init__(self, db_path="music_system.db"):
        # self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # coefficient weights
        # Valence (β): Predicted by danceability (0.567) and loudness (-0.420).
        # Arousal (β): Predicted by danceability (0.559) and acousticness (-0.508).
        # negative coefficient means that they have inverse relationship

        self.danceability_valence = 0.567
        self.loudness_valence = -0.420
        self.danceability_arousal = 0.559
        self.acousticness_arousal = -0.508

    def log_playback(self, track_id, listened_seconds, total_seconds, explicit_skip):
        """
        Safely stores the exact playback metrics recorded by the media player.
        """
        query = """
            INSERT INTO playback_history (
                track_id, 
                duration_listened_seconds, 
                total_duration_seconds, 
                explicit_skip
            )
            VALUES (?, ?, ?, ?);
        """
        self.cursor.execute(query, (track_id, listened_seconds, total_seconds, explicit_skip))
        self.conn.commit()

    def get_best_candidate(self, annoy_ids, active_protocol, bias_a, bias_v):
        """
        Filters the Annoy IDs against recent history and applies the sorting rule
        """

        # w_acoustic, w_loud, w_dance = 0.0, 0.0, 0.0
        m_v, m_a = 0.0, 0.0

        # if active_protocol == "Calm Down Protocol":
        #     # High acoustic (lower arousal) and lower loudness (increase valence)
        #     w_acoustic = 1.0 * bias_a
        #     w_loud = -1.0 * bias_v 
        #     w_dance = 1.0 * bias_v

        # elif active_protocol == "Ramp Up Protocol":
        #     # Lower acoustic (higher arousal) and lower loudness (increase valence)
        #     w_acoustic = -1.0 * bias_a
        #     w_loud = -1.0 * bias_v 
        #     w_dance = 1.0 * bias_v
        
        m_v = 1.0 * bias_v
        if active_protocol == "Emergency Calm Protocol":
            m_a = -1.5 * bias_a  
        if active_protocol == "Calm Down Protocol":
            m_a = -1.0 * bias_a
        elif active_protocol == "Ramp Up Protocol":
            m_a = 1.0 * bias_a

        placeholders = ",".join(["?"] * len(annoy_ids))

        query = query = f"""
            SELECT id, title, artist, valence, arousal
            FROM tracks 
            WHERE annoy_id IN ({placeholders})
            AND id NOT IN (
                SELECT track_id FROM playback_history 
                WHERE played_at > datetime('now', '-1 hour')
            )
            ORDER BY 
                (? * ((? * danceability) + (? * loudness))) +
                (? * ((? * danceability) + (? * acousticness)))
            LIMIT 1;
        """
        params = tuple(annoy_ids) + (
            m_v, self.danceability_valence, self.loudness_valence,
            m_a, self.danceability_arousal, self.acousticness_arousal
        )

        self.cursor.execute(query, params)

        return self.cursor.fetchone()

    def get_fallback_track(self, backup_annoy_id):
        """Emergency fallback query if the history filter clears the candidate pool."""
        query = "SELECT id, title, artist, valence, arousal FROM tracks WHERE annoy_id = ?;"
        self.cursor.execute(query, (backup_annoy_id,))
        return self.cursor.fetchone()


    def close(self):
        self.cursor.close()
        self.conn.close()