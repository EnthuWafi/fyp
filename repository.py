# repository.py
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class MusicRepository:
    def __init__(self, dbname=os.getenv("DATABASE_NAME"), user=os.getenv("DATABASE_USER"), 
    password=os.getenv("DATABASE_PASSWORD"), host=os.getenv("DATABASE_HOST"), port=os.getenv("DATABASE_PORT")):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.cursor = self.conn.cursor()

        # weights
        self.weight_acousticness = 1.0
        self.weight_loudness = 1.0
        self.weight_danceability = 1.0

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
            VALUES (%s, %s, %s, %s);
        """
        self.cursor.execute(query, (track_id, listened_seconds, total_seconds, explicit_skip))
        self.conn.commit()

    def get_best_candidate(self, annoy_ids, active_protocol, bias_a, bias_v):
        """
        Filters the Annoy IDs against recent history and applies the sorting rule
        """

        w_acoustic, w_loud, w_dance = 0.0, 0.0, 0.0

        if active_protocol == "Calm Down Protocol":
            # Resistance scales the demand for high acousticness and lower loudness
            w_acoustic = 1.0 * bias_a
            w_loud = -1.0 * bias_v 
            w_dance = 1.0 * bias_v

        elif active_protocol == "Ramp Up Protocol":
            # Resistance forces lower acousticness and higher energy adjustments
            w_acoustic = -1.0 * bias_a
            w_loud = -1.0 * bias_v 
            w_dance = 1.0 * bias_v

        query = query = """
            SELECT id, title, artist,
                   ((%s * acousticness) + (%s * loudness) + (%s * danceability)) AS suitability_score
            FROM tracks 
            WHERE annoy_id IN %s
            AND id NOT IN (
                SELECT track_id FROM playback_history 
                WHERE played_at > NOW() - INTERVAL '1 hour'
            )
            ORDER BY 
                CASE 
                    WHEN %s IN ('[Calm Down Protocol]', '[Ramp Up Protocol]') THEN 
                        ((%s * acousticness) + (%s * loudness) + (%s * danceability))
                    ELSE RANDOM() 
                END DESC
            LIMIT 1;
        """
        params = (
            w_acoustic, w_loud, w_dance, 
            tuple(annoy_ids), 
            active_protocol, 
            w_acoustic, w_loud, w_dance
        )

        self.cursor.execute(query, params)
        return self.cursor.fetchone()

    def get_fallback_track(self, backup_annoy_id):
        """Emergency fallback query if the history filter clears the candidate pool."""
        query = "SELECT id, title, artist FROM tracks WHERE annoy_id = %s;"
        self.cursor.execute(query, (backup_annoy_id,))
        return self.cursor.fetchone()

    def close(self):
        self.cursor.close()
        self.conn.close()