import sqlite3
import pandas as pd

def get_telemetry(conn):
    """Extracts runtime logs from SQLite and compile a standardized evaluation CSV."""

    query = """
        SELECT 
            p.id AS session_log_id,
            p.played_at,
            t.title,
            t.artist,
            t.valence AS track_valence,
            t.arousal AS track_arousal,
            p.duration_listened_seconds,
            p.total_duration_seconds,
            ROUND((CAST(p.duration_listened_seconds AS REAL) / p.total_duration_seconds) * 100, 2) AS listen_percentage,
            p.explicit_skip
        FROM playback_history p
        JOIN tracks t ON p.track_id = t.id
        ORDER BY p.played_at ASC;
    """
 
    df = pd.read_sql_query(query, conn)

    return df
