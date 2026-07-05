import csv
import os
import sqlite3


def get_telemetry_cursor(conn):
    """Executes the query and returns the cursor along with column headers."""

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

    cursor = conn.cursor()
    cursor.execute(query)

    # Extract column headers from the cursor description
    headers = [description[0] for description in cursor.description]

    return cursor, headers