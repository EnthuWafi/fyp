# youtube_player.py
import yt_dlp
import vlc
import time
import threading
import os

class YouTubeQueuePlayer:
    def __init__(self):
        print("[INFO] Initializing Bulletproof Audio Engine...")
        self.instance = vlc.Instance('--no-video')
        self.player = self.instance.media_player_new()
        
        # Create a temporary folder to hold the downloaded songs
        if not os.path.exists("temp_music"):
            os.makedirs("temp_music")
            
        # Updated yt-dlp settings to DOWNLOAD the file securely
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_music/%(id)s.%(ext)s', # Save it locally
            'noplaylist': True,
            'quiet': True,
            # This line specifically bypasses the new YouTube SABR block
            'extractor_args': {'youtube': {'player_client': ['android']}} 
        }
        
        self.current_track = None
        self.current_duration = 0
        self.start_time = 0
        
        self.next_track = None
        self.next_file_path = None # We now store the local file path!
        self.is_fetching = False

    def prefetch_song(self, search_query):
        if self.is_fetching or search_query == self.next_track:
            return 
            
        self.is_fetching = True
        self.next_track = search_query
        threading.Thread(target=self._fetch_url_worker, args=(search_query,), daemon=True).start()

    def _fetch_url_worker(self, search_query):
        print(f"[AUDIO] Downloading next track: {search_query}...")
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # MAGIC FIX: download=True
                info = ydl.extract_info(f"ytsearch1:{search_query} audio", download=True)
                if 'entries' in info and len(info['entries']) > 0:
                    track_data = info['entries'][0]
                    
                    # Get the exact local file path where yt-dlp saved the audio
                    self.next_file_path = ydl.prepare_filename(track_data)
                    self.current_duration = track_data.get('duration', 0)
                    
                    print(f"[AUDIO] Ready (Downloaded) -> {search_query}")
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to fetch {search_query}: {e}")
            self.next_track = None
            
        self.is_fetching = False

    def play_next_in_queue(self):
        if self.next_file_path and os.path.exists(self.next_file_path):
            # Tell VLC to play the LOCAL file, completely bypassing the internet
            self.player.set_mrl(self.next_file_path)
            self.player.play()
            
            self.current_track = self.next_track
            self.start_time = time.time()
            
            # Clear the queue
            self.next_track = None
            self.next_file_path = None
            return True
        return False

    def get_time_remaining(self):
        if not self.player.is_playing() and self.start_time > 0:
            return 0
            
        elapsed = time.time() - self.start_time
        remaining = self.current_duration - elapsed
        return max(0, remaining)