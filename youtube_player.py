# youtube_player.py
import yt_dlp
import vlc
import time
import threading
import os
import json

class YouTubeQueuePlayer:
    def __init__(self):
        print("[INFO] Initializing Audio Engine...")
        self.instance = vlc.Instance('--no-video')
        self.player = self.instance.media_player_new()
        
        # Create a temporary folder to hold the downloaded songs
        if not os.path.exists("temp_music"):
            os.makedirs("temp_music")
            
        # Updated yt-dlp settings to DOWNLOAD the file securely
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_music/%(id)s.%(ext)s',
            'noplaylist': True,
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['android']}} 
        }
        
        self.current_track = None
        self.current_duration = 0
        self.start_time = 0
        
        self.next_track = None
        # self.next_file_path = None
        self.next_stream_url = None
        self.is_fetching = False

        # self.local_cache = {}       # Maps "Song - Artist" to "temp_music/file.m4a"
        # self.local_durations = {}   # Maps "Song - Artist" to length in seconds

        # self.cache_file = "temp_music/cache_data.json"
        # if os.path.exists(self.cache_file):
        #     try:
        #         with open(self.cache_file, 'r') as f:
        #             saved_data = json.load(f)
        #             # Verify files haven't been deleted manually before loading them
        #             for query, data in saved_data.items():
        #                 if os.path.exists(data['path']):
        #                     self.local_cache[query] = data['path']
        #                     self.local_durations[query] = data['duration']
        #         print(f"[INFO] Successfully loaded {len(self.local_cache)} songs from local cache.")
        #     except Exception as e:
        #         print(f"[WARNING] Could not load cache file: {e}")

    def prefetch_song(self, search_query):
        if self.is_fetching or search_query == self.next_track:
            return 
            
        self.is_fetching = True
        self.next_track = search_query
        threading.Thread(target=self._fetch_url_worker, args=(search_query,), daemon=True).start()

    def _fetch_url_worker(self, search_query):

        print(f"[AUDIO] Fetching stream URL for: {search_query}...")
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # MAGIC FIX: download=False. We ONLY want the raw server URL.
                info = ydl.extract_info(f"ytsearch1:{search_query} audio", download=False)
                
                if 'entries' in info and len(info['entries']) > 0:
                    track_data = info['entries'][0]
                    
                    # Grab the direct audio stream URL
                    self.next_stream_url = track_data['url'] 
                    self.current_duration = track_data.get('duration', 0)
                    
                    print(f"[AUDIO] Stream Ready -> {search_query}")
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to fetch {search_query}: {e}")
            self.next_track = None
            
        self.is_fetching = False

        # print(f"[AUDIO] Downloading next track: {search_query}...")
        # # --- 1. CHECK THE CACHE FIRST ---
        # if search_query in self.local_cache and os.path.exists(self.local_cache[search_query]):
        #     print(f"[AUDIO] CACHE HIT! Instantly loading from disk: {search_query}")
        #     self.next_file_path = self.local_cache[search_query]
        #     self.current_duration = self.local_durations[search_query]
        #     self.is_fetching = False
        #     return

        # # --- 2. IF NOT IN CACHE, DOWNLOAD IT ---
        # print(f"[AUDIO] Downloading next track: {search_query}...")
        # try:
        #     with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
        #         # MAGIC FIX: download=True
        #         info = ydl.extract_info(f"ytsearch1:{search_query} audio", download=True)
        #         if 'entries' in info and len(info['entries']) > 0:
        #             track_data = info['entries'][0]
                    
        #             file_path = ydl.prepare_filename(track_data)
        #             duration = track_data.get('duration', 0)
                    
        #             self.next_file_path = file_path
        #             self.current_duration = duration
                    
        #             self.local_cache[search_query] = file_path
        #             self.local_durations[search_query] = duration

        #             # --- SAVE TO PERSISTENT JSON FILE ---
        #             cache_export = {}
        #             for q in self.local_cache:
        #                 cache_export[q] = {
        #                     'path': self.local_cache[q],
        #                     'duration': self.local_durations[q]
        #                 }
        #             with open(self.cache_file, 'w') as f:
        #                 json.dump(cache_export, f)
                    
        #             print(f"[AUDIO] Ready (Downloaded) -> {search_query}")
        # except Exception as e:
        #     print(f"[AUDIO ERROR] Failed to fetch {search_query}: {e}")
        #     self.next_track = None
            
        # self.is_fetching = False

    def play_next_in_queue(self):
        # if self.next_file_path and os.path.exists(self.next_file_path):
        #     # Tell VLC to play the LOCAL file, completely bypassing the internet
        #     self.player.set_mrl(self.next_file_path)
        #     self.player.play()
            
        #     self.current_track = self.next_track
        #     self.start_time = time.time()
            
        #     # Clear the queue
        #     self.next_track = None
        #     self.next_file_path = None
        #     return True
        # return False
        if self.next_stream_url:
            # Tell VLC to stream directly from the raw YouTube audio URL
            self.player.set_mrl(self.next_stream_url)
            self.player.play()
            
            self.current_track = self.next_track
            self.start_time = time.time()
            
            # Clear the queue
            self.next_track = None
            self.next_stream_url = None
            return True
        return False

    def get_time_remaining(self):
        if not self.player.is_playing() and self.start_time > 0:
            return 0
            
        elapsed = time.time() - self.start_time
        remaining = self.current_duration - elapsed
        return max(0, remaining)