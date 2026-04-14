# music_regulator.py
import pandas as pd
import numpy as np
import os
from annoy import AnnoyIndex

class IsoPrincipleRegulator:
    def __init__(self, dataset_path='muse_v3.csv'):
        print("[INFO] Initializing Music Regulation Logic...")
    
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.dropna(subset=['valence_tags', 'arousal_tags'])
        self.df = self.df.reset_index(drop=True)

        # 1. Build the Annoy Index (if it doesnt already exist)
        self.f = 2 # Dimensions (Valence, Arousal)
        self.annoy_index = AnnoyIndex(self.f, 'euclidean')

        index_filename = 'muse.ann'
        if os.path.exists(index_filename):
            print("[INFO] Loading cached Annoy Index...")
            self.annoy_index.load(index_filename)
        else:
            print("[INFO] Building Annoy Vector Index (This will only happen once)...")
            for i, row in self.df.iterrows():
                norm_v = (row['valence_tags'] - 5.0) / 4.0
                norm_a = (row['arousal_tags'] - 5.0) / 4.0
                self.annoy_index.add_item(i, [norm_v, norm_a])
            self.annoy_index.build(10)
            self.annoy_index.save(index_filename)
            print("[INFO] Annoy Index saved to disk.")
            
        # Recently played songs
        self.recently_played = []

        # --- NEW: Adaptive Step Variables ---
        self.last_a = None
        self.bias_multiplier = 1.0
        self.active_protocol = "None"

    def evaluate_feedback(self, current_a):
        """ Checks if the previous song successfully changed the driver's mood. """
        if self.last_a is None:
            self.last_a = current_a
            return

        # Check Calm Down Protocol effectiveness
        if self.active_protocol == "[Calm Down Protocol]":
            if (self.last_a - current_a) >= 0.1:
                print(f"[FEEDBACK] Success. Arousal dropped by {self.last_a - current_a:.2f}.")
                self.bias_multiplier = 1.0
            else:
                self.bias_multiplier = min(4.0, self.bias_multiplier * 2.0)
                print(f"[FEEDBACK] User resisting! Doubling bias shift to {self.bias_multiplier}x.")

        # Check Ramp Up Protocol effectiveness
        elif self.active_protocol == "[Ramp Up Protocol]":
            if (current_a - self.last_a) >= 0.1:
                print(f"[FEEDBACK] Success. Arousal increased by {current_a - self.last_a:.2f}.")
                self.bias_multiplier = 1.0
            else:
                self.bias_multiplier = min(4.0, self.bias_multiplier * 2.0)
                print(f"[FEEDBACK] User resisting! Doubling bias shift to {self.bias_multiplier}x.")
                
        # If they are in the Safe Zone, reset the bias
        elif self.active_protocol == "[Sustain Protocol]":
            self.bias_multiplier = 1.0

        # Save current state for the NEXT evaluation
        self.last_a = current_a

    def select_track(self, current_valence, current_arousal):
        """ The Iso Principle """
        
        target_v = current_valence
        target_a = current_arousal

        # --- QUADRANT PROTOCOL LOGIC ---
        if current_valence >= 0:
            self.active_protocol = "[Sustain Protocol]"
            # Target remains identical to current
            
        elif current_valence < 0 and current_arousal >= 0:
            self.active_protocol = "[Calm Down Protocol]"
            # Multiply the shift by our adaptive bias!
            target_v = min(1.0, current_valence + (0.2 * self.bias_multiplier))
            target_a = max(-1.0, current_arousal - (0.3 * self.bias_multiplier))
            
        elif current_valence < 0 and current_arousal < 0:
            self.active_protocol = "[Ramp Up Protocol]"
            target_v = min(1.0, current_valence + (0.2 * self.bias_multiplier))
            target_a = min(1.0, current_arousal + (0.2 * self.bias_multiplier))

        # --- ANNOY RETRIEVAL & HISTORY FILTER ---
        # Retrieve the top 10 closest matches to our target vector
        nearest_neighbors = self.annoy_index.get_nns_by_vector([target_v, target_a], 10)
        
        best_match_idx = None
        for idx in nearest_neighbors:
            if idx not in self.recently_played:
                best_match_idx = idx
                break
                
        # Fallback if somehow all 10 were played recently
        if best_match_idx is None:
            best_match_idx = nearest_neighbors[0] 

        # Update History
        self.recently_played.append(best_match_idx)
        if len(self.recently_played) > 20: # Keep memory clean
            self.recently_played.pop(0)

        # Fetch Track Data
        best_match = self.df.iloc[best_match_idx]
        track_string = f"{best_match['track']} - {best_match['artist']}"
        
        return self.active_protocol, track_string