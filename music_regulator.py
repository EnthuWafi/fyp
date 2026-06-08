# music_regulator.py
import pandas as pd
import numpy as np
import os
from annoy import AnnoyIndex
from repository import MusicRepository

class IsoPrincipleRegulator:
    def __init__(self, annoy_index_path='music_vectors.ann'):
        print("[INFO] Initializing Adaptive Music Regulation Logic...")
    
        self.f = 2 
        self.annoy_index = AnnoyIndex(self.f, 'euclidean')
        if os.path.exists(annoy_index_path):
            self.annoy_index.load(annoy_index_path)
        else:
            raise FileNotFoundError(f"Missing {annoy_index_path}. Run setup.py first.")

        # Instantiate the Data Repository
        self.db = MusicRepository()
            
        # Adaptive Step
        self.last_a = None
        self.last_v = None

        self.target_v = None
        self.target_a = None
        
        self.bias_multiplier_a = 1.0
        self.bias_multiplier_v = 1.0

        self.active_protocol = "None"
        self.current_track_id = None

    
    def evaluate_feedback(self, current_a=0.0, current_v=0.0):
        """ Checks if the previous song successfully changed the driver's mood. """
        if self.last_a is None or self.last_v is None:
            self.last_a = current_a
            self.last_v = current_v
            return

        # Check Calm Down Protocol effectiveness
        if self.active_protocol == "Calm Down Protocol":
            #arousal
            if (self.last_a - current_a) >= 0.1:
                print(f"[FEEDBACK] Success. Arousal dropped by {self.last_a - current_a:.2f}.")
                self.bias_multiplier_a = 1.0
            else:
                self.bias_multiplier_a = min(4.0, self.bias_multiplier_a * 2.0)
                print(f"[FEEDBACK] Arousal not improving. Doubling bias shift to {self.bias_multiplier_a}x.")

            #valence
            if (self.last_v - current_v) >= 0.1:
                print(f"[FEEDBACK] Success. Valence dropped by {self.last_v - current_v:.2f}.")
                self.bias_multiplier_v = 1.0
            else:
                self.bias_multiplier_v = min(4.0, self.bias_multiplier_v * 2.0)
                print(f"[FEEDBACK] Valence not improving. Doubling bias shift to {self.bias_multiplier_v}x.")

        # Check Ramp Up Protocol effectiveness
        elif self.active_protocol == "Ramp Up Protocol":
            if (current_a - self.last_a) >= 0.1:
                print(f"[FEEDBACK] Success. Arousal increased by {current_a - self.last_a:.2f}.")
                self.bias_multiplier_a = 1.0
            else:
                self.bias_multiplier_a = min(4.0, self.bias_multiplier_a * 2.0)
                print(f"[FEEDBACK] Arousal not improving. Doubling bias shift to {self.bias_multiplier_a}x.")

            #valence
            if (current_v - self.last_v) >= 0.1:
                print(f"[FEEDBACK] Success. Valence dropped by {self.last_v - current_v:.2f}.")
                self.bias_multiplier_v = 1.0
            else:
                self.bias_multiplier_v = min(4.0, self.bias_multiplier_v * 2.0)
                print(f"[FEEDBACK] Valence not improving. Doubling bias shift to {self.bias_multiplier_v}x.")
                
        # If they are in the Safe Zone, reset the bias
        elif self.active_protocol == "Sustain Protocol":
            self.bias_multiplier_a = 1.0
            self.bias_multiplier_v = 1.0

        # Save current state for the NEXT evaluation
        self.last_a = current_a
        self.last_v = current_v

    def select_track(self, current_valence, current_arousal):
        """ The Iso Principle """
        
        self.target_v = current_valence
        self.target_a = current_arousal

        # QUADRANT PROTOCOL LOGIC
        if current_valence >= 0:
            self.active_protocol = "Sustain Protocol"
            
        elif current_valence < 0 and current_arousal >= 0:
            self.active_protocol = "Calm Down Protocol"
            self.target_v = min(1.0, current_valence + (0.2 * self.bias_multiplier_v))
            self.target_a = max(-1.0, current_arousal - (0.3 * self.bias_multiplier_a))
            
        elif current_valence < 0 and current_arousal < 0:
            self.active_protocol = "Ramp Up Protocol"
            self.target_v = min(1.0, current_valence + (0.2 * self.bias_multiplier_v))
            self.target_a = min(1.0, current_arousal + (0.2 * self.bias_multiplier_a))

        # Annoy logic
        nearest_neighbors = self.annoy_index.get_nns_by_vector([target_v, target_a], 50)
        
        result = self.db.get_best_candidate(
            annoy_ids=nearest_neighbors, 
            active_protocol=self.active_protocol,
            bias_a=self.bias_multiplier_a,
            bias_v=self.bias_multiplier_v
        )

        if result is None:
            print(f"[FEEDBACK] Could not find any good candidate, playing fallback track.")
            result = self.db.get_fallback_track(nearest_neighbors[0])
        
        # best_match_idx = None
        # for idx in nearest_neighbors:
        #     if idx not in self.recently_played:
        #         best_match_idx = idx
        #         break
                
        # # Fallback if somehow all 10 were played recently
        # if best_match_idx is None:
        #     best_match_idx = nearest_neighbors[0] 

        # # Update History
        # self.recently_played.append(best_match_idx)
        # if len(self.recently_played) > 20: # Keep memory clean
        #     self.recently_played.pop(0)

        # Fetch Track Data
        self.current_track_id = result[0]
        # best_match = self.df.iloc[best_match_idx]

        track_string = f"{result[1]} - {result[2]}"
        
        return self.active_protocol, track_string
    
    def close(self):
        """Safely close repository connections"""
        self.db.close()