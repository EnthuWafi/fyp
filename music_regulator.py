# music_regulator.py
import numpy as np
import os
from annoy import AnnoyIndex
from repository import MusicRepository

class IsoPrincipleRegulator:
    def __init__(self, db_path, annoy_index_path):
        print("[INFO] Initializing Adaptive Music Regulation Logic...")
    
        self.f = 2 
        self.annoy_index = AnnoyIndex(self.f, 'euclidean')
        if os.path.exists(annoy_index_path):
            self.annoy_index.load(str(annoy_index_path))
        else:
            raise FileNotFoundError(f"Missing {annoy_index_path}. Run setup.py first.")

        self.db = MusicRepository(db_path)
            
        # Adaptive Step
        self.last_a = None
        self.last_v = None

        self.target_v = None
        self.target_a = None
        
        self.bias_multiplier_a = 1.0
        self.bias_multiplier_v = 1.0

        self.active_protocol = "None"

        self.current_track_id = None

    
    def evaluate_feedback(self, current_v=0.0, current_a=0.0):
        """ Checks if the previous song successfully changed the driver's mood. """
        if self.last_a is None or self.last_v is None:
            self.last_a = current_a
            self.last_v = current_v
            return

        # Check Calm Down Protocol effectiveness
        min_improvement = 0.1
        if self.active_protocol == "Calm Down Protocol" or self.active_protocol == "Emergency Calm Protocol":
            #arousal
            if (self.last_a - current_a) >= min_improvement:
                print(f"[FEEDBACK] Success. Arousal dropped by {self.last_a - current_a:.2f}.")
                self.bias_multiplier_a = 1.0
            else:
                self.bias_multiplier_a = min(4.0, self.bias_multiplier_a * 2.0)
                print(f"[FEEDBACK] Arousal not improving. Doubling bias shift to {self.bias_multiplier_a}x.")

            #valence
            if (current_v - self.last_v) >= min_improvement:
                print(f"[FEEDBACK] Success. Valence increased by {current_v - self.last_v:.2f}.")
                self.bias_multiplier_v = 1.0
            else:
                self.bias_multiplier_v = min(4.0, self.bias_multiplier_v * 2.0)
                print(f"[FEEDBACK] Valence not improving. Doubling bias shift to {self.bias_multiplier_v}x.")

        # Check Ramp Up Protocol effectiveness
        elif self.active_protocol == "Ramp Up Protocol":
            if (current_a - self.last_a) >= min_improvement:
                print(f"[FEEDBACK] Success. Arousal increased by {current_a - self.last_a:.2f}.")
                self.bias_multiplier_a = 1.0
            else:
                self.bias_multiplier_a = min(4.0, self.bias_multiplier_a * 2.0)
                print(f"[FEEDBACK] Arousal not improving. Doubling bias shift to {self.bias_multiplier_a}x.")

            #valence
            if (current_v - self.last_v) >= min_improvement:
                print(f"[FEEDBACK] Success. Valence increased by {self.last_v - current_v:.2f}.")
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

    def select_track(self, current_valence, current_arousal, force_calm=False):
        """ The Iso Principle """
        
        self.target_v = current_valence
        self.target_a = current_arousal


        # QUADRANT PROTOCOL LOGIC

        if current_valence >= 0:
            new_protocol = "Sustain Protocol"
        elif current_valence < 0 and current_arousal >= 0:
            new_protocol = "Emergency Calm Protocol" if force_calm else "Calm Down Protocol"
        elif current_valence < 0 and current_arousal < 0:
            new_protocol = "Ramp Up Protocol"

        if new_protocol != self.active_protocol:
            if self.active_protocol == "Emergency Calm Protocol" and new_protocol == "Calm Down Protocol":
                print("[SYSTEM] No protocol shift needed.") # equivalent
            elif new_protocol == "Emergency Calm Protocol":
                # starting biases
                self.bias_multiplier_a = 1.5
                self.bias_multiplier_v = 1.5
            else:
                print(f"[SYSTEM] Protocol shifting from {self.active_protocol} to {new_protocol}. Resetting adaptive biases.")
                self.bias_multiplier_a = 1.0
                self.bias_multiplier_v = 1.0
        
        self.active_protocol = new_protocol

        if self.active_protocol == "Calm Down Protocol" or self.active_protocol == "Emergency Calm Protocol":
            self.target_v = min(1.0, current_valence + (0.2 * self.bias_multiplier_v))
            self.target_a = max(-1.0, current_arousal - (0.2 * self.bias_multiplier_a))
            
        elif self.active_protocol == "Ramp Up Protocol":
            self.target_v = min(1.0, current_valence + (0.2 * self.bias_multiplier_v))
            self.target_a = min(1.0, current_arousal + (0.2 * self.bias_multiplier_a))

        # Annoy logic
        nearest_neighbors = self.annoy_index.get_nns_by_vector([self.target_v, self.target_a], 50)
        
        result = self.db.get_best_candidate(
            annoy_ids=nearest_neighbors, 
            active_protocol=self.active_protocol,
            bias_a=self.bias_multiplier_a,
            bias_v=self.bias_multiplier_v
        )

        if result is None:
            print(f"[FEEDBACK] Could not find any good candidate, playing fallback track.")
            result = self.db.get_fallback_track(nearest_neighbors[0])
        
        # Fetch Track Data
        self.current_track_id = result[0]

        track_string = f"{result[1]} - {result[2]}"
        va_data = [result[3], result[4]] #maybe return the exact va data too

        return self.current_track_id, self.active_protocol, track_string, va_data
    
    
    def close(self):
        """Safely close repository connections"""
        self.db.close()