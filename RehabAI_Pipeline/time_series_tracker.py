import numpy as np
from collections import deque

class ExerciseTracker:
    def __init__(self, exercise_type="seated_knee_extension", window_size=30):
        self.exercise_type = exercise_type
        self.angle_window = deque(maxlen=window_size)
        
        self.state = "DOWN"
        self.rep_count = 0
        self.current_score = 0
        
        # Immediate clear instructions
        if self.exercise_type == "seated_knee_extension":
            self.feedback = "Sit in chair, camera side. Action: Knee Extension."
        elif self.exercise_type == "shoulder_abduction":
            self.feedback = "Stand facing camera. Action: Shoulder Abduction."
        elif self.exercise_type == "standing_march":
            self.feedback = "Stand sideways, hold chair. Action: Standing March."
        else:
            self.feedback = "Ready. Begin!"

    def process_frame_angles(self, current_angle):
        self.angle_window.append(current_angle)
        
        if len(self.angle_window) < 5:
            return

        if self.exercise_type == "seated_knee_extension":
            self._evaluate_knee_extension()
        elif self.exercise_type == "shoulder_abduction":
            self._evaluate_shoulder_abduction()
        elif self.exercise_type == "standing_march":
            self._evaluate_standing_march()

    def _evaluate_knee_extension(self):
        # Angle goes UP when extending leg (90 -> 160+)
        current_angle = self.angle_window[-1]
        
        if self.state == "DOWN":
            if current_angle > 140:
                self.state = "UP"
                self.feedback = "Good extension! Now lower leg."
            elif current_angle > 100:
                self.feedback = "Keep straightening leg..."
                
        elif self.state == "UP":
            if current_angle < 110:
                self.state = "DOWN"
                self.rep_count += 1
                self.current_score += 100
                self.feedback = f"Rep {self.rep_count}! Straighten leg again."

    def _evaluate_shoulder_abduction(self):
        # Angle goes UP when raising arm (10 -> 140+)
        current_angle = self.angle_window[-1]

        if self.state == "DOWN":
            if current_angle > 130:
                self.state = "UP"
                self.feedback = "Good raise! Lower your arm."
            elif current_angle > 40:
                self.feedback = "Keep raising arm..."
                
        elif self.state == "UP":
            if current_angle < 40:
                self.state = "DOWN"
                self.rep_count += 1
                self.current_score += 100
                self.feedback = f"Rep {self.rep_count}! Raise arm again."

    def _evaluate_standing_march(self):
        # Angle goes DOWN when lifting knee (180 -> 100-)
        current_angle = self.angle_window[-1]

        if self.state == "DOWN":
            if current_angle < 120:
                self.state = "UP"
                self.feedback = "Good march! Lower your knee."
            elif current_angle < 150:
                self.feedback = "Keep lifting knee higher..."
                
        elif self.state == "UP":
            if current_angle > 150:
                self.state = "DOWN"
                self.rep_count += 1
                self.current_score += 100
                self.feedback = f"Rep {self.rep_count}! March knee again."
        
    def get_stats(self):
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback
        }
