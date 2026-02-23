import numpy as np
from collections import deque

class ExerciseTracker:
    def __init__(self, exercise_type="seated_knee_extension", window_size=30):
        self.exercise_type = exercise_type
        self.angle_window = deque(maxlen=window_size)
        
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = 0
        
        if self.exercise_type == "seated_knee_extension":
            self.feedback = "Face right side. Straighten your leg."
        elif self.exercise_type == "shoulder_abduction":
            self.feedback = "Face front. Raise arm to the side."
        elif self.exercise_type == "standing_march":
            self.feedback = "Face right side. March your knee up."
        else:
            self.feedback = "Ready. Begin!"

    def process_frame_angles(self, current_angle):
        self.angle_window.append(current_angle)
        
        # Need enough history to check if moving up/down (e.g. comparing to 5 frames ago)
        if len(self.angle_window) < 5:
            return

        if self.exercise_type == "seated_knee_extension":
            self._evaluate_knee_extension()
        elif self.exercise_type == "shoulder_abduction":
            self._evaluate_shoulder_abduction()
        elif self.exercise_type == "standing_march":
            self._evaluate_standing_march()

    def _evaluate_knee_extension(self):
        # Track: Hip-Knee-Ankle (Angle increases as leg straightens)
        # IDLE (~90) -> EXTENDING -> PEAK (~165+) -> IDLE
        current_angle = self.angle_window[-1]
        past_angle = self.angle_window[-5]
        
        if self.state == "IDLE" and current_angle > 110:
            self.state = "EXTENDING"
            self.feedback = "Keep straightening! Push higher."
            
        elif self.state == "EXTENDING":
            if current_angle > 165:
                self.state = "PEAK"
                self.feedback = "Hold it! Perfect. Now lower slowly."
            elif current_angle < 130 and current_angle < past_angle - 2:
                # Dropped leg prematurely
                self.feedback = "You dropped it early! Straighten fully."
                
        elif self.state == "PEAK" and current_angle < 100:
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100
            self.feedback = f"Good rep! [{self.rep_count}] Go again."

    def _evaluate_shoulder_abduction(self):
        # Track: Hip-Shoulder-Elbow (Angle increases as arm raises)
        # IDLE (~10) -> RAISING -> PEAK (~150+) -> IDLE
        current_angle = self.angle_window[-1]
        past_angle = self.angle_window[-5]

        if self.state == "IDLE" and current_angle > 40:
            self.state = "RAISING"
            self.feedback = "Keep raising! Reach for the sky."
            
        elif self.state == "RAISING":
            if current_angle > 150:
                self.state = "PEAK"
                self.feedback = "Excellent height! Lower arm slowly."
            elif current_angle < 80 and current_angle < past_angle - 2:
                # Dropped arm prematurely
                self.feedback = "Try to push higher next time!"
                
        elif self.state == "PEAK" and current_angle < 40:
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100
            self.feedback = f"Great form! [{self.rep_count}] Raise again."

    def _evaluate_standing_march(self):
        # Track: Shoulder-Hip-Knee (Angle decreases as knee lifts)
        # IDLE (~170) -> LIFTING -> PEAK (<100) -> IDLE
        current_angle = self.angle_window[-1]
        past_angle = self.angle_window[-5]

        if self.state == "IDLE" and current_angle < 150:
            self.state = "LIFTING"
            self.feedback = "Keep lifting! Knee higher."
            
        elif self.state == "LIFTING":
            if current_angle < 100:
                self.state = "PEAK"
                self.feedback = "Hold balance! Lower leg slowly."
            elif current_angle > 140 and current_angle > past_angle + 2:
                # Put foot down prematurely
                self.feedback = "Foot down too early! Lift higher."
                
        elif self.state == "PEAK" and current_angle > 160:
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100
            self.feedback = f"Solid march! [{self.rep_count}] Knee up again."
        
    def get_stats(self):
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback
        }
