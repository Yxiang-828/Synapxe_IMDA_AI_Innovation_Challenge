import numpy as np
from collections import deque

class ExerciseTracker:
    def __init__(self, exercise_type="squat", window_size=30):
        """
        Tracks a time-series window of joint angles to evaluate exercises.
        window_size: How many frames to keep in memory (e.g. 30 frames = ~1 second at 30fps)
        """
        self.exercise_type = exercise_type
        # deque automatically pops the oldest item when it reaches maxlen
        self.angle_window = deque(maxlen=window_size)
        
        # State Tracking
        self.state = "IDLE"  # IDLE -> DESCENDING -> ASCENDING -> COMPLETED
        self.rep_count = 0
        
        # Scoring metrics purely based on kinematics
        self.current_score = 0
        self.feedback = "Ready"

    def process_frame_angles(self, current_angle):
        """
        Feed the incoming mathematical angle into our continuous time-series buffer.
        """
        self.angle_window.append(current_angle)
        
        # We need a full window of frames before we can evaluate a "pattern"
        if len(self.angle_window) < self.angle_window.maxlen:
            return

        self._evaluate_squat()

    def _evaluate_squat(self):
        """
        A heuristic-based evaluation simulating what an LSTM does:
        Analyzing the continuous pattern of the knee angle.
        
        Standing straight -> Knee angle ~170-180 degrees
        Deep squat -> Knee angle ~70-90 degrees
        """
        current_angle = self.angle_window[-1]
        
        # 1. State Machine Translation
        if self.state == "IDLE" and current_angle < 150:
            self.state = "DESCENDING"
            self.feedback = "Going down..."
            
        elif self.state == "DESCENDING" and current_angle < 90:
            # They hit the perfect depth
            self.state = "BOTTOM"
            self.feedback = "Good depth! Push up!"
            
        elif self.state == "BOTTOM" and current_angle > 160:
            # They returned to standing
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100 # Perfect rep
            self.feedback = "Excellent rep!"
            
        elif self.state == "DESCENDING" and current_angle > 140 and current_angle > self.angle_window[-5]:
            # They started standing up without hitting the 90-degree depth mark
            # (Checking window[-5] sees if the angle is increasing over the last 5 frames)
            self.feedback = "Go deeper!"
            
    def get_stats(self):
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback
        }
