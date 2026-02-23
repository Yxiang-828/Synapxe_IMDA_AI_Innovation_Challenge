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
        self.state = "CALIBRATING"  # CALIBRATING -> IDLE -> DESCENDING -> ASCENDING -> COMPLETED
        self.rep_count = 0
        self.calibration_frames = 0 # Need 30 continuous frames to unlock tracking
        
        # Scoring metrics purely based on kinematics
        self.current_score = 0
        
        # Set initial calibration instructions
        if self.exercise_type == "seated_knee_extension":
            self.feedback = "Sit in chair. Camera side. Feet flat."
        elif self.exercise_type == "shoulder_abduction":
            self.feedback = "Stand facing camera. Arms at sides."
        elif self.exercise_type == "standing_march":
            self.feedback = "Stand sideways. Hold a chair."
        else:
            self.feedback = "Calibrating posture..."

    def process_frame_angles(self, current_angle):
        """
        Feed the incoming mathematical angle into our continuous time-series buffer.
        """
        self.angle_window.append(current_angle)
        
        # We need a full window of frames before we can evaluate a "pattern"
        if len(self.angle_window) < self.angle_window.maxlen:
            return

        if self.exercise_type == "seated_knee_extension":
            self._evaluate_knee_extension()
        elif self.exercise_type == "shoulder_abduction":
            self._evaluate_shoulder_abduction()
        elif self.exercise_type == "standing_march":
            self._evaluate_standing_march()

    def _evaluate_knee_extension(self):
        """
        Track: Hip-Knee-Ankle
        IDLE (~90 deg) -> EXTENDING -> PEAK (~170+ deg) -> IDLE
        """
        current_angle = self.angle_window[-1]
        
        # --- CALIBRATION ROUTINE ---
        if self.state == "CALIBRATING":
            if 80 <= current_angle <= 110:
                self.calibration_frames += 1
                if self.calibration_frames >= 30: # Held for 1 second
                    self.state = "IDLE"
                    self.feedback = "Begin: Straighten your leg!" # TELL THEM WHAT TO DO
            else:
                self.calibration_frames = 0 # Reset if they move
            return # Block reps until calibrated
            
        # --- EXECUTION STATE MACHINE ---
        if self.state == "IDLE" and current_angle > 110:
            self.state = "EXTENDING"
            self.feedback = "Straighten it out..."
            
        elif self.state == "EXTENDING" and current_angle > 165:
            # Reached full healthy extension
            self.state = "PEAK"
            self.feedback = "Hold it! Now lower slowly."
            
        elif self.state == "PEAK" and current_angle < 100:
            # Returned to resting seated position
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100
            self.feedback = f"Good extension! ({self.rep_count}/10)"
            
        elif self.state == "EXTENDING" and current_angle < 130 and current_angle < self.angle_window[-5]:
            # They dropped their leg before reaching full extension
            self.feedback = "Try to straighten it fully!"
            
        elif self.state == "IDLE":
             self.feedback = "Begin: Straighten your leg!"

    def _evaluate_shoulder_abduction(self):
        """
        Track: Hip-Shoulder-Elbow
        IDLE (~10 deg) -> RAISING -> PEAK (~160+ deg) -> IDLE
        """
        current_angle = self.angle_window[-1]

        # --- CALIBRATION ROUTINE ---
        if self.state == "CALIBRATING":
            if current_angle <= 25:
                self.calibration_frames += 1
                if self.calibration_frames >= 30:
                    self.state = "IDLE"
                    self.feedback = "Begin: Raise arm to side!" # TELL THEM WHAT TO DO
            else:
                self.calibration_frames = 0
            return

        if self.state == "IDLE" and current_angle > 30:
            self.state = "RAISING"
            self.feedback = "Push higher..."
            
        elif self.state == "RAISING" and current_angle > 150:
            # Good mobility achieved
            self.state = "PEAK"
            self.feedback = "Great height! Lower slowly."
            
        elif self.state == "PEAK" and current_angle < 40:
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100
            self.feedback = f"Good raise! ({self.rep_count}/5)"
            
        elif self.state == "RAISING" and current_angle < 80 and current_angle < self.angle_window[-5]:
            # Dropped arm early
            self.feedback = "Push higher if you can!"
            
        elif self.state == "IDLE":
             self.feedback = "Begin: Raise arm to side!"

    def _evaluate_standing_march(self):
        """
        Track: Shoulder-Hip-Knee
        IDLE (~170 deg) -> LIFTING -> PEAK (< 100 deg) -> IDLE
        NOTE: Safety logic (Wrist velocity) is injected directly in webcam_tracker before this is called.
        """
        current_angle = self.angle_window[-1]

        # --- CALIBRATION ROUTINE ---
        if self.state == "CALIBRATING":
            if 165 <= current_angle <= 185: # Standing mostly straight
                self.calibration_frames += 1
                if self.calibration_frames >= 30:
                    self.state = "IDLE"
                    self.feedback = "Begin: March knee HIGH!" # TELL THEM WHAT TO DO
            else:
                self.calibration_frames = 0
            return

        if self.state == "IDLE" and current_angle < 150:
            self.state = "LIFTING"
            self.feedback = "Lift knee up..."
            
        elif self.state == "LIFTING" and current_angle < 100:
            # Knee is parallel to floor
            self.state = "PEAK"
            self.feedback = "Hold balance! Lower slowly."
            
        elif self.state == "PEAK" and current_angle > 160:
            self.state = "IDLE"
            self.rep_count += 1
            self.current_score += 100
            self.feedback = f"Good march! ({self.rep_count}/10)"
            
        elif self.state == "LIFTING" and current_angle > 140 and current_angle > self.angle_window[-5]:
            # Put foot down too early
            self.feedback = "Lift knee higher!"
            
        elif self.state == "IDLE":
             self.feedback = "Begin: March knee HIGH!"
            
    def get_stats(self):
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback
        }
