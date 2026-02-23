import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from kinematics import calculate_angle
from time_series_tracker import ExerciseTracker

# Setup the modern Tasks API Landmarker
base_options = python.BaseOptions(model_asset_path='RehabAI_Pipeline/pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)

detector = vision.PoseLandmarker.create_from_options(options)

# We no longer need mp.solutions for drawing, we will draw the specific angles manually.
def main():
    cap = cv2.VideoCapture(0)
    tracker = ExerciseTracker(exercise_type="squat", window_size=30)
    
    print("Starting webcam tracker... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert OpenCV's BGR format to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Step 1: Extract Local Keypoints Spatial Coordinates
        detection_result = detector.detect(mp_image)

        # Step 2: Kinematic Translation
        if len(detection_result.pose_landmarks) > 0:
            pose_landmarks = detection_result.pose_landmarks[0] # Get the first person detected
            
            # The indices for Left Hip(23), Left Knee(25), Left Ankle(27) in the BlazePose topology
            hip = [pose_landmarks[23].x, pose_landmarks[23].y]
            knee = [pose_landmarks[25].x, pose_landmarks[25].y]
            ankle = [pose_landmarks[27].x, pose_landmarks[27].y]
            
            # We transform visual frames into mathematical values representing joint angles
            angle = calculate_angle(hip, knee, ankle)
            
            # --- Phase 3: Time-Series Sequence Buffer ---
            # Feed the continuous stream of angles into our tracker heuristic
            tracker.process_frame_angles(angle)
            stats = tracker.get_stats()
            
            # --- Visualizing the Result for Debugging ---
            h, w, c = frame.shape
            knee_px = tuple(np.multiply(knee, [w, h]).astype(int))
            
            # Draw circles on the joints we care about
            for joint in [hip, knee, ankle]:
                px = tuple(np.multiply(joint, [w, h]).astype(int))
                cv2.circle(frame, px, 5, (0, 0, 255), -1) # Red dots
            
            # Display the calculated angle text right next to the knee joint
            cv2.putText(frame, str(int(angle)), knee_px, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
            # UI Dashboard (Top Left Corner)
            cv2.rectangle(frame, (0,0), (400, 160), (245, 117, 16), -1)
            cv2.putText(frame, f"STATE: {stats['state']}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"REPETITIONS: {stats['reps']}", (15, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"SCORE: {stats['score']}", (15, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FEEDBACK: {stats['feedback']}", (15, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the video with overlaid analytics
        cv2.imshow('Rehab AI - Kinematic Translation', frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
