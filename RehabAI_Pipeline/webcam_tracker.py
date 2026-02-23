import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from kinematics import calculate_angle
from time_series_tracker import ExerciseTracker

import os
import time

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pose_landmarker_lite.task')

# Setup the modern Tasks API Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)

detector = vision.PoseLandmarker.create_from_options(options)

# We no longer need mp.solutions for drawing, we will draw the specific angles manually.
def main():
    cap = cv2.VideoCapture(0)
    
    # Define the Clinical Test Progression & Goals
    test_queue = [
        {"name": "seated_knee_extension", "target_reps": 3}, # Shortened to 3 for quick testing
        {"name": "shoulder_abduction",    "target_reps": 3},
        {"name": "standing_march",        "target_reps": 3}
    ]
    
    # Store the final database records
    clinical_results = {}
    
    # Initialize the first test
    current_test_idx = 0
    tracker = ExerciseTracker(exercise_type=test_queue[current_test_idx]["name"], window_size=30)
    
    is_prep_phase = True
    prep_start_time = time.time()
    test_start_time = 0 # Will be explicitly set when prep finishes
    
    print("Starting Clinical Testing Pipeline... Press 'q' to exit early.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        target_reps = test_queue[current_test_idx]["target_reps"]
        
        if is_prep_phase:
            elapsed_prep = time.time() - prep_start_time
            prep_time_left = max(10.0 - elapsed_prep, 0.0)
            if prep_time_left <= 0:
                is_prep_phase = False
                test_start_time = time.time()
                time_left = 15.0
            else:
                time_left = 15.0 # Keep standard display variable bounded
        else:
            elapsed_time = time.time() - test_start_time
            time_left = max(15.0 - elapsed_time, 0.0)
        
        # --- Check for Test Completion ---
        if not is_prep_phase and time_left <= 0:
            # Save the Detailed Metrics
            clinical_results[tracker.exercise_type] = {
                "score": tracker.current_score,
                "max_score": target_reps * 100, # 100 pts per rep
                "reps": tracker.rep_count,
                "target_reps": target_reps
            }
            print(f"[TEST COMPLETE] {tracker.exercise_type} - Score: {tracker.current_score}")
            
            # Move to the next test
            current_test_idx += 1
            if current_test_idx >= len(test_queue):
                print("All clinical tests completed successfully!")
                break # Exit the webcam loop
                
            # Re-initialize the tracker for the next exercise
            new_test = test_queue[current_test_idx]["name"]
            tracker = ExerciseTracker(exercise_type=new_test, window_size=30)
            
            is_prep_phase = True
            prep_start_time = time.time()
            
            print(f"[PREPARING NEXT TEST] {new_test}...")
            continue # Skip rendering this transition frame

        # Convert OpenCV's BGR format to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Step 1: Extract Local Keypoints Spatial Coordinates
        detection_result = detector.detect(mp_image)

        # Step 2: Kinematic Translation
        if len(detection_result.pose_landmarks) > 0:
            pose_landmarks = detection_result.pose_landmarks[0] # Get the first person detected
            
            if tracker.exercise_type == "seated_knee_extension":
                # LEFT LEG
                l_hip = [pose_landmarks[23].x, pose_landmarks[23].y]
                l_knee = [pose_landmarks[25].x, pose_landmarks[25].y]
                l_ankle = [pose_landmarks[27].x, pose_landmarks[27].y]
                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                
                # RIGHT LEG
                r_hip = [pose_landmarks[24].x, pose_landmarks[24].y]
                r_knee = [pose_landmarks[26].x, pose_landmarks[26].y]
                r_ankle = [pose_landmarks[28].x, pose_landmarks[28].y]
                r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                
                avg_angle = (l_angle + r_angle) / 2.0
                joints_to_draw = [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
                text_positions = [(l_angle, l_knee), (r_angle, r_knee)]
                
            elif tracker.exercise_type == "shoulder_abduction":
                # LEFT ARM (Hip-Shoulder-Elbow)
                l_hip = [pose_landmarks[23].x, pose_landmarks[23].y]
                l_shoulder = [pose_landmarks[11].x, pose_landmarks[11].y]
                l_elbow = [pose_landmarks[13].x, pose_landmarks[13].y]
                l_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
                
                # RIGHT ARM
                r_hip = [pose_landmarks[24].x, pose_landmarks[24].y]
                r_shoulder = [pose_landmarks[12].x, pose_landmarks[12].y]
                r_elbow = [pose_landmarks[14].x, pose_landmarks[14].y]
                r_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
                
                avg_angle = (l_angle + r_angle) / 2.0
                joints_to_draw = [l_hip, l_shoulder, l_elbow, r_hip, r_shoulder, r_elbow]
                text_positions = [(l_angle, l_shoulder), (r_angle, r_shoulder)]
                
            elif tracker.exercise_type == "standing_march":
                # LEFT LEG (Shoulder-Hip-Knee)
                l_shoulder = [pose_landmarks[11].x, pose_landmarks[11].y]
                l_hip = [pose_landmarks[23].x, pose_landmarks[23].y]
                l_knee = [pose_landmarks[25].x, pose_landmarks[25].y]
                l_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                
                # RIGHT LEG
                r_shoulder = [pose_landmarks[12].x, pose_landmarks[12].y]
                r_hip = [pose_landmarks[24].x, pose_landmarks[24].y]
                r_knee = [pose_landmarks[26].x, pose_landmarks[26].y]
                r_angle = calculate_angle(r_shoulder, r_hip, r_knee)
                
                avg_angle = (l_angle + r_angle) / 2.0
                joints_to_draw = [l_shoulder, l_hip, l_knee, r_shoulder, r_hip, r_knee]
                text_positions = [(l_angle, l_hip), (r_angle, r_hip)]
                
                # --- SAFETY CHECK: Wrist Velocity ---
                # Grab current wrists
                l_wrist_curr = np.array([pose_landmarks[15].x, pose_landmarks[15].y])
                r_wrist_curr = np.array([pose_landmarks[16].x, pose_landmarks[16].y])
                
                # Check distance moved between frames if we have history
                if hasattr(tracker, 'prev_l_wrist'):
                    l_dist = np.linalg.norm(l_wrist_curr - tracker.prev_l_wrist)
                    r_dist = np.linalg.norm(r_wrist_curr - tracker.prev_r_wrist)
                    
                    # If BOTH wrists are moving significantly (e.g. > 0.05 normalized screen space per frame)
                    # It means they aren't holding onto a stationary object like a chair.
                    if l_dist > 0.02 and r_dist > 0.02:
                        tracker.feedback = "WARNING: Hold onto a chair!"
                
                # Update history
                tracker.prev_l_wrist = l_wrist_curr
                tracker.prev_r_wrist = r_wrist_curr

            # --- Phase 3: Time-Series Sequence Buffer ---
            if not is_prep_phase:
                tracker.process_frame_angles(avg_angle)
            stats = tracker.get_stats()
            
            # --- Visualizing the Result for Debugging ---
            h, w, c = frame.shape
            
            # Draw circles on ALL active joints
            for joint in joints_to_draw:
                px = tuple(np.multiply(joint, [w, h]).astype(int))
                cv2.circle(frame, px, 5, (0, 0, 255), -1) # Red dots
            
            # Display the calculated angle text
            for angle_val, joint_pos in text_positions:
                px = tuple(np.multiply(joint_pos, [w, h]).astype(int))
                cv2.putText(frame, str(int(angle_val)), px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
            # --- UI Dashboard (Moved to Right Sidebar) ---
            SIDEBAR_WIDTH = 420
            
            # Create a larger canvas: webcam frame on the left, black sidebar on the right
            canvas = np.zeros((h, w + SIDEBAR_WIDTH, 3), dtype=np.uint8)
            canvas[:, :w] = frame # Copy the webcam feed to the left side
            
            # --- Top Banner: Activity & Direction (Now in the Sidebar) ---
            if tracker.exercise_type == "seated_knee_extension":
                ex_name = "SEATED KNEE EXT."
                ex_dir = "DIR: SIDE FACING"
            elif tracker.exercise_type == "shoulder_abduction":
                ex_name = "SHOULDER ABD."
                ex_dir = "DIR: FRONT FACING"
            elif tracker.exercise_type == "standing_march":
                ex_name = "STANDING MARCH"
                ex_dir = "DIR: SIDE FACING"
            else:
                ex_name = tracker.exercise_type.upper()
                ex_dir = "DIR: ANY"
                
            # Draw Sidebar Background
            cv2.rectangle(canvas, (w, 0), (w + SIDEBAR_WIDTH, h), (30, 30, 30), -1)

            # Draw Banner Background
            cv2.rectangle(canvas, (w + 10, 10), (w + SIDEBAR_WIDTH - 10, 80), (0, 0, 0), -1)
            cv2.putText(canvas, f"ACTIVITY: {ex_name}", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{ex_dir}", (w + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # --- Dashboard Background ---
            cv2.rectangle(canvas, (w + 10, 100), (w + SIDEBAR_WIDTH - 10, 310), (245, 117, 16), -1)
            
            if is_prep_phase:
                cv2.putText(canvas, f"PREPARING: {int(prep_time_left)}s", (w + 20, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, f"TIME LEFT: {int(time_left)}s", (w + 20, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            
            cv2.putText(canvas, f"STATE: {stats['state']}", (w + 20, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"REPETITIONS: {stats['reps']}/{target_reps}", (w + 20, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"SCORE: {stats['score']}/{target_reps * 100}", (w + 20, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Action Feedback Block
            cv2.rectangle(canvas, (w + 10, 330), (w + SIDEBAR_WIDTH - 10, h - 10), (0, 0, 0), -1)
            cv2.putText(canvas, "SYSTEM FEEDBACK:", (w + 20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Wrap long feedback text
            feedback_text = "GET INTO POSITION!" if is_prep_phase else stats['feedback']
            cv2.putText(canvas, feedback_text, (w + 20, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_prep_phase else (0, 255, 0), 2, cv2.LINE_AA)

        # Display the video with overlaid analytics
        cv2.imshow('Rehab AI - Kinematic Translation', canvas)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Even if they quit early, save the current progress of the active test
            if tracker.exercise_type not in clinical_results:
                clinical_results[tracker.exercise_type] = {
                    "score": tracker.current_score,
                    "max_score": target_reps * 100,
                    "reps": tracker.rep_count,
                    "target_reps": target_reps
                }
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # --- Data Export Phase ---
    if clinical_results:
        import datetime
        export_path = os.path.join(script_dir, "clinical_results.txt")
        print(f"\n[EXPORT] Saving Clinical Results to: {export_path}")
        
        with open(export_path, "w") as f:
            f.write("==============================================================\n")
            f.write("            REHAB AI - CLINICAL TRACKER REPORT                \n")
            f.write("==============================================================\n")
            f.write(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"{'EXERCISE MODULE':<25} | {'REPETITIONS':<12} | {'SCORE MAP':<12}\n")
            f.write("-" * 62 + "\n")
            
            total_score = 0
            max_possible_score = 0
            
            for test_name, data in clinical_results.items():
                clean_name = test_name.replace("_", " ").title()
                score_str = f"{data['score']}/{data['max_score']}"
                rep_str = f"{data['reps']}/{data['target_reps']}"
                
                f.write(f"{clean_name:<25} | {rep_str:<12} | {score_str:<12}\n")
                
                total_score += data['score']
                max_possible_score += data['max_score']
                
            f.write("\n" + "=" * 62 + "\n")
            f.write(f"AGGREGATE CLINICAL SCORE  : {total_score} / {max_possible_score} Points\n")
            f.write("==============================================================\n")
        print("[EXPORT] Complete. Safe to close.")
    else:
        print("\n[EXPORT] No completed tests to save.")

if __name__ == '__main__':
    main()
