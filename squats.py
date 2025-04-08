import mediapipe as mp
import cv2
import math

choice  = "squats"
# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    # Calculate the angle
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Initialize variables for counting repetitions for both arms
squat_rep_count = 0
prev_squat_movement = None
squat_neutral_detected = False  # To track if the squat passed through Neutral

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame
        results = pose.process(image)

        # Convert the frame back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        if results.pose_landmarks:
            # Display the current exercise choice on the frame
            cv2.putText(image, f'{choice.capitalize()}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks for the right arm
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Extract landmarks for the left arm
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # Extract landmarks for the hips and knees
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Calculate the angles for both knees
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Detect movement direction for squats
            if right_knee_angle < 120 or left_knee_angle < 120:
                squat_movement = "Eccentric"
            elif right_knee_angle > 120 and left_knee_angle > 120:
                squat_movement = "Concentric"
            else:
                squat_movement = "Neutral"

            # Squat rep counting logic
            if prev_squat_movement == "Eccentric" and squat_movement == "Concentric":
                squat_rep_count += 1
            elif prev_squat_movement == "Eccentric" and squat_movement == "Neutral":
                squat_neutral_detected = True
            elif squat_neutral_detected and prev_squat_movement == "Neutral" and squat_movement == "Concentric":
                squat_rep_count += 1
                squat_neutral_detected = False

            # Update the previous movement state
            prev_squat_movement = squat_movement

            # Print the squat movement and rep count to the console
            print(f"Squat Movement: {squat_movement}")
            print(f"Squat Reps: {squat_rep_count}")

            # Display the squat movement and rep count on the frame, positioned below the exercise tag
            cv2.putText(image, f'Squat Movement: {squat_movement}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'Squat Reps: {squat_rep_count}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
              

        # Show the frame
        cv2.imshow('Movement Detection', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit the program
            break

cap.release()
cv2.destroyAllWindows()