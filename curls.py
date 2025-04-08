import mediapipe as mp
import cv2
import math

choice  = "curls"
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
right_curls_rep_count = 0
left_curls_rep_count = 0
prev_right_curls_movement = None  # To track the previous movement state for the right arm
prev_left_curls_movement = None  # To track the previous movement state for the left arm
right_neutral_curls_detected = False  # To track if the right arm passed through Neutral
left_neutral_curls_detected = False  # To track if the left arm passed through Neutral

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

            # Calculate the angles for both elbows
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Detect movement direction for the right arm
            if right_angle < 90:
                right_movement = "Concentric"
            elif right_angle > 120:
                right_movement = "Eccentric"
            else:
                right_movement = "Neutral"

            # Detect movement direction for the left arm
            if left_angle < 90:
                left_movement = "Concentric"
            elif left_angle > 120:
                left_movement = "Eccentric"
            else:
                left_movement = "Neutral"

            # Right arm rep counting logic
            if prev_right_curls_movement == "Eccentric" and right_movement == "Concentric":
                right_curls_rep_count += 1
            elif prev_right_curls_movement == "Eccentric" and right_movement == "Neutral":
                right_neutral_curls_detected = True
            elif right_neutral_curls_detected and prev_right_curls_movement == "Neutral" and right_movement == "Concentric":
                right_curls_rep_count += 1
                right_neutral_curls_detected = False

            # Left arm rep counting logic
            if prev_left_curls_movement == "Eccentric" and left_movement == "Concentric":
                left_curls_rep_count += 1
            elif prev_left_curls_movement == "Eccentric" and left_movement == "Neutral":
                left_neutral_curls_detected = True
            elif left_neutral_curls_detected and prev_left_curls_movement == "Neutral" and left_movement == "Concentric":
                left_curls_rep_count += 1
                left_neutral_curls_detected = False

            # Update the previous movement states
            prev_right_curls_movement = right_movement
            prev_left_curls_movement = left_movement

            # Calculate the minimum of both rep counts
            min_curls_rep_count = min(right_curls_rep_count, left_curls_rep_count)

            # Print the angles, movements, and rep counts to the console
            print(f"Right Elbow Angle: {int(right_angle)}, Left Elbow Angle: {int(left_angle)}")
            print(f"Right Movement: {right_movement}, Left Movement: {left_movement}")
            print(f"Right Reps: {right_curls_rep_count}, Left Reps: {left_curls_rep_count}, Min Reps: {min_curls_rep_count}")
            # Display the movement type and rep counts on the frame, positioned below the exercise tag
            cv2.putText(image, f'Right Movement: {right_movement}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'Left Movement: {left_movement}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'Right Reps: {right_curls_rep_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f'Left Reps: {left_curls_rep_count}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f'Min Reps: {min_curls_rep_count}', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Movement Detection', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit the program
            break

cap.release()
cv2.destroyAllWindows()