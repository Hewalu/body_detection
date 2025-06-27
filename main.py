import cv2
import mediapipe as mp
import numpy as np
from send import DataSender
from pose_detection import get_screen_dimensions, calculate_angle
import threading

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Get screen dimensions
    screen_width, screen_height = get_screen_dimensions()

    # --- Mode Management ---
    MOVEMENT_MODE = 0
    ANGLE_MODE = 1
    NUM_MODES = 2
    current_mode = ANGLE_MODE

    # Variables for movement tracking
    prev_left_wrist = None  # This is actually the right hand due to mirroring
    right_movement_history = []
    prev_right_wrist = None  # This is actually the left hand due to mirroring
    left_movement_history = []
    max_history = 10  # Number of frames to consider for movement calculation
    prev_right_movement_value = 0  # Track previous movement value for jump detection
    prev_left_movement_value = 0
    max_jump = 80  # Maximum allowed jump between frames

    # Initialize DataSender
    sender = DataSender()

    # --- Threading for sending data ---
    stop_sending = threading.Event()

    def send_data_periodically(sender_instance, stop_event):
        while not stop_event.is_set():
            sender_instance.send()
            # Wait for 500ms before the next send
            stop_event.wait(0.1)

    sender_thread = threading.Thread(target=send_data_periodically, args=(sender, stop_sending))
    sender_thread.start()

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # --- Key Press Handling ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                current_mode = (current_mode + 1) % NUM_MODES

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Get original frame dimensions
            original_height, original_width = frame.shape[:2]

            # Calculate new dimensions maintaining aspect ratio
            aspect_ratio = original_width / original_height

            # Calculate dimensions to fit screen while maintaining aspect ratio
            if screen_width / screen_height > aspect_ratio:
                # Screen is wider than frame aspect ratio - use full height
                new_height = screen_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Screen is taller than frame aspect ratio - use full width
                new_width = screen_width
                new_height = int(new_width / aspect_ratio)

            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height))

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Create black canvas of screen size
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Calculate position to center the frame
            y_offset = (screen_height - new_height) // 2
            x_offset = (screen_width - new_width) // 2

            # Place the frame in the center of the canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = image

            # --- Mode-specific Logic and Display ---
            if current_mode == MOVEMENT_MODE:
               # Calculate hand movements
                right_movement_value = 0  # Right hand in mirror
                left_movement_value = 0  # Left hand in mirror

                if results.pose_landmarks:
                    # Get left wrist landmark (appears as right hand due to mirroring)
                    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                    # Get right wrist landmark (appears as left hand due to mirroring)
                    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                    # Process left wrist (right hand in mirror)
                    if left_wrist.visibility > 0.5:
                        current_left_pos = (left_wrist.x * new_width, left_wrist.y * new_height)

                        if prev_left_wrist is not None:
                            # Calculate distance moved
                            distance = np.sqrt((current_left_pos[0] - prev_left_wrist[0])**2 +
                                               (current_left_pos[1] - prev_left_wrist[1])**2)

                            # Filter out extreme distances (likely detection errors)
                            max_reasonable_distance = 100
                            if distance < max_reasonable_distance:
                                right_movement_history.append(distance)
                            else:
                                current_left_pos = prev_left_wrist

                            # Keep only recent movements
                            if len(right_movement_history) > max_history:
                                right_movement_history.pop(0)

                            # Calculate movement value
                            if right_movement_history:
                                avg_movement = np.mean(right_movement_history)
                                scaled_movement = avg_movement * 0.25
                                calculated_value = min(100, int(scaled_movement ** 2))

                                # Filter extreme jumps between frames
                                jump = abs(calculated_value - prev_right_movement_value)
                                if jump <= max_jump or prev_right_movement_value == 0:
                                    right_movement_value = calculated_value
                                else:
                                    if calculated_value > prev_right_movement_value:
                                        right_movement_value = prev_right_movement_value + max_jump
                                    else:
                                        right_movement_value = max(0, prev_right_movement_value - max_jump)

                        prev_left_wrist = current_left_pos
                        prev_right_movement_value = right_movement_value
                    else:
                        prev_left_wrist = None
                        right_movement_history = []
                        right_movement_value = max(0, prev_right_movement_value - 10)
                        prev_right_movement_value = right_movement_value

                    # Process right wrist (left hand in mirror)
                    if right_wrist.visibility > 0.5:
                        current_right_pos = (right_wrist.x * new_width, right_wrist.y * new_height)

                        if prev_right_wrist is not None:
                            distance = np.sqrt((current_right_pos[0] - prev_right_wrist[0])**2 +
                                               (current_right_pos[1] - prev_right_wrist[1])**2)

                            max_reasonable_distance = 100
                            if distance < max_reasonable_distance:
                                left_movement_history.append(distance)
                            else:
                                current_right_pos = prev_right_wrist

                            if len(left_movement_history) > max_history:
                                left_movement_history.pop(0)

                            if left_movement_history:
                                avg_movement = np.mean(left_movement_history)
                                scaled_movement = avg_movement * 0.25
                                calculated_value = min(100, int(scaled_movement ** 2))

                                jump = abs(calculated_value - prev_left_movement_value)
                                if jump <= max_jump or prev_left_movement_value == 0:
                                    left_movement_value = calculated_value
                                else:
                                    if calculated_value > prev_left_movement_value:
                                        left_movement_value = prev_left_movement_value + max_jump
                                    else:
                                        left_movement_value = max(0, prev_left_movement_value - max_jump)

                        prev_right_wrist = current_right_pos
                        prev_left_movement_value = left_movement_value
                    else:
                        prev_right_wrist = None
                        left_movement_history = []
                        left_movement_value = max(0, prev_left_movement_value - 10)
                        prev_left_movement_value = left_movement_value
                else:
                    # No pose detected, reset everything
                    prev_left_wrist = None
                    right_movement_history = []
                    right_movement_value = max(0, prev_right_movement_value - 10)
                    prev_right_movement_value = right_movement_value

                    prev_right_wrist = None
                    left_movement_history = []
                    left_movement_value = max(0, prev_left_movement_value - 10)
                    prev_left_movement_value = left_movement_value

                # Display movement values
                cv2.putText(canvas, f'Left Hand: {left_movement_value}',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(canvas, f'Right Hand: {right_movement_value}',
                            (screen_width - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # Draw movement bars on black borders
                if x_offset > 0:
                    # Left bar (blue) for left hand movement
                    bar_width = min(x_offset - 20, 60)
                    bar_height = int((left_movement_value / 100) * (screen_height - 100))
                    bar_x = 10
                    bar_y = screen_height - 50 - bar_height
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, screen_height - 50), (255, 0, 0), -1)

                    # Right bar (red) for right hand movement
                    bar_height_right = int((right_movement_value / 100) * (screen_height - 100))
                    bar_x_right = screen_width - bar_width - 10
                    bar_y_right = screen_height - 50 - bar_height_right
                    cv2.rectangle(canvas, (bar_x_right, bar_y_right), (bar_x_right + bar_width, screen_height - 50), (0, 0, 255), -1)

                # send data
                sender.add_data("left_movement_value", left_movement_value)
                sender.add_data("right_movement_value", right_movement_value)

            elif current_mode == ANGLE_MODE:
                right_arm_angle = 0
                left_arm_angle = 0
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Left Arm Angle (appears as right arm in mirror)
                    try:
                        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                        right_arm_angle = calculate_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])

                        # Visualize angle on canvas
                        elbow_pos = (int(elbow.x * new_width + x_offset), int(elbow.y * new_height + y_offset))
                        cv2.circle(canvas, elbow_pos, 25, (255, 255, 0), 2)
                        cv2.putText(canvas, str(int(right_arm_angle)),
                                    (elbow_pos[0] - 20, elbow_pos[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    except:
                        pass

                    # Right Arm Angle (appears as left arm in mirror)
                    try:
                        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                        left_arm_angle = calculate_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])

                        # Visualize angle on canvas
                        elbow_pos = (int(elbow.x * new_width + x_offset), int(elbow.y * new_height + y_offset))
                        cv2.circle(canvas, elbow_pos, 25, (255, 255, 0), 2)
                        cv2.putText(canvas, str(int(left_arm_angle)),
                                    (elbow_pos[0] - 20, elbow_pos[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    except:
                        pass

                # Display angle values
                cv2.putText(canvas, f'Left Arm Angle: {int(left_arm_angle)}',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(canvas, f'Right Arm Angle: {int(right_arm_angle)}',
                            (screen_width - 550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # Draw movement bars on black borders
                if x_offset > 0:
                    # Left bar (blue) for left hand movement
                    bar_width = min(x_offset - 20, 60)
                    bar_height = int((left_arm_angle / 180) * (screen_height - 100))
                    bar_x = 10
                    bar_y = screen_height - 50 - bar_height
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, screen_height - 50), (255, 0, 0), -1)

                    # Right bar (red) for right hand movement
                    bar_height_right = int((right_arm_angle / 180) * (screen_height - 100))
                    bar_x_right = screen_width - bar_width - 10
                    bar_y_right = screen_height - 50 - bar_height_right
                    cv2.rectangle(canvas, (bar_x_right, bar_y_right), (bar_x_right + bar_width, screen_height - 50), (0, 0, 255), -1)

                # send data
                sender.add_data("left_arm_angle", left_arm_angle)
                sender.add_data("right_arm_angle", right_arm_angle)

            # Display current mode
            mode_text = "Mode: Movement" if current_mode == MOVEMENT_MODE else "Mode: Angles"
            cv2.putText(canvas, f"{mode_text} (Press 'm' to switch)", (20, screen_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display fullscreen
            cv2.namedWindow('MediaPipe Pose Detection', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('MediaPipe Pose Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('MediaPipe Pose Detection', canvas)

    # stop the sender thread
    stop_sending.set()
    sender_thread.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
