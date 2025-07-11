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
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = get_screen_dimensions()
    manual_mode = False
    sender = DataSender("192.168.161.255", 8080)
    stop_sending = threading.Event()

    def send_data_periodically(sender_instance, stop_event):
        while not stop_event.is_set():
            sender_instance.send()
            # Wait for 100ms before the next send
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
                manual_mode = not manual_mode
                # Send mode switch message to ESP
                sender.add_data("mode_switch", "manual" if manual_mode else "automatic")

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

            # --- Angle Mode Logic (always active now) ---
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
                # Left bar (blue) for left arm angle
                bar_width = min(x_offset - 20, 60)
                bar_height = int((left_arm_angle / 180) * (screen_height - 100))
                bar_x = 10
                bar_y = screen_height - 50 - bar_height
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, screen_height - 50), (255, 0, 0), -1)

                # Right bar (red) for right arm angle
                bar_height_right = int((right_arm_angle / 180) * (screen_height - 100))
                bar_x_right = screen_width - bar_width - 10
                bar_y_right = screen_height - 50 - bar_height_right
                cv2.rectangle(canvas, (bar_x_right, bar_y_right), (bar_x_right + bar_width, screen_height - 50), (0, 0, 255), -1)

            # Send angle data only if not in manual mode
            if not manual_mode:
                sender.add_data("left_arm_angle", left_arm_angle)
                sender.add_data("right_arm_angle", right_arm_angle)

            # Display current mode
            if manual_mode:
                mode_text = "Manueller Modus aktiv - M um Modus wechseln"
                text_color = (0, 255, 255)  # Yellow for manual mode
            else:
                mode_text = "Automatischer Modus - M um Modus wechseln"
                text_color = (0, 255, 0)  # Green for automatic mode

            cv2.putText(canvas, mode_text, (20, screen_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

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
