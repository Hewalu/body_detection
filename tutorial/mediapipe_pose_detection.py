import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def get_screen_dimensions():
    """Get screen width and height using tkinter"""
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Get screen dimensions
    screen_width, screen_height = get_screen_dimensions()

    # Variables for movement tracking
    prev_right_wrist = None
    movement_history = []
    max_history = 10  # Number of frames to consider for movement calculation
    prev_movement_value = 0  # Track previous movement value for jump detection
    max_jump = 80  # Maximum allowed jump between frames

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

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

            # Calculate right hand movement
            movement_value = 0
            if results.pose_landmarks:
                # Get right wrist landmark (index 16)
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                # Check if right wrist is visible enough
                if right_wrist.visibility > 0.5:
                    current_pos = (right_wrist.x * new_width, right_wrist.y * new_height)

                    if prev_right_wrist is not None:
                        # Calculate distance moved
                        distance = np.sqrt((current_pos[0] - prev_right_wrist[0])**2 +
                                           (current_pos[1] - prev_right_wrist[1])**2)

                        # Filter out extreme distances (likely detection errors)
                        max_reasonable_distance = 100  # Adjust this value as needed
                        if distance < max_reasonable_distance:
                            movement_history.append(distance)
                        else:
                            # Skip this frame, use previous position
                            current_pos = prev_right_wrist

                        # Keep only recent movements
                        if len(movement_history) > max_history:
                            movement_history.pop(0)

                        # Calculate movement value (0-100) with much lower sensitivity
                        if movement_history:  # Only calculate if we have valid movements
                            avg_movement = np.mean(movement_history)
                            # Even lower scaling factor and higher exponential curve
                            scaled_movement = avg_movement * 0.25  # Reduced from 0.5 to 0.1
                            calculated_value = min(100, int(scaled_movement ** 2))  # Cubic scaling for even more extreme curve

                            # Filter extreme jumps between frames
                            jump = abs(calculated_value - prev_movement_value)
                            if jump <= max_jump or prev_movement_value == 0:
                                movement_value = calculated_value
                            else:
                                # Smooth the transition
                                if calculated_value > prev_movement_value:
                                    movement_value = prev_movement_value + max_jump
                                else:
                                    movement_value = max(0, prev_movement_value - max_jump)

                    prev_right_wrist = current_pos
                    prev_movement_value = movement_value
                else:
                    # Hand not visible enough, reset tracking
                    prev_right_wrist = None
                    movement_history = []
                    movement_value = max(0, prev_movement_value - 10)  # Gradually decrease instead of instant 0
                    prev_movement_value = movement_value
            else:
                # No pose detected, reset everything
                prev_right_wrist = None
                movement_history = []
                movement_value = max(0, prev_movement_value - 10)  # Gradually decrease instead of instant 0
                prev_movement_value = movement_value

            # Create black canvas of screen size
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Calculate position to center the frame
            y_offset = (screen_height - new_height) // 2
            x_offset = (screen_width - new_width) // 2

            # Place the frame in the center of the canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = image

            # Display movement value in top-left corner
            cv2.putText(canvas, f'Right Hand Movement: {movement_value}',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Display fullscreen
            cv2.namedWindow('MediaPipe Pose Detection', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('MediaPipe Pose Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('MediaPipe Pose Detection', canvas)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
