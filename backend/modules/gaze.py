import config


class GazeDetector:
    def __init__(self):
        pass

    def detect(self, landmarks):
        if not landmarks:
            return "No Gaze"

        # Left eye landmarks
        left_eye_left = landmarks[33]
        left_eye_right = landmarks[133]
        left_iris = landmarks[468]  # iris center

        # Get x positions
        eye_left_x = left_eye_left[0]
        eye_right_x = left_eye_right[0]
        iris_x = left_iris[0]

        # Normalize iris position within eye
        eye_width = eye_right_x - eye_left_x
        if eye_width == 0:
            return "Center"

        ratio = (iris_x - eye_left_x) / eye_width

        # Debug
        print(f"[GAZE] Ratio: {ratio:.2f}")

        # Classification
        if ratio < config.GAZE_LEFT_THRESHOLD:
            return "Looking Left"
        elif ratio > config.GAZE_RIGHT_THRESHOLD:
            return "Looking Right"
        else:
            return "Center"
