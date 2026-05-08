import config
from utils.math_utils import angle_between_points


class HeadMovementDetector:
    def __init__(self):
        self.prev_x = None
        self.prev_y = None

    def detect(self, landmarks):
        if not landmarks:
            self.prev_x, self.prev_y = None, None
            return "No Movement"

        # Nose tip landmark
        nose = landmarks[1]
        x, y = nose

        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return "Stable"

        # Calculate movement
        dx = x - self.prev_x
        dy = y - self.prev_y

        # Update previous
        self.prev_x, self.prev_y = x, y

        # Normalize movement using face width
        face_width = abs(landmarks[234][0] - landmarks[454][0])
        if face_width == 0:
            return "Stable"

        dx /= face_width
        dy /= face_width

        # Debug
        print(f"[HEAD] dx: {dx:.3f}, dy: {dy:.3f}")

        # Detect movement
        if abs(dy) > config.HEAD_MOVEMENT_Y_THRESHOLD:
            return "Nodding"

        elif abs(dx) > config.HEAD_MOVEMENT_X_THRESHOLD:
            return "Shaking"

        else:
            return "Stable"

    def detect_tilt(self, landmarks):
        if not landmarks:
            return "No Tilt"

        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        tilt_angle = angle_between_points(left_eye_corner, right_eye_corner)

        print(f"[HEAD] Tilt Angle: {tilt_angle:.2f}")

        if tilt_angle > config.HEAD_TILT_ANGLE_THRESHOLD:
            return "Tilted Right"
        if tilt_angle < -config.HEAD_TILT_ANGLE_THRESHOLD:
            return "Tilted Left"
        return "Level"
