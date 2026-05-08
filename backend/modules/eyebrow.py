import config

from utils.math_utils import euclidean_distance


class EyebrowDetector:
    def __init__(self):
        self.baseline_gap = None
        self.baseline_spread = None
        self.calibration_frames = 0

    def reset(self):
        self.baseline_gap = None
        self.baseline_spread = None
        self.calibration_frames = 0

    def detect(self, landmarks):
        if not landmarks:
            self.reset()
            return "No Eyebrow Cue"

        left_brow_points = [landmarks[70], landmarks[63], landmarks[105]]
        right_brow_points = [landmarks[300], landmarks[293], landmarks[334]]
        left_eye_points = [landmarks[159], landmarks[145]]
        right_eye_points = [landmarks[386], landmarks[374]]
        left_inner_brow = landmarks[105]
        right_inner_brow = landmarks[334]
        chin = landmarks[152]
        forehead = landmarks[10]
        left_face = landmarks[234]
        right_face = landmarks[454]

        face_height = euclidean_distance(forehead, chin)
        face_width = euclidean_distance(left_face, right_face)
        if face_height == 0 or face_width == 0:
            return "Eyebrows Neutral"

        left_brow_y = sum(point[1] for point in left_brow_points) / len(left_brow_points)
        right_brow_y = sum(point[1] for point in right_brow_points) / len(right_brow_points)
        left_eye_y = sum(point[1] for point in left_eye_points) / len(left_eye_points)
        right_eye_y = sum(point[1] for point in right_eye_points) / len(right_eye_points)

        left_gap = abs(left_eye_y - left_brow_y) / face_height
        right_gap = abs(right_eye_y - right_brow_y) / face_height
        average_gap = (left_gap + right_gap) / 2
        brow_spread = euclidean_distance(left_inner_brow, right_inner_brow) / face_width

        if self.baseline_gap is None:
            self.baseline_gap = average_gap
            self.baseline_spread = brow_spread
            self.calibration_frames = 1
            return "Eyebrows Neutral"

        if self.calibration_frames < config.EYEBROW_BASELINE_FRAMES:
            weight = self.calibration_frames / (self.calibration_frames + 1)
            self.baseline_gap = (self.baseline_gap * weight) + (average_gap / (self.calibration_frames + 1))
            self.baseline_spread = (self.baseline_spread * weight) + (brow_spread / (self.calibration_frames + 1))
            self.calibration_frames += 1
            print(f"[BROW] Calibrating: gap={average_gap:.3f}, spread={brow_spread:.3f}")
            return "Eyebrows Neutral"

        gap_delta = average_gap - self.baseline_gap
        spread_delta = self.baseline_spread - brow_spread

        print(
            f"[BROW] Gap: {average_gap:.3f}, Base: {self.baseline_gap:.3f}, "
            f"Delta: {gap_delta:.3f}, Spread Delta: {spread_delta:.3f}"
        )

        if gap_delta >= config.EYEBROW_RAISED_DELTA:
            return "Eyebrows Raised"
        if (
            gap_delta <= -config.EYEBROW_FURROWED_DELTA
            or spread_delta >= config.EYEBROW_SPREAD_FURROWED_DELTA
        ):
            return "Eyebrows Furrowed"

        self.baseline_gap = (self.baseline_gap * 0.9) + (average_gap * 0.1)
        self.baseline_spread = (self.baseline_spread * 0.9) + (brow_spread * 0.1)
        return "Eyebrows Neutral"
