from utils.math_utils import euclidean_distance
import config


class ExpressionDetector:
    def __init__(self):
        self.baseline_features = None
        self.calibration_frames = 0

    def reset(self):
        self.baseline_features = None
        self.calibration_frames = 0

    def _measure_features(self, landmarks):
        left_mouth = landmarks[78]
        right_mouth = landmarks[308]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        upper_mid_lip = landmarks[0]
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        left_face = landmarks[234]
        right_face = landmarks[454]
        upper_face = landmarks[10]
        lower_face = landmarks[152]

        face_width = euclidean_distance(left_face, right_face)
        face_height = euclidean_distance(upper_face, lower_face)
        mouth_width = euclidean_distance(left_mouth, right_mouth)
        mouth_height = euclidean_distance(upper_lip, lower_lip)

        if face_width == 0 or face_height == 0 or mouth_height == 0:
            return None

        mouth_corner_y = (left_corner[1] + right_corner[1]) / 2
        mouth_curve = (mouth_corner_y - upper_mid_lip[1]) / face_height

        return {
            "mouth_ratio": mouth_width / mouth_height,
            "mouth_width_ratio": mouth_width / face_width,
            "mouth_open_ratio": mouth_height / face_height,
            "mouth_curve": mouth_curve,
        }

    def _update_baseline(self, features):
        if self.baseline_features is None:
            self.baseline_features = features.copy()
            self.calibration_frames = 1
            return

        count = self.calibration_frames
        for key, value in features.items():
            self.baseline_features[key] = (
                (self.baseline_features[key] * count) + value
            ) / (count + 1)
        self.calibration_frames += 1

    def detect(self, landmarks, eyebrow_state="Eyebrows Neutral", eye_state="Eyes Open"):
        if not landmarks:
            self.reset()
            return "No Face"

        features = self._measure_features(landmarks)
        if features is None:
            return "Neutral"

        if self.calibration_frames < config.EXPRESSION_BASELINE_FRAMES:
            self._update_baseline(features)
            print(
                f"[EXP] Calibrating: open={features['mouth_open_ratio']:.3f}, "
                f"width={features['mouth_width_ratio']:.3f}, curve={features['mouth_curve']:.3f}"
            )
            return "Neutral"

        deltas = {
            key: features[key] - self.baseline_features[key]
            for key in features
        }

        print(
            f"[EXP] Ratio: {features['mouth_ratio']:.2f}, Open: {features['mouth_open_ratio']:.3f}, "
            f"Width: {features['mouth_width_ratio']:.3f}, Curve: {features['mouth_curve']:.3f}, "
            f"dOpen: {deltas['mouth_open_ratio']:.3f}, dWidth: {deltas['mouth_width_ratio']:.3f}, "
            f"dCurve: {deltas['mouth_curve']:.3f}"
        )

        surprise_score = 0
        happy_score = 0
        sad_score = 0
        angry_score = 0

        if deltas["mouth_open_ratio"] >= config.EXPRESSION_SURPRISE_OPEN_DELTA:
            surprise_score += 2
        if eyebrow_state == "Eyebrows Raised":
            surprise_score += 1
        if eye_state == "Eyes Wide":
            surprise_score += 1

        if deltas["mouth_width_ratio"] >= config.EXPRESSION_HAPPY_WIDTH_DELTA:
            happy_score += 2
        if deltas["mouth_curve"] <= -config.EXPRESSION_HAPPY_CURVE_DELTA:
            happy_score += 2
        if eyebrow_state == "Eyebrows Raised":
            happy_score += 1

        if deltas["mouth_curve"] >= config.EXPRESSION_SAD_CURVE_DELTA:
            sad_score += 2
        if deltas["mouth_open_ratio"] <= 0:
            sad_score += 1
        if eye_state == "Blinking":
            sad_score += 1

        if eyebrow_state == "Eyebrows Furrowed":
            angry_score += 2
        if deltas["mouth_curve"] >= config.EXPRESSION_ANGRY_CURVE_DELTA:
            angry_score += 1
        if deltas["mouth_width_ratio"] <= -config.EXPRESSION_ANGRY_WIDTH_DELTA:
            angry_score += 1

        best_expression = "Neutral"
        best_score = 0

        candidates = {
            "Surprised": surprise_score,
            "Happy": happy_score,
            "Sad": sad_score,
            "Angry": angry_score,
        }

        for expression, score in candidates.items():
            if score > best_score:
                best_expression = expression
                best_score = score

        if best_score >= 2:
            if best_expression in {"Sad", "Angry"} and deltas["mouth_curve"] < 0:
                best_expression = "Neutral"
            else:
                return best_expression

        for key, value in features.items():
            self.baseline_features[key] = (self.baseline_features[key] * 0.95) + (value * 0.05)

        if features["mouth_ratio"] > config.HAPPY_RATIO_THRESHOLD and features["mouth_curve"] < 0:
            return "Happy"

        if (
            features["mouth_open_ratio"] > self.baseline_features["mouth_open_ratio"] + 0.02
            and (eyebrow_state == "Eyebrows Raised" or eye_state == "Eyes Wide")
        ):
            return "Surprised"

        return "Neutral"
