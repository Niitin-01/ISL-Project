import config

from utils.math_utils import euclidean_distance


class EyeStateDetector:
    def __init__(self):
        pass

    def _eye_ratio(self, top, bottom, left_corner, right_corner):
        eye_width = euclidean_distance(left_corner, right_corner)
        if eye_width == 0:
            return 0.0
        eye_height = euclidean_distance(top, bottom)
        return eye_height / eye_width

    def detect(self, landmarks):
        if not landmarks:
            return "No Eye State"

        left_ratio = self._eye_ratio(
            landmarks[159],
            landmarks[145],
            landmarks[33],
            landmarks[133],
        )
        right_ratio = self._eye_ratio(
            landmarks[386],
            landmarks[374],
            landmarks[362],
            landmarks[263],
        )
        average_ratio = (left_ratio + right_ratio) / 2

        print(f"[EYE] Opening Ratio: {average_ratio:.3f}")

        if average_ratio <= config.EYE_CLOSED_RATIO_THRESHOLD:
            return "Blinking"
        if average_ratio >= config.EYE_WIDE_RATIO_THRESHOLD:
            return "Eyes Wide"
        return "Eyes Open"
