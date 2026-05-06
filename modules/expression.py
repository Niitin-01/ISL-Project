from utils.math_utils import euclidean_distance


class ExpressionDetector:
    def __init__(self):
        pass

    def detect(self, landmarks):
        if not landmarks:
            return "No Face"

        # Mouth landmarks
        left_mouth = landmarks[78]
        right_mouth = landmarks[308]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        # Distances
        mouth_width = euclidean_distance(left_mouth, right_mouth)
        mouth_height = euclidean_distance(upper_lip, lower_lip)

        # Avoid division error
        if mouth_height == 0:
            return "Neutral"

        ratio = mouth_width / mouth_height

        # Debug
        print(f"[DEBUG] Width: {mouth_width:.2f}, Height: {mouth_height:.2f}, Ratio: {ratio:.2f}")

        # 🎯 Better classification
        if mouth_height > 13 and ratio < 2.0:
            return "Surprised"

        elif ratio > 3.0:
            return "Happy"

        else:
            return "Neutral"