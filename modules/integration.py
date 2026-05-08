class ISLTextGenerator:
    def __init__(self):
        pass

    def generate(self, expression, head, gaze, eyebrow=None, eye_state=None, head_tilt=None):
        if expression == "No Face":
            return "No person detected"

        parts = ["Person is"]

        if head == "Nodding":
            parts.append("nodding")
        elif head == "Shaking":
            parts.append("shaking head")
        else:
            parts.append("keeping head stable")

        if expression == "Happy":
            parts.append("with a happy expression")
        elif expression == "Surprised":
            parts.append("with a surprised expression")
        elif expression == "Sad":
            parts.append("with a sad expression")
        elif expression == "Angry":
            parts.append("with an angry expression")
        else:
            parts.append("with a neutral expression")

        if eyebrow == "Eyebrows Raised":
            parts.append("and raised eyebrows")
        elif eyebrow == "Eyebrows Furrowed":
            parts.append("and furrowed eyebrows")

        if eye_state == "Blinking":
            parts.append("while blinking")
        elif eye_state == "Eyes Wide":
            parts.append("with widened eyes")

        if gaze == "Looking Left":
            parts.append("while looking left")
        elif gaze == "Looking Right":
            parts.append("while looking right")
        else:
            parts.append("while looking forward")

        return " ".join(parts).strip()
