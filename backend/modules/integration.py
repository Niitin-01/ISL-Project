class ISLTextGenerator:
    def __init__(self):
        pass

    def generate(self, expression, head, gaze):
        if expression == "No Face":
            return "No person detected"

        sentence = "Person is "

        # Head movement
        if head == "Nodding":
            sentence += "nodding "
        elif head == "Shaking":
            sentence += "shaking head "
        else:
            sentence += "keeping head stable "

        # Expression
        if expression == "Happy":
            sentence += "with a happy expression "
        elif expression == "Surprised":
            sentence += "with a surprised expression "
        else:
            sentence += "with a neutral expression "

        # Gaze
        if gaze == "Looking Left":
            sentence += "while looking left"
        elif gaze == "Looking Right":
            sentence += "while looking right"
        else:
            sentence += "while looking forward"

        return sentence.strip()