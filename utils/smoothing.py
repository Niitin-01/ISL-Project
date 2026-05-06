from collections import deque, Counter


class TemporalSmoother:
    def __init__(self, window_size=10):
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)

        # Return most common value
        most_common = Counter(self.window).most_common(1)[0][0]
        return most_common