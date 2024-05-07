class PredictionBuffer:
    def __init__(self, buffer_size = 30):
        self.buffer_size = buffer_size
        self.accuracy_buffer = []

    def add_accuracy(self, accuracy):
        """
        Add accuracy of a prediction to the buffer.
        """
        self.accuracy_buffer.append(accuracy)
        # Maintain the buffer size
        if len(self.accuracy_buffer) > self.buffer_size:
            self.accuracy_buffer.pop(0)

    def get_average_accuracy(self):
        """
        Calculate the average accuracy from the buffer.
        """
        if not self.accuracy_buffer and len(self.accuracy_buffer) >= self.buffer_size - 1:
            return 0.0
        return sum(self.accuracy_buffer) / len(self.accuracy_buffer)

    def restart_buffer(self):
        self.accuracy_buffer = []