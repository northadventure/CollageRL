import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, canvas, source, goal, step, action, shape, reward, next_canvas, next_source, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (canvas, source, goal, step, action, shape, reward, next_canvas, next_source, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        canvas, source, goal, step, action, shape, reward, next_canvas, next_source, done = map(np.stack, zip(*batch))
        
        canvas = canvas.squeeze(1)
        source = source.squeeze(1)
        goal = goal.squeeze(1)
        shape = shape.squeeze(1)
        next_canvas = next_canvas.squeeze(1)
        next_source = next_source.squeeze(1)

        action = action.squeeze(1)

        return canvas, source, goal, step, action, shape, reward, next_canvas, next_source, done

    def __len__(self):
        return len(self.buffer)