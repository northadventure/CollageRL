from collections import deque
import numpy as np
import wandb

class Logger:
    def __init__(self, project_name, args, no_log):
        self.no_log = no_log
        if not no_log:
            wandb.init(project=project_name)
            wandb.run.name = wandb.run.id
            wandb.config.update(args)

    def log(self, tag, value, step, type='scalar'):
        if not self.no_log:
            if type == 'scalar':
                wandb.log({
                    tag: value,
                }, step=step)
            elif type == 'image':
                value[value < 0] = 1
                wandb.log({
                    tag: [wandb.Image(value)],
                }, step=step)

class AverageBuffer(object):
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.reset()
        
    def reset(self):
        self.vals = deque()
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.vals.append(val)
        if len(self.vals) > self.capacity:
            self.sum += val * n
            self.sum -= self.vals.popleft()
            self.avg = self.sum / self.count
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count