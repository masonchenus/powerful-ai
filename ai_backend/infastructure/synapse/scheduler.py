# ai_backend/infrastructure/synapse/scheduler.py
from collections import deque

class PriorityScheduler:
    def __init__(self):
        self.high_prio = deque() # User chat
        self.low_prio = deque()  # 100M math equations
        self.batch_size = 32

    def add_request(self, task, priority="low"):
        if priority == "high":
            self.high_prio.append(task)
        else:
            self.low_prio.append(task)

    def get_next_batch(self):
        """
        Always gives the high-priority user tasks first.
        If there's room left in the GPU, it fills it with math.
        """
        batch = []
        # Fill with high priority first
        while len(batch) < self.batch_size and self.high_prio:
            batch.append(self.high_prio.popleft())
        
        # Fill remaining gaps with background math
        while len(batch) < self.batch_size and self.low_prio:
            batch.append(self.low_prio.popleft())
            
        return batch