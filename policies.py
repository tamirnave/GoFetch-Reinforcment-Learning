import random
import numpy as np

class policies:
    def __init__(self, eps0=0.9, eps_decay=0, eps_min=0.2, selected_policy="greedy_eps", tau=5):
        self.iter = 0
        self.tau = tau
        self.eps = eps0
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.selected_policy = selected_policy

    def select_action(self, q):
        if self.selected_policy=="greedy_eps":
            return self.greedy_eps(q)

        if self.selected_policy == "softmax":
            return self.softmax_selection(q)

    def greedy_eps(self, q):
        self.iter = self.iter + 1
        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        if random.random() <= self.eps:
            return random.randint(0, q.shape[0] - 1)
        else:
            return np.argmax(q)

    def softmax_selection(self, q):
        tmp = np.cumsum(np.exp(q / self.tau))
        tmp = np.hstack((0, tmp))
        a = random.uniform(0, tmp[-1])
        max_p = [x for x in range(tmp.shape[0] - 1) if a >= tmp[x] and a <= tmp[x + 1]]
        if len(max_p) > 1:
            return max_p[random.randint(0, len(max_p) - 1)]
        else:
            return max_p[0]