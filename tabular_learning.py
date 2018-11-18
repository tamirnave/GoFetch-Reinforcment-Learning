import numpy as np

class tabular_learning:
    def __init__(self, game, alpha=0.01, gamma=0.8, type_of_evaluation="Q_Learning"):
        self.Q_table = np.zeros((game.number_of_actions, game.total_num_of_states()))
        self.alpha = alpha
        self.type_of_evaluation = type_of_evaluation
        self.game = game
        self.gamma = gamma

    def Monte_Carlo(self, action, reward):
        self.Q_table[action, self.game.state_to_ind(game.prev_state)] = reward

    def Q_Learning(self, action, reward):
        '''max_val = 0
        for a in range(game.number_of_actions):
            tmp = Q_table[a, game.state_to_ind()]
            if tmp > max_val:
                max_val = tmp'''
        max_val = np.max(self.Q_table[:, self.game.state_to_ind()])
        self.Q_table[action, self.game.state_to_ind(self.game.prev_state)] += self.alpha * (reward + self.gamma * max_val)  # = (reward + gamma * max_val) #

    def tabular_play_iter(self, policy):
        # Select Best Action
        action = policy.select_action(self.Q_table[:, self.game.state_to_ind()])

        # Play
        reward, finish = self.game.action(self.game.ind_to_action(action))

        # Update Q Function
        if self.type_of_evaluation=="Q_Learning":
            self.Q_Learning(action, reward)
        if self.type_of_evaluation == "Monte Carlo":
            self.Monte_Carlo(action, reward)

        return reward, finish