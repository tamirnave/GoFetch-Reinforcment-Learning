import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import copy
import imageio
from sklearn.model_selection import ParameterGrid

class GAME:
    def __init__(self):
        # Game Parameters
        self.width = 2
        self.height = 2
        self.angle = 45 # [deg]
        self.target_area = [(self.width // 2 - 1,self.height - 1),(self.width // 2,self.height - 1)]

        self.state = {'Ball': [0, 0], 'Agent': [0, 0, 0], 'Ball_on_Agent': False}
        self.prev_state = {'Ball': [0, 0], 'Agent': [0, 0, 0], 'Ball_on_Agent': False}
        states = {'ball_x':range(0,self.width), 'ball_y':range(0,self.height), 'agent_x':range(0,self.width), 'agent_y':range(0,self.height), 'agent_ang':range(0,360,self.angle), 'Ball_on_Agent':[False, True]}
        self.list_of_all_states = list(ParameterGrid(states))

        # Graphics Inits
        tmp = plt.imread("agent.png")[:,:,0:3] * 255
        self.square_to_pixels = int((1 + math.sqrt(2) / 2) * max(tmp.shape[0], tmp.shape[1]))
        self.agent_pic = np.zeros((self.square_to_pixels, self.square_to_pixels, 3))
        self.agent_pic[int((self.square_to_pixels - tmp.shape[0]) / 2):int((self.square_to_pixels + tmp.shape[0]) / 2), int((self.square_to_pixels - tmp.shape[1]) / 2):int((self.square_to_pixels + tmp.shape[1]) / 2), :] = tmp
        self.ball_pic = 255 * plt.imread("ball.png")[:, :, 0:3]

        self.number_of_actions = 5
        self.recording = False
        self.game_counter = 0
        self.actions_per_game = []

        self.start_game()

    def state_as_tensor(self):
        # Represent the current state as a tensor in the board dimensions with depth 2
        # Channel 0 represents where and in which attitude (rotation) the agent is
        # Channel 1 represents where the ball is and wether it is hold by the agent or not

        m = np.zeros((self.width, self.height, 2))
        #  num_of_agent_attitudes = 360 // self.angle
        agent_attitude = self.state['Agent'][2] // self.angle
        m[self.state['Agent'][0], self.state['Agent'][1], 0] = agent_attitude
        # if agent hold ball
        if self.state['Ball_on_Agent']:
            m[self.state['Ball'][0], self.state['Ball'][1], 1] = 2
        # if agent doesn't hold ball
        else:
            m[self.state['Ball'][0], self.state['Ball'][1], 1] = 1


        ''' if agent and ball in same spot
        if self.state['Ball'][0] == self.state['Agent'][0] and self.state['Ball'][1] == self.state['Agent'][1]:
            # if agent hold ball
            if self.state['Ball_on_Agent']:
                m[self.state['Agent'][0], self.state['Agent'][1]] = 4
            # if agent doesn't hold ball
            else:
                m[self.state['Agent'][0], self.state['Agent'][1]] = 3
        else:
            if self.state['Ball_on_Agent']:
                print("Error - if ball and agent are not in same coordinates, agent cannot hold ball!")
            else:
                m[self.state['Ball'][0], self.state['Ball'][1]] = 2
                m[self.state['Agent'][0], self.state['Agent'][1]] = agent_attitude'''

        return m

    def set_state(self,state):
        if state is not None:
            self.state['Ball'][0] = state['Ball'][0]
            self.state['Ball'][1] = state['Ball'][1]
            self.state['Agent'][0] = state['Agent'][0]
            self.state['Agent'][1] = state['Agent'][1]
            self.state['Agent'][2] = state['Agent'][2]
            self.state['Ball_on_Agent'] = state['Ball_on_Agent']

    def start_game(self, state=None):
        if hasattr(self,'action_counter'):
            self.actions_per_game.append(self.action_counter)
        self.game_counter = self.game_counter + 1
        self.action_counter = 0
        if state==None:
            self.Toss_Start_State()
        else:
            self.set_state(state)

    def status(self, reward):
        return "Action #" + str(self.action_counter) + " Game #" + str(self.game_counter) + " Reward: " + str(reward)

    def start_recording(self):
        self.recording = True
        self.record_data = []

    def animate(self, i, save_every_n_games):
        reward, finish = self.action(self.record_data[i][1])
        if finish:
            if isinstance(self.record_data[0][0], str):
                self.start_game(self.ind_to_state(int(self.record_data[i + 1][0])))
            else:
                self.start_game(self.ind_to_state(self.record_data[i + 1][0]))

        if self.game_counter % save_every_n_games == 0:
            I = self.show()
            pad_size = 16 - (I.shape[0] % 16)
            if pad_size % 2 == 0:
                pad_img = cv2.copyMakeBorder(I, pad_size//2, pad_size//2, pad_size//2, pad_size//2, cv2.BORDER_CONSTANT)
            else:
                pad_img = cv2.copyMakeBorder(I, pad_size//2, pad_size//2 + 1, pad_size//2, pad_size//2 + 1, cv2.BORDER_CONSTANT)
            font_size = pad_img.shape[0] / 700
            cv2.putText(pad_img, self.status(reward), (10, int(font_size * 40)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255))
            # TODO: For now it doesn't add to the video the first frame of the new game
            if finish:
                cv2.putText(pad_img,"Game Over", (10, int(font_size * 80)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255))
            return pad_img
        else:
            return None

    def stop_recording(self, save_to_file=None, generate_video=None, video_fps=40, save_every_n_games=10):
        self.recording = False

        if generate_video != None:
            self.game_counter = 0
            if isinstance(self.record_data[0][0], str):
                self.start_game(self.ind_to_state(int(self.record_data[0][0])))
            else:
                self.start_game(self.ind_to_state(self.record_data[0][0]))
            frames = []
            for i in range(len(self.record_data)):
                tmp = self.animate(i, save_every_n_games)
                if tmp is not None:
                    frames.append(tmp)
            imageio.mimsave(generate_video + '.avi', frames, fps=video_fps) #[self.animate(i, save_every_n_games) for i in range(len(self.record_data))]
            print(generate_video + '.avi' + " was saved!")

    def total_num_of_states(self):
        return len(self.list_of_all_states) #self.width * self.height * self.width * self.height * (360//self.angle) * 2

    def Toss_Start_State(self):
        self.state['Ball'] = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.state['Agent'] = [random.randint(0, self.width - 1), random.randint(0, self.height - 1), self.angle * random.randint(0, 360 / self.angle - 1)]
        self.state['Ball_on_Agent'] = False

    def limitXY(self,x,y):
        if x > self.width - 1:
            x = self.width - 1
        if x < 0:
            x = 0
        if y > self.height - 1:
            y = self.height - 1
        if y < 0:
            y = 0
        return x, y

    def calculate_reward(self, end_state):
        reward = -1

        if not self.prev_state['Ball_on_Agent'] and self.state['Ball_on_Agent']:
            reward = 5

        if end_state:
            reward = 20

        return reward

    def action(self, action):
        self.prev_state = copy.deepcopy(self.state)
        if action == "right":
            self.state['Agent'][2] = (self.state['Agent'][2] - self.angle) % 360

        if action == "left":
            self.state['Agent'][2] = (self.state['Agent'][2] + self.angle) % 360

        if action == "up":
            x = round(self.state['Agent'][0] - math.sin(self.state['Agent'][2] * math.pi/180))
            y = round(self.state['Agent'][1] - math.cos(self.state['Agent'][2] * math.pi/180))
            x,y = self.limitXY(x,y)
            self.state['Agent'][0] = x
            self.state['Agent'][1] = y

        if action == "down":
            x = round(self.state['Agent'][0] + math.sin(self.state['Agent'][2] * math.pi/180))
            y = round(self.state['Agent'][1] + math.cos(self.state['Agent'][2] * math.pi/180))
            x, y = self.limitXY(x, y)
            self.state['Agent'][0] = x
            self.state['Agent'][1] = y

        if self.state['Ball_on_Agent'] == True:
            self.state['Ball'][0] = self.state['Agent'][0]
            self.state['Ball'][1] = self.state['Agent'][1]

        if self.state['Agent'][0] == self.state['Ball'][0] and self.state['Agent'][1] == self.state['Ball'][1]:
            agent_on_ball = True
        else:
            agent_on_ball = False

        if action == "p" and agent_on_ball:
            self.state['Ball_on_Agent'] = True

        agent_on_target = False
        for k in self.target_area:
            if self.state['Agent'][0] == k[0] and self.state['Agent'][1] == k[1]:
                agent_on_target = True

        end_state = False
        if self.state['Ball_on_Agent'] == True and agent_on_target:
            end_state = True

        self.action_counter = self.action_counter + 1
        if self.recording:
            self.record_data.append((self.state_to_ind(self.prev_state), action))

        return self.calculate_reward(end_state), end_state

    def show(self):
        # Game Board
        I = np.zeros((self.height * self.square_to_pixels , self.width * self.square_to_pixels, 3))

        # Target Area
        for k in self.target_area:
            I[k[1] * self.square_to_pixels:(k[1] + 1) * self.square_to_pixels, k[0] * self.square_to_pixels:(k[0] + 1) * self.square_to_pixels, :] = [255, 0, 0]

        # Ball
        x = int(self.state['Ball'][0] * self.square_to_pixels + self.square_to_pixels / 2 - self.ball_pic.shape[0] / 2)
        y = int(self.state['Ball'][1] * self.square_to_pixels + self.square_to_pixels / 2 - self.ball_pic.shape[1] / 2)
        I[y:y + self.ball_pic.shape[1], x:x + self.ball_pic.shape[0], :] = self.ball_pic

        # Agent
        rotation_matrix = cv2.getRotationMatrix2D((self.square_to_pixels / 2, self.square_to_pixels / 2), self.state['Agent'][2], 1.0)
        tmp = cv2.warpAffine(self.agent_pic, rotation_matrix, (self.square_to_pixels, self.square_to_pixels))
        x = int(self.state['Agent'][0] * self.square_to_pixels) # + self.square_to_pixels / 2 - tmp.shape[0] / 2)
        y = int(self.state['Agent'][1] * self.square_to_pixels) # + self.square_to_pixels / 2 - tmp.shape[1] / 2)
        I[y:y + tmp.shape[1], x:x + tmp.shape[0], :] = tmp

        # Grid
        for x in range(0, self.width * self.square_to_pixels, self.square_to_pixels):
            I[0:, x, :] = [0, 0, 255]
        for y in range(0, self.height * self.square_to_pixels, self.square_to_pixels):
            I[y, 0:, :] = [0, 0, 255]

        return I.astype('uint8')

    def state_to_ind(self, state=None):
        if state==None:
            state=self.state

        ind = self.list_of_all_states.index({'Ball_on_Agent': state['Ball_on_Agent'], 'agent_y': state['Agent'][1], 'agent_ang': state['Agent'][2] % 360, 'agent_x': state['Agent'][0], 'ball_x': state['Ball'][0], 'ball_y': state['Ball'][1]})

        '''angles_states = 360 // self.angle
        d0 = int(self.state["Ball_on_Agent"])
        d1 = ((self.state["Agent"][2] % 360) // self.angle)
        d2 = self.state["Agent"][1]
        d3 = self.state["Agent"][0]
        d4 = self.state["Ball"][1]
        d5 = self.state["Ball"][0]

        p0 = 1
        p1 = 2 * p0
        p2 = angles_states * p1
        p3 = self.height * p2
        p4 = self.width * p3
        p5 = self.height * p4
        ind = d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3 + d4 * p4 + d5 * p5'''
        return ind

    def ind_to_state(self, ind):
        '''angles_states = 360 // self.angle
        p0 = 1
        p1 = 2 * p0
        p2 = angles_states * p1
        p3 = self.height * p2
        p4 = self.width * p3
        p5 = self.height * p4

        d0 = ind % p0
        d1 = (ind-d0 * p0) % p1
        d2 = (ind - d0 * p0 - d1 * p1) % p2
        d3 = (ind - d0 * p0 - d1 * p1 - d2 * p2) % p3
        d4 = (ind - d0 * p0 - d1 * p1 - d2 * p2 - d3 * p3) % p4
        d5 = (ind - d0 * p0 - d1 * p1 - d2 * p2 - d3 * p3 - d4 * p4) % p5
        state = {'Ball': [d5, d4], 'Agent': [d3, d2, d1 * self.angle], 'Ball_on_Agent': bool(d0)}'''

        tmp = self.list_of_all_states[ind]
        state = {'Ball': [tmp['ball_x'], tmp['ball_y']], 'Agent': [tmp['agent_x'], tmp['agent_y'], tmp['agent_ang']], 'Ball_on_Agent': tmp['Ball_on_Agent']}
        return state

    def dillema_string(self, Q):
        s = ""
        ind = Q.argmax()
        for k in range(self.number_of_actions):
            tmp = self.ind_to_action(k) + " : {:.8f}".format(Q[k])
            if k==ind:
                s = s +'\033[1m' + tmp + '\033[0m' + "\n"
            else:
                s = s + tmp + "\n"
        return s[:-1]

    def ind_to_action(self, ind):
        if ind == 0:
            return "right"
        if ind == 1:
            return "left"
        if ind == 2:
            return "up"
        if ind == 3:
            return "down"
        if ind == 4:
            return "p"

    def generate_states_from_locations(self, ball_locs, agent_locs):
        relevant_states = []
        for k in range(agent_locs.shape[0]):
            for ang in range(0,360,45):
                relevant_states.append(game.state_to_ind({'Ball': ball_locs, 'Agent': [agent_locs[k,0], agent_locs[k,1], ang], 'Ball_on_Agent': False}))
                relevant_states.append(game.state_to_ind({'Ball': ball_locs, 'Agent': [agent_locs[k,0], agent_locs[k,1], ang], 'Ball_on_Agent': True}))

        return relevant_states

if __name__ == "__main__":
    game=GAME()

    plt.imshow(game.show())

    game.action("Right")
    plt.figure()
    plt.imshow(game.show())

    game.action("Up")
    plt.figure()
    plt.imshow(game.show())
    plt.show()

    print("A")