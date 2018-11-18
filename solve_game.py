'''
http://outlace.com/rlpart3.html
- lecun_uniform
https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
'''

import game
from policies import policies
import matplotlib.pyplot as plt
import numpy as np
import interactive_play
import time
from Deep_Q_learning import Deep_Q_learning
from tabular_learning import tabular_learning

def print_Q_values(game, learning):
    print("############# Action #" + str(game.action_counter) + "#############")
    #Qs = tabular_learning.Q_table[:, game.state_to_ind()]
    Qs = learning.predict_q()[0,:,0]
    print(game.dillema_string(Qs))

# Initializations
num_of_iters = 3000000
batch_size = 100
game = game.GAME()
tabular_learning = tabular_learning(game, alpha = 100 / num_of_iters)
deepQ = Deep_Q_learning(game)
policy = policies(eps_decay=2 / num_of_iters)

initial_state = {'Ball': [0, 0], 'Agent': [1, 1, 0], 'Ball_on_Agent': False} #None
game.set_state(initial_state)
game.start_recording()

# Restore checkpoint
# deepQ.saver.restore(deepQ.sess, "algo")

# Main Loop of Training
start_time = time.time()
loss_graph = []
batch_ind = 0
states = np.zeros((batch_size, game.width, game.height, 2))
predictions = np.zeros((batch_size, game.number_of_actions, 1))
loss = -1
for iter in range(0, num_of_iters):
    reward, finish = tabular_learning.tabular_play_iter(policy)

    '''s0 = np.expand_dims(game.state_as_tensor(), axis=0)
    reward, finish, pred = deepQ.play_iter(policy)

    batch_ind = batch_ind + 1
    # im = game.show()
    if batch_ind < batch_size:
        states[batch_ind, :, :, :] = s0
        predictions[batch_ind, :] = pred
        loss = -1
    else:
        loss = deepQ.train_step(states, predictions)
        loss_graph.append(loss)
        batch_ind = 0'''

    status = game.status(reward)
    status = status + " eps: " + str(policy.eps)[0:4]

    if loss != -1:
        status = status + " Loss: " + str(loss)
    if finish:
        status = status + " Game Over!"
        game.start_game(initial_state)

    print("Iter: #" + str(iter) + " " + status)
    #if iter % int(num_of_iters / 15) == 0:
        #plt.figure()
        #plt.imshow(Q_table[:,relevant_states])

# Show results
print("End of Training: {:.2f} secs".format(time.time() - start_time))
plt.figure()
plt.plot(game.actions_per_game)
plt.title("Number of turns per game")
#plt.figure()
#plt.imshow(tabular_learning.Q_table)

game.start_game(initial_state)
fig = plt.figure()
plt.imshow(game.show())
interactive_play.attach_interactive(game, fig, print_Q_values, (game, tabular_learning), initial_state) # deepQ

plt.figure()
plt.plot(loss_graph)
plt.title("Loss")
plt.show()

save_path = deepQ.saver.save(deepQ.sess, "checkpoint/")
print("Model was saved: " + save_path)
deepQ.close_session()

#np.save("algo", game.record_data)
#print("algo.npy was saved!")
#game.stop_recording(generate_video="algo", video_fps=20, save_every_n_games=1 + game.game_counter // 20)

