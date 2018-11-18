import tensorflow as tf
import numpy as np

class Deep_Q_learning:
    def __init__(self, game):
        self.game = game
        self.gamma = 0.9
        self.alpha = 0.01

        decay_steps = int(2 * 10)
        self.global_step = tf.train.get_or_create_global_step()  # get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=0.05, global_step=self.global_step,
                                             decay_steps=decay_steps, decay_rate=0.999,
                                             staircase=True)

        self.update = tf.placeholder(shape=[None, game.number_of_actions, 1], dtype=tf.float32)
        self.game_state = tf.placeholder(shape=[None, game.width, game.height, 2], dtype=tf.float32)

        #weights1 = tf.get_variable('weights', shape=[3, 3, 2, 4])
        #biases1 = tf.get_variable('biases', shape=[4])
        #self.layer1 = tf.nn.conv2d(self.game_state, weights1, strides=[1, 1, 1, 1], padding='SAME')
        #self.bias = tf.nn.bias_add(self.layer1, biases1)

        weights1 = tf.get_variable('weights', shape=[game.width * game.height * 2, 8])
        biases1 = tf.get_variable('biases', shape=[8])
        self.bias = tf.nn.xw_plus_b(tf.reshape(self.game_state, [-1, game.width * game.height * 2]), weights1, biases1)

        relu = tf.nn.leaky_relu(self.bias)
        relu = tf.reshape(relu, [-1, relu.shape[1]]) #*relu.shape[2]*relu.shape[3]

        weights2 = tf.get_variable('weights2', shape=[relu.shape[1]._value, game.number_of_actions])
        biases2 = tf.get_variable('biases2', [game.number_of_actions])
        self.q = tf.nn.xw_plus_b(relu, weights2, biases2) #tf.transpose(
        self.q = tf.expand_dims(self.q, 2)

        self.loss = tf.losses.mean_squared_error(labels=self.update, predictions=self.q)
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.update, logits=self.q))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr) # AdamOptimizer
        self.training_op = optimizer.minimize(self.loss, name="training_op")

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def predict_q(self):
        s0 = np.expand_dims(self.game.state_as_tensor(), axis=0)
        feed_dict = {self.game_state: s0}
        q = self.sess.run(self.q, feed_dict=feed_dict)
        return q

    def play_iter(self, policy):
        # 1) Net forward from current state
        q1 = self.predict_q()

        # 2) Select Best Action and play
        action = policy.select_action(q1[0])
        reward, finish = self.game.action(self.game.ind_to_action(action))

        # 3) Net forward from new state
        q2 = self.predict_q()
        max_q2 = np.max(q2)

        # 4) Calculate prediction vector
        corr = np.zeros(q1.shape)
        corr[0, action, 0] = (reward + self.gamma * max_q2) # self.alpha *
        q1[0, action, 0] = 0
        if finish:
            corr[0, action, 0] = reward
        pred = np.transpose(q1 + corr)

        return reward, finish, pred

    def train_step(self, states, predictions):
        # Training step using the prediction vectors
        feed_dict = {self.game_state: states, self.update: predictions}
        _, loss = self.sess.run([self.training_op, self.loss], feed_dict=feed_dict)
        return loss

    def close_session(self):
        self.sess.close()