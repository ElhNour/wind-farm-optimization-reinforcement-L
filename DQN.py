import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import random

MAX_STEPS=180

class DQN_agent:
    def __init__(self):
        '''
            The DQN agent maps to each state, the Q-values of the different
            actions from that state
        '''
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.Input(shape=(1,)))
        model.add(keras.layers.Dense(10, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(5, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(3, activation='linear', kernel_initializer=init))
        model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        self.model=model

    def policy(self, rel_wind_heading) -> int:
        return np.argmax(self.model.predict(np.array([rel_wind_heading])))
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class DQN:
    def __init__(self,probs,STATE_SPACE):
        self.probs_transition=probs
        self.final_states=[-180,0]
        self.state_space=STATE_SPACE

    def learn_qs(self,replay_memory, model, target_model, done):
        print("learn_qs")
        learning_rate = 0.9 # Learning rate
        discount_factor = 0.62

        MIN_REPLAY_SIZE = 50
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 40
        mini_batch = random.sample(replay_memory, batch_size)
        s_t = np.array([transition[0] for transition in mini_batch])
        print('s_t',s_t)
        qs_t= np.array([model.predict(np.array([s])) for s in s_t])
        st_1 = np.array([transition[3] for transition in mini_batch])
        qs_t_1= np.array([target_model.predict(np.array([s])) for s in st_1 ])

        In = []
        Out = []
        for index, (obs, action, reward, new_obs, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(qs_t_1[index])
            else:
                max_future_q = reward

            current_qs = qs_t[index][0]
            print('current qs, action',current_qs,action)
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            In.append(obs)
            Out.append(current_qs)
        model.fit(np.array(In), np.array(Out), batch_size=batch_size, verbose=0, shuffle=True)

    def train(self,nb_episodes=500):
        epsilon = 1 # every step is random at start
        max_epsilon = 1 # we allow exploration at most 100% of the time
        min_epsilon = 0.01 # at a minimum, we'll always explore 1% of the time
        decay = 0.01 # regularization technique that is used in ML to reduce the complexity of a model and prevent overfitting

        # initialize the target and Q-Learning networks
        # Update the Q-Learning model every 5 steps
        q_model = DQN_agent()
        # Target Model (updated every 100 steps)
        target_model = DQN_agent()
        target_model.model.set_weights(q_model.model.get_weights())

        replay_memory = deque(maxlen=250)

        update_target_model_steps = 0

        for episode in range(nb_episodes):
            timestep=-1
            total_training_rewards = 0
            # initial State
            s=np.random.choice(self.state_space)
            print('episode {}, initial state {}'.format(episode,s))
            done = False
            while not done:
                timestep+=1
                update_target_model_steps += 1

                random_number = np.random.rand()
                # explore using the Epsilon Greedy Exploration Strategy
                if random_number <= epsilon:
                    # explore
                    action = random.randrange(3)
                else:
                    # exploit best known action
                    predicted = q_model.model.predict(np.array([s]))
                    action = np.argmax(predicted)
                print("action: ",action)
                new_obs, reward=self.probs_transition[(s,action)]
                print("new s {}, r {}:".format(new_obs,reward))
                if new_obs in self.final_states or timestep>MAX_STEPS:
                    done=True
                replay_memory.append([s, action, reward, new_obs, done])

                # update the q-network using the Bellman Equation
                if update_target_model_steps % 5 == 0 or done:
                    self.learn_qs(replay_memory, q_model.model, target_model.model,done)

                s = new_obs
                total_training_rewards += reward

                if done:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, timestep, reward))
                    update_target_model_steps += 1

                    if update_target_model_steps >= 50:
                        print('Copying main network weights to the target network weights')
                        target_model.model.set_weights(q_model.model.get_weights())
                        update_target_model_steps = 0
                    break

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        return target_model