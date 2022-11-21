import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp

MAX_STEPS=180

class REINFORCE_agent:
    def __init__(self):
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.Input(shape=(1,)))
        model.add(keras.layers.Dense(10,activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(5, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=init)) # action probabilities
        self.model=model
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 1
    
    def policy(self, rel_wind_heading) -> int:
        '''Using TensorFlow probability library, we turn our probabilities into a distribution.
        Then we sample an action from that distribution.'''
        prob = self.model.predict(np.array([rel_wind_heading]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy())
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class REINFORCE:
    def __init__(self,probs,STATE_SPACE,r_agent):

        self.agent=r_agent
        self.state_space=STATE_SPACE
        self.probs_transition=probs
        self.final_states=[-180,0]

    def a_loss(self,prob, action, reward): 
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss 

    def policy_gradient(self, states, rewards, actions):
        '''
            This function takes the list of states, actions, and rewards which represent a set of trajectories.
            It calculates the expected cumulative reward for each state.
            Then, calculates the gradient of loss and apply optimizer.
        '''
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.agent.gamma*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()  

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.agent.model.predict(np.array([state]))
                loss = self.a_loss(p, action, reward)
            grads = tape.gradient(loss[0], self.agent.model.trainable_variables)
            self.agent.opt.apply_gradients(zip(grads, self.agent.model.trainable_variables))
    
    def train(self,nb_episodes=500):
        '''
            We have maintained three lists that keep records of the state, reward, action.
            The model is trained after every episode.
        '''
        for episode in range(nb_episodes):
            # initial state
            timestep=-1
            s=np.random.choice(self.state_space)
            done = False
            total_reward = 0
            rewards = []
            states = []
            actions = []
            # simulate a trajectory
            while not done:
                timestep+=1
                action = self.agent.policy(s) #sample an action from the agent's policy
                next_state, reward = self.probs_transition[(s,action)] # make a transition
                # record the trajectory
                rewards.append(reward)
                states.append(s)
                actions.append(action)
                s = next_state
                total_reward += reward
                if s in self.final_states or timestep>MAX_STEPS:
                    done=True
                if done:
                    # calculate the gradient of loss and apply optimizer
                    self.policy_gradient(states, rewards, actions)
                    print("total reward for episode {} are {}".format(episode,total_reward))
        
        return self.agent
