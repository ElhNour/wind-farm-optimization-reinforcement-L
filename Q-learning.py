import numpy as np

STATE_SPACE = range(-180,181) # theta, which is the relative wind heading (wind heading - wind turbine heading)
ACTION_SPACE = [0,1,2]
NB_ACTIONS=len(ACTION_SPACE)
MAX_STEPS=180


class Q_Learning_Agent:
    def __init__(self,epsilon, rewards,probs):
        self.rewards=rewards
        self.probs_transition=probs
        self.epsilon=epsilon
        self.final_states=[-180,0]
        #Initialization of Q(s,a) for all s in STATE SPACE
        self.Qs_a={}
        action_reward=[0., 0., 0.]
        for s in STATE_SPACE:
            self.Qs_a[s]=action_reward
        
    def train(self,alpha,gamma,nb_episodes):
        
        for i in range(nb_episodes):
            #Initial State
            s=np.random.choice(STATE_SPACE)
            #print('episode {}, initial state {}'.format(i,s))
            done=False
            timestep=-1
            while (not done):
                # get probabilities of all actions from current state
                action_probabilities = np.ones(NB_ACTIONS,
                    dtype = float) * self.epsilon / NB_ACTIONS
                best_action = np.argmax(self.Qs_a[s])
                action_probabilities[best_action] += (1.0 - self.epsilon)
   
                # choose action according to 
                # the probability distribution
                action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p = action_probabilities)
                #print("action: ",action)

                # simulate a transition
                new_s,reward=self.probs_transition[(s,action)]  
                #print("new s {}, r {}:".format(new_s,reward))
                # update Q(s,a)
                #print('Qs_a old: ',self.Qs_a[s])
                self.Qs_a[s][action]+= alpha*(reward+ gamma*np.max(self.Qs_a[new_s])-self.Qs_a[s][action])
                #print('Qs_a new: ',self.Qs_a[s])
                s=new_s
                timestep+=1
                if s in self.final_states or timestep>MAX_STEPS:
                    done=True
    
    def policy(self, state):
        return np.argmax(self.Qs_a[round(state)])
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)