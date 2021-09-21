from car import Car
from ui import Interface
from circuit import Circuit
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class CarEnvironment(Environment):
    NUM_SENSORS = 15

    def __init__(self, render=True):
        super().__init__()
        
        #circuit = Circuit([(0, 0), (-0.5, 1), (0, 2), (2, 2), (5, 2), (6, 3), (10, 5), (15, 4), (16, 2),(14, 0), (10, -1), (6, 0)], width=0.8)
    
        self.generateCircuit(plot=False)
        
        

        # To render the environment
        self.render = render
        
        
        if render:
            self.ui = Interface(self.circuit, self.car)
            self.ui.show(block=False)

        
        
        
        # Build the possible actions of the environment
        self.Lactions = []
        for turn_step in range(-2, 3, 1):
            for speed_step in range(-1, 2, 1):
                self.Lactions.append((speed_step, turn_step))

        self.count = 0
        
        self.crash_value = 0 #-0.1 # A CHANGER
        
    def generateCircuit(self, n=10, plot=True):
        listePoints = [(0, 0)]
        x, y = 0, 0
        for _ in range(n):
            x += randint(-1, +1)
            y += 1
            listePoints.append((x, y))
            
        for _ in range(n):
            x += 1
            y += randint(-1, +1)
            listePoints.append((x, y))
            
        for _ in range(n):
            x += 1 + randint(-1, +1)
            y -= 1
            listePoints.append((x, y))
            
        for _ in range(n):
            x -= 1
            y += randint(-1, +1)
            listePoints.append((x, y))
        
        
        self.circuit = Circuit(listePoints, width=0.8)
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)
        
        fig, ax = plt.subplots()
        self.circuit.plot(ax)
        plt.show()
        
        assert self.circuit != None
        assert self.car != None


        
    def states(self):
        return dict(type='float', shape=(CarEnvironment.NUM_SENSORS+1,), min_value=-100.0, max_value=1000.0)

    def actions(self):
        return dict(type='int', num_values=15)
    
    #def max_episode_timesteps(self):
    #    return 1000

    def reward(self) -> float:
        """Computes the reward at the present moment"""
        """
        isCrash = self.car.car not in self.circuit
        unit = self.car.speed / self.car.SPEED_UNIT
        return unit ** 1.1 + int(isCrash) * self.crash_value
        """
        
        isCrash = self.car.car not in self.circuit
        return self.circuit.laps + self.circuit.progression + int(isCrash) * self.crash_value 

    def isEnd(self) -> bool:
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        return isCrash or hasStopped

    def reset(self):
        self.count = 0
        self.car.reset()
        self.circuit.reset()
        return tuple(self.current_state)
    
    def execute(self, actions):
        #print("A : ", actions)
        self.count += 1
        #print(i)
        #print(self.actions[i])
        self.car.action(*self.Lactions[actions])

        next_state = self.current_state
        terminal = self.isEnd()
        reward = self.reward()
        
        #print("S : ", next_state)

        if self.render:
            self.ui.update()

        return next_state, terminal, reward
        

    @property
    def current_state(self):
        result = self.car.distances()
        result.append(self.car.speed)
        return tuple(result)

    def step(self, i: int, greedy):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        self.count += 1
        #print(i)
        #print(self.actions[i])
        self.car.action(*self.actions[i])

        state = self.current_state
        isEnd = self.isEnd()
        reward = self.reward()

        if self.render:
            self.ui.update()

        return state, reward, isEnd

    def mayAddTitle(self, title):
        if self.render:
            self.ui.setTitle(title)
    
    
    def train(self, agent, n=1000):
        R = np.zeros(n)
        #for k in progressbar.progressbar(range(n)):
        for k in range(n):
            states = self.reset()
            terminal = False
            cumulReward = 0
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = self.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                cumulReward += reward
            R[k] = cumulReward
            #print(cumulReward)
        plt.plot(np.arange(R.shape[0]), R)
        
        agent.save('/Users/arnaudgardille/Documents/CA_RL/Lab/src/goodAgent', 'dueling_dqn')
        
    def evaluate(self, agent, n=10):
        # Evaluate for 100 episodes
        sum_rewards = 0.0
        for _ in range(n):
            states = self.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = agent.act(
                    states=states, internals=internals,
                    independent=True, deterministic=True
                )
                states, terminal, reward = self.execute(actions=actions)
                sum_rewards += reward
        
        print('Mean episode reward:', sum_rewards / 100)
        

if __name__ == '__main__':
    
    environment = Environment.create(environment=CarEnvironment)
    
    #agent = Agent.create(agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3, max_episode_timesteps=1000,  exploration=0.001, discount=0.9)
    agent = Agent.create(agent='dueling_dqn', environment=environment, memory=10000,  batch_size=10, learning_rate=1e-3, max_episode_timesteps=1000,  discount=0.9)
    
    
    #agent = Agent.load('/Users/arnaudgardille/Documents/CA_RL/Lab/src/goodAgent', 'dueling_dqn')
    """
    agent = Agent.create(
    agent='tensorforce', environment=environment, update=64,
    optimizer=dict(optimizer='adam', learning_rate=1e-3),
    objective='policy_gradient', reward_estimation=dict(horizon=20)
    )
    
    runner = Runner(agent=agent,environment=environment)
    
    runner.run(num_episodes=200)
    
    runner.run(num_episodes=100, evaluation=True)
    
    runner.close()
    
    return Circuit([(0, 0), (-0.5, 1), (0, 2), (2, 2), (5, 2), (6, 3), (10, 5), (15, 4), (16, 2),(14, 0), (10, -1), (6, 0)], width=0.8)
    """
    
    for k in progressbar.progressbar(range(100)):
        environment.generateCircuit()

        environment.train(agent, 1000)
    
    environment.evaluate(agent, 10)
    
    # Close agent and environment
    #agent.close()
    #environment.close()

    
    