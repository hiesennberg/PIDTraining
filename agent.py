from collections import deque
from turtle import pd
import pandas as pd
import numpy as np
import random
import model
import torch
import ExcelManipulation
from importlib import reload
reload(ExcelManipulation)
reload(model)
from model import LinearQNet,Trainer
from helper import statescale,getAction
from ExcelManipulation import PIDSimulator


MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001
PercentChange = 0.3
pRange = 1000
iRange = 5000
dRange = 10
mxState = [1,1,1,1]

class Agent:
    def __init__(self) -> None:
        
        self.memory = deque(maxlen=MAX_MEMORY)
        self.n_trials = 0
        self.gamma = 0.9
        self.epsilon = 0
        self.model = LinearQNet(7,256,4)
        #self.Imodel = LinearQNet(7,256,1)
        self.trainer = Trainer(self.model,lr=LR,gamma=self.gamma)
        #self.iTrainer = Trainer(self.Imodel,lr=LR,gamma=self.gamma)
        self.done = False

    def getState(self,pidsim):
        state = pidsim.getState()
        return state

    def getPID(self,pidsim):
        return pidsim.getPID()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self,iteration):
        if len(self.memory) > BATCH_SIZE:
            mini_sample =  random.sample(self.memory,BATCH_SIZE)

        else:
            mini_sample = self.memory

        state,action,reward,next_state,done = zip(*mini_sample)      

        print("Training Long memery")

        self.trainer.train_step(state,action,reward,next_state,done,iteration)
        #self.iTrainer.train_step(state,action,reward,next_state,done,iteration)

    def train_short_memory(self,state,action,reward,next_state,done,iteration):
        self.trainer.train_step(state,action,reward,next_state,done,iteration)
        #self.iTrainer.train_step(state,action,reward,next_state,done,iteration)

    def get_action_exploit(self, state, maxState, pid):
        state0 = torch.tensor(statescale(state, maxState), dtype=torch.float)
        Prediction = self.model(state0)

        action = torch.argmax(Prediction)
        binary = bin(int(action)).split('b')[1]
        if len(binary) == 1:
            binary = '0' + binary

        pid[0] = pid[0] + pid[0]*PercentChange*(int(binary[0]) - 0.5)
        pid[1] = pid[1] + pid[1]*PercentChange*(int(binary[1]) - 0.5)
        pid[2] = 0  # += PercentChange*(Predition[2].item() - 0.5)
        print(Prediction)
        return pid

    def get_action(self,state,maxState,pid):
        
        self.epsilon = 1500 - self.n_trials
        #pid = state[0:3]
        
        if random.randint(0,1000) < self.epsilon:
            change =random.randint(0,4)
            if change == 0:
                pid[0] = pid[0] - pid[0]*PercentChange
                pid[1] = pid[1] - pid[1]*PercentChange
            
            elif change == 1:
                pid[0] = pid[0] - pid[0]*PercentChange
                pid[1] = pid[1] + pid[1]*PercentChange
            
            elif change == 2:
                pid[0] = pid[0] + pid[0]*PercentChange
                pid[1] = pid[1] - pid[1]*PercentChange

            else:
                pid[0] = pid[0] + pid[0]*PercentChange
                pid[1] = pid[1] + pid[1]*PercentChange
            
            
            
            
            pid[2] = 0

            # if self.n_trials % 2 == 0:
            #     pid[0] = float(random.randint(2,10))
            #     pid[1] = float(random.randint(2,20))
            #     pid[2] = 0
            # ChangePar = random.randint(0,1)
            # n = random.uniform(-1,1)
            # k = pid[ChangePar] 
            # pid[ChangePar] = pid[ChangePar] + n*k
            print("Random : ",pid)

        else:
            state0 = torch.tensor(statescale(state,maxState),dtype=torch.float)
            Prediction = self.model(state0)
            #PredictionI = self.Imodel(state0)
            # incr = torch.argmax(Predition).item()
            # decr = torch.argmin(Predition).item()

            action = torch.argmax(Prediction)
            
            binary = bin(int(action)).split('b')[1]
            if len(binary)==1:
                binary = '0' + binary
            
            pid[0] = pid[0] + pid[0]*PercentChange*(int(binary[0]) - 0.5)
            pid[1] = pid[1] + pid[1]*PercentChange*(int(binary[1]) - 0.5)
            pid[2] = 0  #+= PercentChange*(Predition[2].item() - 0.5)
            print("Prediction : ", Prediction)
         

            # pid[0] = Predition[0].item()*pRange
            # pid[1] = Predition[1].item()*iRange
            # pid[2] = 0
        #print("from Model : ",action, "    ", "Prediction : ", pid[0], "  ;  ",pid[1])

        return pid


def train():
    jitter = []
    maxError = []
    totalError = []
    AbserrorQlist = []
    plist = []
    ilist = []
    dlist = []
    rewardlist = []
    trailErrorList = []

    agent = Agent()
    agent.model.load_state_dict(torch.load(".\models\model_569.pth"))
    agent.n_trials = 1
    sim = PIDSimulator()
    iter = 0
    sim.SetInitialProcessValues()
    sim.saveInitialState()
    # p = [0,1000]
    # i = [0, 5000]
    # d  = [0,1]
    # maxscale = [max(maxError),max(totalError),max(maxError)]
    maxState = agent.getState(sim)
    minstate = [0,0,0,1]

    
    sp = sim.getProcessValues()

    while True:
        
        state_old = agent.getState(sim)
        pid = sim.getPID()
        pid2 = pid.copy()
        Final_Move= agent.get_action(state_old,maxState,pid)

        reward = sim.Simulate(Final_Move)

        state_new = agent.getState(sim)

        maxError.append(state_new[0])
        totalError.append(state_new[1])
        jitter.append(state_new[2])
        AbserrorQlist.append(state_new[3])
        trailErrorList.append(state_new[4])
        plist.append(state_new[5])
        ilist.append(state_new[6])
        # dlist.append(Final_Move[2])
        rewardlist.append(reward)

        maxState = [max(maxError),max(totalError),max(jitter),max(AbserrorQlist),max(trailErrorList),max(plist),max(ilist)]


        maxerror,totalerror,pvjitter,AbserrorQ,trailError,p,i=state_new[0],state_new[1],state_new[2],state_new[3],state_new[4],state_new[5],state_new[6]

        #print("P : {0} ; I : {1} ; D : {2} ; MaxError : {3} ; TotalError : {4} ; Jitter : {5} ; ErrorQ75 : {6}  ; reward : {7} ; TrailError : {8}".format(Final_Move[0],
        #Final_Move[1],Final_Move[2],maxerror,totalerror,pvjitter,AbserrorQ,reward,trailError))

        #EpisodeFinish = True if (maxerror > 0.5*sp or (pvjitter < 3 and AbserrorQ < 0.0001) or iter>50 or trailError > 0.25*sp)  else False

        EpisodeFinish = True if (maxerror > 0.5*sp or  AbserrorQ < 0.001 or iter>50 or trailError > 0.25*sp or Final_Move[1] <=1 or Final_Move[0] <=1 or pvjitter > 25 )  else False



        if maxerror>0.5*sp:
            reward -= 100

        if trailError>0.25*sp:
            reward -= 100
        
        if iter>100:
            reward -= 100
        if Final_Move[0] <=1: # or Final_Move[0] > 500:
            reward -=100
        if Final_Move[1] <=1: # or Final_Move[1] > 500:
            reward -=100
        


        if AbserrorQ < 0.001: # and pvjitter < 3:
            reward += 100

        if pvjitter > 25 : # and pvjitter < 3:
            reward -= 100

        if sp-trailError>0.001:
            reward -= 100


        
        state_new = statescale(state_new,maxState)

       

        action = getAction(Final_Move,pid2)

        print("Action : ", action)

        #train Short memory
        agent.train_short_memory(state_old,action,reward,state_new,EpisodeFinish,iter)

        #remember
        agent.remember(state_old,action,reward,state_new,EpisodeFinish)

        
        iter+=1

        if EpisodeFinish:

            iter = 0

            if sp-trailError>0.001:
                reward -= 100
            



            if pvjitter <= 2:
                print('Jitter suggests trailing error is zero {0}'.format(str(state_old)))

            elif maxerror > 0.5*sp:
                print('maxError high, Episode Ending {0}'.format(str(state_old)))


            if agent.n_trials%2 == 0:
                p = random.randint(5,20)
                i = random.randint(220,300)
                d = 0
            
            else:
                p = random.randint(5,20)
                i = random.randint(220,300)
                d = 0
            sim.SetInitialProcessValues(agent.n_trials+1,agent.n_trials + 1,d)
            sim.saveInitialState()
            agent.n_trials += 1
            agent.train_long_memory(iter)

            
            
            df = pd.DataFrame(list(zip(maxError,totalError,jitter,AbserrorQlist,plist,ilist,dlist,rewardlist)),columns=['MaxError','TotalError','Jitter','AbserrorQlist','p','i','d','reward'])
            #df = pd.DataFrame(list(zip(**agent.memory)))
            
            df.to_csv('Trend')

            print("After Episode {0} MaxError is {1} TotalError is {2} PVJitter is {3}".format(agent.n_trials,state_new[0],state_new[1],state_new[2]))
            agent.model.save(agent.n_trials)
            if iter>10000:
                break





if __name__ == '__main__':
    train()
    