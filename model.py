
from fileinput import filename
import os
from sre_parse import State
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import torchvision.transforms as tf
from sklearn.preprocessing import StandardScaler
import numpy as np


class LinearQNet(nn.Module):

    def __init__(self,inputSize,hiddenSize,outputSize) -> None:
        super().__init__()
        self.linear1 = nn.Linear(inputSize,hiddenSize)
        self.linearmid = nn.Linear(hiddenSize,hiddenSize)
        self.linearmid2 = nn.Linear(hiddenSize,hiddenSize)
        self.linear2 = nn.Linear(hiddenSize,outputSize)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linearmid.weight)
        torch.nn.init.xavier_uniform_(self.linearmid2.weight)
        


    def forward(self,x):
        #x= F.normalize(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linearmid(x))
        x = F.relu(self.linearmid2(x))
        x = self.linear2(x)
        

        return x

    def save(self,iter):
        time = datetime.now()
        filename = "model_" + str(iter)  + ".pth"
        filepath = os.path.join(".\models",filename)

        torch.save(self.state_dict(),filepath)


class Trainer:

    def __init__(self,model,lr,gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr = self.lr)
        self.criterion = nn.L1Loss()
        self.StateSC = StandardScaler()
        self.ActionSC = StandardScaler()


    def train_step(self,stateIn,actionIn,rewardIn,next_stateIn,done,iteration):
        
        state = torch.tensor(stateIn,dtype =torch.float)
        next_state = torch.tensor(next_stateIn,dtype=torch.float)
        reward = torch.tensor(rewardIn,dtype=torch.float)
        action = torch.tensor(actionIn,dtype=torch.float)
        

        

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            done = (done,)
        
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            #Q_new = torch.tensor(np.full((1,1),reward[idx]), dtype=torch.float)
            Q_new = reward[idx]
            z=0
            if not done[idx]:
                #Q_new = torch.add(torch.tensor(np.full((1,1),reward[idx]), dtype=torch.float) , torch.mul(self.gamma,self.model(next_state[idx])))
                z = torch.max(self.model(next_state[idx]))
                Q_new = reward[idx] + self.gamma*z

            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            # if reward[idx]>0:
            #     pass
            #     target[idx][torch.argmax(action[idx]).item()] += 10
            # else:
            #     target[idx][torch.argmax(action[idx]).item()] -= 10

            #print("Target = ",target[idx], "Prediction = ", pred[idx])
        
       
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        print("Target = ",target[idx], "Prediction = ", pred[idx],"Loss = ", loss, "reward = ",reward[idx], "Z = ",z)
        loss.backward()
        self.optimizer.step()




