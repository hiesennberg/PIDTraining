


def statescale(val,mx,mn=[0,0,0,1,0,0,0]):

    #return [(s-mn[idx])/mx[idx] for idx,s in enumerate(val)]
    return [(s-mn[idx])/mx[idx] for idx,s in enumerate(val)]
    #return (val-mn)/mx

def getAction(FinalMove,pid):
    
    action = [0,0,0,0]
    action[0] = 1 if ((FinalMove[0] - pid [0] < 0) and  (FinalMove[1] - pid [1] < 0)) else 0
    action[1] = 1 if ((FinalMove[0] - pid [0] < 0) and  (FinalMove[1] - pid [1] > 0)) else 0
    action[2] = 1 if ((FinalMove[0] - pid [0] > 0) and  (FinalMove[1] - pid [1] < 0)) else 0
    action[3] = 1 if ((FinalMove[0] - pid [0] > 0) and  (FinalMove[1] - pid [1] > 0)) else 0

    #print("******************",FinalMove,pid,action)

    return action
