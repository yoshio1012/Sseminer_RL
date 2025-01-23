import numpy as np

gamma=0.9
theta=1e-6
num_state=5
goal_state=4

V=np.zeros(num_state)


def reward(s,a):
    if s==goal_state-1 and a=="right":
        return 1
    if s==goal_state and a=="right":
        return 1
    else:
        return -1
    
def next_state(s,a):
    if a=="left" and s>0:
        return s-1
    if a=="right" and s<goal_state:
        return s+1
    else:
        return s

def value_iteration():
    while True:
        delta=0
        for spot in range(num_state):
            action_values=[]
            for action in ["left","right"]:
                s_next=next_state(spot,action)
                r=reward(spot,action)
                action_values.append(r+gamma*V[s_next])
            ave_value=sum(action_values)/len(action_values)
            delta=max(delta,abs(ave_value-V[spot]))
            V[spot] = ave_value

        if delta < theta:
            break
        print(V)
        input("ENTER")
    return V

if __name__ == "__main__":
    optimal_values = value_iteration()
    print("value_function:")
    print(optimal_values) 
#方策評価



