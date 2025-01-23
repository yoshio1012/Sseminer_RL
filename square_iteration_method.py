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

# 方策評価
def policy_evaluation(policy):
    V = np.zeros(num_state)  # 価値関数の初期化
    while True:
        new_V=np.zeros(num_state)
        delta = 0
        for s in range(num_state):
            # 現在の方策に従った価値を計算
            a = policy[s]
            s_next = next_state(s, a)
            r = reward(s, a)
            new_value = r + gamma * V[s_next]
            delta = max(delta, abs(new_value - V[s]))
            new_V[s] = new_value
        V=new_V
        if delta < theta:  # 収束判定
            break
    return V

# 方策改善
def policy_improvement(V):
    policy = ["right"] * num_state  # 初期方策（すべて右に進む）
    for s in range(num_state):
        # 左と右の行動を比較して最適な方策を選択
        action_values = {}
        for a in ["left", "right"]:
            s_next = next_state(s, a)
            r = reward(s, a)
            action_values[a] = r + gamma * V[s_next]
        policy[s] = max(action_values, key=action_values.get)  # 最大価値の行動を選択
    return policy

# 方策反復法
def policy_iteration():
    # 初期方策（すべて右に進む）
    policy = ["left"] * num_state
    while True:
        # 方策評価
        V = policy_evaluation(policy)
        # 方策改善
        new_policy = policy_improvement(V)
        # 方策が変わらなければ終了
        if new_policy == policy:
            break
        policy = new_policy
        print(policy)
        print(V)
        input("ENTER")
    
    return policy, V

# 実行
if __name__ == "__main__":
    optimal_policy, optimal_values = policy_iteration()
    print("最適方策:")
    print(optimal_policy)
    print("最適価値関数:")
    print(optimal_values)