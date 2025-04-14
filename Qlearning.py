import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Environment():

    def __init__(self, grid, move_prob=1.0):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    # def transit_func(self, state, action):
    #     transition_probs = {}
    #     if not self.can_action_at(state):
    #         # Already on the terminal cell.
    #         return transition_probs

    #     opposite_direction = Action(action.value * -1)

    #     for a in self.actions:
    #         prob = 0
    #         if a == action:
    #             prob = self.move_prob
    #         elif a != opposite_direction:
    #             prob = (1 - self.move_prob) / 2

    #         next_state = self._move(state, a)
    #         if next_state not in transition_probs:
    #             transition_probs[next_state] = prob
    #         else:
    #             transition_probs[next_state] += prob

    #     return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0 or -1:
            return True
        else:
            return False

    def can_action(self, state, actions):
        can_actions = []
        # Check whether the agent bumped a block cell.
        if self.grid[state.row -1][state.column] != 9:
            can_actions.append(0)
        if self.grid[state.row +1][state.column] != 9:
            can_actions.append(1)
        if self.grid[state.row][state.column -1] != 9:
            can_actions.append(2)
        if self.grid[state.row][state.column +1] != 9:
            can_actions.append(3)
        return can_actions


    def _move(self, state, action):
        if not self.can_action_at(state):
            print(state.row, state.column)
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == 0:
            next_state.row -= 1
        elif action == 1:
            next_state.row += 1
        elif action == 2:
            next_state.column -= 1
        elif action == 3:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == -1:
            # Get reward! and the game ends.
            reward = 1
            done = True

        return reward, done

    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(1, 1)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        next_state = self._move(state, action)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done


class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions):
        if np.random.random() < self.epsilon:
            return actions[np.random.randint(len(actions))]
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return actions[np.random.randint(len(actions))]

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=25, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print('At Episode {} average reward is {} (+/-{})'.format(episode, mean, std))

        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title('Step History')
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color='g')
            plt.plot(indices, means, 'o-', color='g', label='Rewards for each {} episode'.format(interval))
            plt.legend(loc='best')
            plt.savefig('Step History_{}.png'.format(self.epsilon))
            plt.show()

class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.20):
        super().__init__(epsilon)

    def learn(self, env, episode_count=500, gamma=0.9, learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        actions = list(range(4))
        for e in range(episode_count):
            s = env.reset()
            done = False
            count = 0
            while not done:
                if render:
                    env.render()
                can_actions = env.can_action(s, actions)
                a = self.policy(s, can_actions)
                n_state, reward, done = env.step(a)
                gain = reward + gamma * (max(self.Q[n_state]))
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                count += 1

            else:
                self.log(count)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    grid = [
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 9],
    [9, 0, 0, 9, 0, 9, 0, 0, 9, 9, 0, 9],
    [9, 9, 0, 9, 0, 9, 9, 9, 9, 0, 0, 9],
    [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9],
    [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9],
    [9, 9, 9, 9, 9, 0, 9, 0, 9, 0, 0, 9],
    [9, 0, 0, 0, 0, 0, 9, 0, 9, 0, 0, 9],
    [9, 0, 9, 9, 9, 9, 9, 0, 9, 0, 0, 9],
    [9, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 9],
    [9, 0, 0, 9, 0, 0, 0, 0, 9, 0, -1, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]

    # grid = [
    # [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    # [9, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 9, 0, 0, 9, 0, 0, 9],
    # [9, 9, 0, 9, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 0, 9, 0, 0, 9, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 9, 0, 0, 9, 0, 0, 9],
    # [9, 0, 9, 9, 9, 9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 0, 0, 9, 9, 0, 9, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9],
    # [9, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 9, 9, 9, 9, 9, 0, 9, 9, 9, 0, 0, 0, 9, 0, 9, 9, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 9, 0, 9, 0, 9 ,9 ,0, 9, 9, 9, 9, 9 ,9],
    # [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 9],
    # [9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 9],
    # [9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 9, 0, 0, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 9, 9, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0 ,0 ,0, 0, 0, 0, 0, 0, 9],
    # [9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 9, 9, 9, 9, 9, 9, 9],
    # [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    # [9, 0, 0, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 9, 0 ,0 ,0, 9, 0, 0, 0, -1, 9],
    # [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    # ]


    env = Environment(grid)
    agent = QLearningAgent()
    agent.learn(env, episode_count=500)
    agent.show_reward_log()

if __name__ == '__main__':
    train()