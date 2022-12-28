import numpy as np
import matplotlib.pyplot as plt
from PGAgent import PolicyGradientAgent
import gym
import argparse
from datetime import datetime

parser = argparse.ArgumentParser('')
parser.add_argument('--lr',
                    help='learning rate',
                    type=float,
                    default=.1)
parser.add_argument('--N',
                    help='Number of updates',
                    type=int,
                    default=2000)
parser.add_argument('--b',
                    help='batch size',
                    type=int,
                    default=1)
parser.add_argument('--seed',
                    help='random seed',
                    type=int,
                    default=1)
parser.add_argument('--RTG',help='reward to go', dest='RTG', action='store_true')
parser.add_argument('--baseline',help='use baseline', dest='baseline', action='store_true')



args = parser.parse_args()
np.random.seed(args.seed)

env = gym.make('CartPole-v1')

print('Enviourment: CartPole-v1 \nNumber of actions: ' ,env.action_space.n,'\nDimension of state space: ',np.prod(env.observation_space.shape))
def run_episode(env, agent):
    state = env.reset()
    state = state[0]
    rewards = []
    terminal = False
    decay = agent.get_decay()
    current_decay = 1
    dW, db = np.zeros_like(agent.W), np.zeros_like(agent.b)
    while not terminal:
        action = agent.get_action(state)
        state, reward, terminal, _, _ = env.step(action)
        rewards.append(current_decay * reward)
        current_decay = current_decay * decay
        temp_dW, temp_db = agent.grad_log_prob(state, action)
        dW, db = dW + temp_dW, db + temp_db
    return dW, db, sum(rewards), len(rewards)


def train(env, agent, args):
    all_rewards = []
    for i in range(args.N):
        dW_list, db_list = [], []
        rewards_in_batch = []
        for j in range(args.b):
            dW, db, episode_reward, episode_len = run_episode(env, agent)
            rewards_in_batch.append(episode_reward)
            dW_list.append(dW)
            db_list.append(db)

        all_rewards.extend(rewards_in_batch)
        n = len(db_list)
        gradient_w, gradient_b = np.zeros_like(agent.W), np.zeros_like(agent.b)
        if args.RTG:
            for dW, db, index in zip(dW_list, db_list, range(n)):
                gradient_w += dW * sum(rewards_in_batch[index:n])
                gradient_b += db * sum(rewards_in_batch[index:n])
        else:
            baseline_factor = 0
            if args.baseline and len(all_rewards) >= 10:
                baseline_factor = sum(all_rewards[-10:])
            sum_rewards = sum(rewards_in_batch) - baseline_factor * n
            for dW, db in zip(dW_list, db_list):
                gradient_w += dW * sum_rewards
                gradient_b += db * sum_rewards

        agent.update_weights(gradient_w, gradient_b)

        if i%100 == 25:
            temp = np.array(all_rewards[i - 25:i])
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%H:%M:%S")
            print('{}: [{}-{}] reward {:.1f}{}{:.1f}'.format(timestampStr,i-25,i,np.mean(temp),u"\u00B1",np.std(temp)/np.sqrt(25)))
    return agent, all_rewards

def test(env, agent):
    rewards = []
    print('_________________________')
    print('Running 500 test episodes....')
    for i in range(500):
        _,_,r,counter = run_episode(env,agent)
        rewards.append(r)
    rewards = np.array(rewards)
    print('Test reward {:.1f}{}{:.1f}'.format(np.mean(rewards),u"\u00B1",np.std(rewards)/np.sqrt(500.)))
    return agent, rewards


agent = PolicyGradientAgent(env)
agent, rewards = train(env,agent,args)
print('Average training rewards: ',np.mean(np.array(rewards)))
test(env,agent)
plt.plot(np.cumsum(rewards))
plt.show()