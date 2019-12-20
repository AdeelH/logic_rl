import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt
from multiprocessing import Pool
from functools import partial
import mtl

from rl687.environments.graph import graph1, graph1_tl

# from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
# from rl687.agents.ga import GA

from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.linear_approx import LinearApprox



def run_episode(env, policy, seed, max_steps=2000, timeout_penalty=None, mtl_reward_expr=None):
    np.random.seed(seed*10)
    env.reset()
    g = 0.
    history = []
    for step in range(max_steps):
        if env.isEnd:
            break
        s, r, _ = env.step(policy.samplAction(env.state))
        g += (env.gamma ** step) * r
        history.append(s)

    if mtl_reward_expr is not None:
        g += mtl_reward(mtl_reward_expr, history)

    out = timeout_penalty if timeout_penalty is not None else g
    return out

def run_n_episodes_p(env, policy, p, n=10, theta=None, log_evals=False, max_steps=10, timeout_penalty=None, mtl_reward_expr=None):
    if theta is not None:
        policy.parameters = theta

    fn = partial(run_episode, env, policy, max_steps=max_steps, timeout_penalty=timeout_penalty, mtl_reward_expr=mtl_reward_expr)
    gs = p.starmap(fn, zip(range(n)))

    if log_evals:
        return sum(gs) / n, gs
    return sum(gs) / n

def train_agent(env, policy, bbo_agent, iters=10):
    bbo_agent.reset()
    policy.parameters = bbo_agent.parameters

    evals = [None] * iters
    for i in range(iters):
        _, log = bbo_agent.train(log_evals=True)
        evals[i] = log
    return np.array(evals).flatten()

def train_agent_n_trials(env, policy, bbo_agent, iters=10, trials=1):
    evals = [None] * trials
    for i in range(trials):
        print('trial', i)
        bbo_agent.reset()
        evals[i] = train_agent(env, policy, bbo_agent, iters)
    return np.array(evals).reshape(trials, -1)


def mtl_reward(expr, history):
    phi = mtl.parse(expr)
    data = {
        'p': []
    }
    for i, s in enumerate(history):
        data['p'].append((i, s in [0, 2, 3, 6, 7]))
    r = all(val for _, val in phi(data, time=None))
    return r

def mdp_discrete():
    gr = graph1()
    numStates = gr.numStates
    numActions = 2
    
    env = gr
    numEpisodes, iters, trials = 4, 400, 1
    policy = TabularSoftmax(numStates, numActions)
    np.random.seed(42)
    with Pool(8) as p:
        eval_fn = partial(run_n_episodes_p, env, policy, p, max_steps=200)
        bbo = FCHC(
            theta = policy.parameters,
            sigma = 1, 
            numEpisodes = numEpisodes, 
            evaluationFunction = lambda theta, numEpisodes, **kwargs: eval_fn(theta=theta, n=numEpisodes, **kwargs)
        )
        t0 = perf_counter()
        returns = train_agent_n_trials(env, policy, bbo, iters=iters, trials=trials)
        print(f'elapsed: {perf_counter() - t0}')

    print(policy.probs)

def mdp_discrete_tl():
    gr = graph1_tl()
    numStates = gr.numStates
    numActions = 2
    
    env = gr
    numEpisodes, iters, trials = 4, 400, 1
    policy = TabularSoftmax(numStates, numActions)
    np.random.seed(42)
    with Pool(8) as p:
        eval_fn = partial(run_n_episodes_p, env, policy, p, max_steps=200, mtl_reward_expr='G p')
        bbo = FCHC(
            theta = policy.parameters,
            sigma = 1, 
            numEpisodes = numEpisodes, 
            evaluationFunction = lambda theta, numEpisodes, **kwargs: eval_fn(theta=theta, n=numEpisodes, **kwargs)
        )
        t0 = perf_counter()
        returns = train_agent_n_trials(env, policy, bbo, iters=iters, trials=trials)
        print(f'elapsed: {perf_counter() - t0}')

    print(policy.probs)


def main():
    mdp_discrete_tl()
    mdp_discrete()


if __name__ == "__main__":
    main()
