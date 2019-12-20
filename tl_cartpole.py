import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt
from multiprocessing import Pool
from functools import partial
import mtl

from rl687.environments.cartpole import Cartpole
from rl687.viz import viz_cartpole

from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA

from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.linear_approx import LinearApprox



def run_episode(env, policy, seed, max_steps=10000, mtl_reward_expr=None):
    np.random.seed(seed*10)
    env.reset(hard_reset=True, random=True)
    g = 0.
    history = []
    for step in range(max_steps):
        if env.isEnd:
            break
        s, r, _ = env.step(policy.samplAction(env.state))
        g += (env.gamma ** step) * r
        history.append(s)

    if mtl_reward_expr is not None:
        g = g * 2 if mtl_reward(mtl_reward_expr, history) and (env._t >= env._t_lim) else g
        # return mtl_reward(mtl_reward_expr, history)

    return g

def run_episode_v(env, policy, seed, mtl_reward_expr, max_steps=10000):
    np.random.seed(seed*10)
    history = []
    for step in range(max_steps):
        if env.isEnd:
            break
        s, _, _ = env.step(policy.samplAction(env.state))
        history.append(s)
    return mtl_reward(mtl_reward_expr, history)

def run_n_episodes_p(env, policy, p, n=10, theta=None, log_evals=False, max_steps=10, mtl_reward_expr=None):
    if theta is not None:
        policy.parameters = theta

    fn = partial(run_episode, env, policy, max_steps=max_steps, mtl_reward_expr=mtl_reward_expr)
    gs = p.starmap(fn, zip(range(n)))

    if log_evals:
        return sum(gs) / n, gs
    return sum(gs) / n

def train_agent(env, policy, bbo_agent, iters=10, verbose=True):
    bbo_agent.reset()
    policy.parameters = bbo_agent.parameters

    evals = [None] * iters
    for i in range(iters):
        if verbose and (i + 1) % 10 == 0:
            print('iter', i)
        _, log = bbo_agent.train(log_evals=True)
        evals[i] = log
    return np.array(evals).flatten()

def train_agent_n_trials(env, policy, bbo_agent, iters=10, trials=1):
    evals = [None] * trials
    for i in range(trials):
        print('trial', i)
        bbo_agent.reset()
        evals[i] = train_agent(env, policy, bbo_agent, iters)

        # env.reset()
        # viz_cartpole(env, policy, delay_ms=10)
    return np.array(evals).reshape(trials, -1)

def mtl_reward(expr, history, p=None, q=None):
    phi = mtl.parse(expr)
    data = {
        'p': [],
        'q': []
    }
    for i, s in enumerate(history):
        data['p'].append((i * .02, float(s[0]) <= -.5))
        data['q'].append((i * .02, float(s[0]) >=  .5))
    r = sum(val for _, val in phi(data, time=None))
    return phi(data)
    # history = np.array(history)[:, 0]
    # print(((history < -.5) | (history > .5)).any())
    # return (history < -.25).any() and (history > .25).any()
    # return (history > -.5).all() and (history < .5).all()


def verif(env, policy, mtl_spec, inputsToTry=10):
    inputsTried = np.empty((inputsToTry, env.state_dim))
    results = np.empty(inputsToTry).astype(np.bool)
    for i in range(inputsToTry):
        s = env.generate_random_state()
        inputsTried[i] = s
        env.reset(state=s)
        results[i] = run_episode_v(env, policy, i, mtl_spec)
    return inputsTried[results], inputsTried[~results]


def cartpole_verify():
    env = Cartpole()
    numActions = 2
    
    numEpisodes, iters, trials = 1, 50, 1
    policy = LinearApprox(numActions)
    np.random.seed(42)

    populationSize = 8
    dim = policy.parameters.shape[0]
    with Pool(8) as p:
        eval_fn = partial(run_n_episodes_p, env, policy, p, max_steps=10000)
        bbo = GA(
            populationSize = populationSize, 
            numEpisodes = numEpisodes, 
            numElite = 2, 
            K_p = 2, 
            alpha = 1., 
            initPopulationFunction = lambda populationSize: np.random.randn(populationSize, dim), 
            evaluationFunction = lambda theta, numEpisodes, **kwargs: eval_fn(theta=theta, n=numEpisodes, **kwargs)
        )
        t0 = perf_counter()
        returns = train_agent_n_trials(env, policy, bbo, iters=iters, trials=trials)
        print(f'elapsed: {perf_counter() - t0}')

    pos, neg = verif(env, policy, '((F p) & (F q))', inputsToTry=1000, p=lambda x: x < -0.5, q=lambda x: x > 0.5)
    print(len(pos) / (len(pos) + len(neg)))
    pos, neg = pos[:, [0, 2]], neg[:, [0, 2]]
    plt.scatter(pos[:, 0], pos[:, 1], color='black', alpha=.5)
    plt.scatter(neg[:, 0], neg[:, 1], color='red', alpha=.5)
    plt.title(f'Verification of Gp')
    plt.xlabel('x')
    plt.ylabel('theta')
    plt.show()


def cartpole_normal():
    env = Cartpole()
    numActions = 2
    
    numEpisodes, iters, trials = 1, 50, 1
    policy = LinearApprox(numActions)
    np.random.seed(42)

    populationSize = 8
    dim = policy.parameters.shape[0]
    with Pool(8) as p:
        eval_fn = partial(run_n_episodes_p, env, policy, p, max_steps=10000)
        bbo = GA(
            populationSize = populationSize, 
            numEpisodes = numEpisodes, 
            numElite = 2, 
            K_p = 2, 
            alpha = 1., 
            initPopulationFunction = lambda populationSize: np.random.randn(populationSize, dim), 
            evaluationFunction = lambda theta, numEpisodes, **kwargs: eval_fn(theta=theta, n=numEpisodes, **kwargs)
        )
        t0 = perf_counter()
        returns = train_agent_n_trials(env, policy, bbo, iters=iters, trials=trials)
        print(f'elapsed: {perf_counter() - t0}')

    pos, neg = verif(env, policy, '((F p) & (F q))', inputsToTry=1000, p=lambda x: x < -0.25, q=lambda x: x > 0.25)
    print(len(pos) / (len(pos) + len(neg)))
    pos, neg = pos[:, [0, 2]], neg[:, [0, 2]]
    plt.scatter(pos[:, 0], pos[:, 1], color='black', alpha=.5)
    plt.scatter(neg[:, 0], neg[:, 1], color='red', alpha=.5)
    plt.title(f'No TL reward')
    plt.xlabel('x')
    plt.ylabel('theta')
    plt.show()


def cartpole_tl():
    env = Cartpole()
    numActions = 2
    
    numEpisodes, iters, trials = 1, 50, 1
    policy = LinearApprox(numActions)
    np.random.seed(42)

    populationSize = 8
    dim = policy.parameters.shape[0]
    with Pool(8) as p:
        eval_fn = partial(run_n_episodes_p, env, policy, p, max_steps=10000, mtl_reward_expr='((F p) & (F q))')
        bbo = GA(
            populationSize = populationSize, 
            numEpisodes = numEpisodes, 
            numElite = 2, 
            K_p = 2, 
            alpha = 1., 
            initPopulationFunction = lambda populationSize: np.random.randn(populationSize, dim), 
            evaluationFunction = lambda theta, numEpisodes, **kwargs: eval_fn(theta=theta, n=numEpisodes, **kwargs)
        )
        t0 = perf_counter()
        returns = train_agent_n_trials(env, policy, bbo, iters=iters, trials=trials)
        print(f'elapsed: {perf_counter() - t0}')

    pos, neg = verif(env, policy, '((F p) & (F q))', inputsToTry=1000, p=lambda x: x < -0.25, q=lambda x: x > 0.25)
    print(len(pos) / (len(pos) + len(neg)))
    pos, neg = pos[:, [0, 2]], neg[:, [0, 2]]
    plt.scatter(pos[:, 0], pos[:, 1], color='black', alpha=.5)
    plt.scatter(neg[:, 0], neg[:, 1], color='red', alpha=.5)
    plt.title(f'TL reward based on Fp ∧ Fq')
    plt.xlabel('x')
    plt.ylabel('theta')
    plt.show()

def main():
    cartpole_verify()
    cartpole_normal()
    cartpole_tl()


if __name__ == "__main__":
    main()
