import gym

ENV_NAME = 'rlsp-env-v1'

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    print(env)