import gym
import gym_togyzkumalak
import sys
import io

# Set encoding to UTF-8 for Kazakh characters in terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    env = gym.make('Togyzkumalak-v0')
    env.reset()

    done = False
    state, reward = None, None

    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        pass

    print(state.shape, reward)
    pass


if __name__ == '__main__':
    main()
