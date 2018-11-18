import game
import matplotlib.pyplot as plt
import sys

def press(event, game, fig, do_something, do_something_args, initial_state):
    if event.key=="escape":
        plt.close()
    else:
        # print('press', event.key)
        sys.stdout.flush()
        reward, finish = game.action(event.key)
        plt.imshow(game.show())
        plt.title("Latest Reward: " + str(reward))
        fig.canvas.draw()
        if do_something is not None:
            do_something(*do_something_args)
        if finish:
            print("You Won!")
            game.start_game(initial_state)

def attach_interactive(game, fig, do_something=None, do_something_args=None, initial_state=None):
    fig.canvas.mpl_connect('key_press_event', lambda event: press(event, game, fig, do_something,do_something_args, initial_state))

if __name__ == "__main__":
    game = game.GAME()
    fig = plt.figure()

    print("Press ESC to quit!")
    game.start_recording()
    plt.imshow(game.show())
    attach_interactive(game, fig)
    plt.show()
    game.stop_recording(generate_video="human")