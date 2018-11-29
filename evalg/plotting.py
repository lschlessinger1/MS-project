import matplotlib.pyplot as plt


def plot_best_so_far(best_so_far):
    # Display the maximum fitness value at each generation
    plt.title('Best-So-Far Curve')
    plt.plot(best_so_far)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Best So Far')
    plt.show()
