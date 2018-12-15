import matplotlib.pyplot as plt


def plot_best_so_far(best_so_far, title='Best-So-Far Curve', x_label='Generation', y_label='Fitness Best So Far'):
    # Display the maximum fitness value at each generation
    plt.title(title)
    plt.plot(best_so_far)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
