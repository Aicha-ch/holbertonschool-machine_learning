#!/usr/bin/env python3
"""
plot a stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    This function creates a stacked bar chart to represent
    the quantity of different fruits owned by three people:
    Farrah, Fred, and Felicia.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    people = ['Farrah', 'Fred', 'Felicia']

    fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    fruit_colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    bottom_values = np.zeros(len(people))

    bar_width = 0.5
    for row, label, color in zip(fruit, fruit_labels, fruit_colors):
        plt.bar(people, row, bottom=bottom_values, color=color,
                width=bar_width, label=label)
        bottom_values += row

    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.legend()

    plt.show()
