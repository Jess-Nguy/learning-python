import matplotlib.pyplot as plt
import numpy as np


def gen_normal(mean, sd):
    """Generate 20 points normally distributed
    Keyword arguments:
    mean -- the distribution average (default 10.0)
    sd -- the distribution standard deviation (default 1.0)
    Return Value
    A triple incl (mean, sd, and data distribution)
    """
    assert sd > 0, "Standard Deviation must be > 0"
    dat = np.random.normal(mean, sd, 20)
    return [mean, sd, dat]


def gen_subplot(sp, d1, d2):
    """Generate one of 4 subplots as specified in Assignment #1
    Arguments:
    sp -- a number representing the subplot
    d1 -- a triple (mean, sd, data)
    d2 -- a triple (mean, sd, data)
    """
    # 2x2 subplot
    plt.subplot(int(str(22)+str(sp)))

    # concat numbers for legend
    d1_legend = "mean="+str(d1[0])+", sd="+str(d1[1])
    d2_legend = "mean="+str(d2[0])+", sd="+str(d2[1])
    plt.xlabel("Array Index", fontsize=7)
    plt.ylabel("Round Value", fontsize=7)

    # x & y axis limit
    plt.ylim(top=50)
    plt.xlim(right=20)

    # individual x ticks for bar graph
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # x ticks of an interval of 2 & y ticks of an interval of 5
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

    if sp == 1:
        graph_type = "Line Graph"
        plt.plot(d1[2], label=d1_legend, color="aqua", linestyle='dotted')
        plt.axhline(y=d1[0], color='aqua')

        plt.plot(d2[2], label=d2_legend, linestyle='dashed')
        plt.axhline(y=d2[0])
    elif sp == 2:
        graph_type = "Line Graph"
        plt.plot(d1[2], label=d1_legend, color="y", linewidth=3)
        plt.axhline(y=d1[0], color='y', linestyle='dotted')

        plt.plot(d2[2], label=d2_legend, color="green", linewidth=3)
        plt.axhline(y=d2[0], color='green', linestyle='dotted')
    elif sp == 3:
        graph_type = "Bar Graph"
        plt.bar(x, d1[2], label=d1_legend, color="aqua")
        plt.axhline(y=d1[0], color='aqua', linestyle='dotted')

        plt.bar(x, d2[2], label=d2_legend)
        plt.axhline(y=d2[0], linestyle='dotted')
    elif sp == 4:
        graph_type = "Bar Graph"
        plt.bar(x, d1[2], label=d1_legend, color="y")
        plt.axhline(y=d1[0], color='y')

        plt.bar(x, d2[2], label=d2_legend, color="green", width=0.4)
        plt.axhline(y=d2[0], color='green')
    plt.title("Normal Distributions (" + graph_type + ")", fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)


first_normal_data = gen_normal(25, 5)
second_normal_data = gen_normal(10, 2)
plt.figure()  # new figure
for subplot_num in range(1, 5):
    gen_subplot(subplot_num, first_normal_data, second_normal_data)
plt.show()  # show figure
