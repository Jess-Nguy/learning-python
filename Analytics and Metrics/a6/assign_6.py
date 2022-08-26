import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats
import os
dir_path='linux_arch_x86'


def line_counter(file_name):
    """count the number of lines in the supplied file name. Lines of Code."""
    tot_line = 0
    with open(file_name, 'r') as f:
        for line in f:
            tot_line += 1
    return (tot_line)
def max_depth(file_name):
    """Return the maximum nesting depth of a file. Indentation Depth."""
    max_dep = 0
    dep = 0
    with open(file_name, 'r') as f:
        for line in f:
            for ch in line:
                if ch == '{':
                    dep += 1
                if ch == '}':
                    dep -= 1
                if dep > max_dep:
                    max_dep = dep
    return max_dep
def block_counter(file_name):
    """Count the number of root level code blocks in a file. Function COUNTER."""
    block_tot = 0
    dep = 0
    with open(file_name, 'r') as f:
        for line in f:
            for ch in line:
                if ch == '{':
                    dep += 1
                if ch == '}':
                    dep -= 1
                    if dep == 0:
                        block_tot += 1
    return block_tot
def semi_counter(file_name):
    """Count the number of lines that end in a semicolon. Logical SLOC."""
    tot_semi = 0
    with open(file_name, 'r') as f:
        for line in f:
            sline = line.strip()
            if len(sline) > 0 and sline[-1] == ';':
                tot_semi += 1
    return (tot_semi)

def char_counter(file_name):
    """Return the char count for a file."""
    tot_chars = 0
    with open(file_name, 'r') as f:
        for line in f:
            tot_chars += len(line)
    return (tot_chars)

def comm_counter(file_name):
    """Count the number of chars within a comment block or line."""
    comm_chars = 0
    with open(file_name, 'r') as f:
        block_comment = False
        for line in f:
            lch = None
            line_comment = False
            for ch in line:
                if lch == '/' and ch == '*':
                    block_comment = True
                if lch == '*' and ch == '/':
                    block_comment = False
                if lch == '/' and ch == '/':
                    line_comment = True
                lch = ch
                if line_comment or block_comment:
                    comm_chars += 1
    return comm_chars

## Parse file for data
func_lines = []
semi_colons = []
max_depths = []
percent_comments = []
for root, dirs, files in os.walk(dir_path):
    for file in files:
        fpfile = root + '\\' + file
        file_ext = file.split('.')[-1]
        if file_ext == "c":
            if block_counter(fpfile) > 0:
                line = (line_counter(fpfile) // block_counter(fpfile))
                func_lines.append(line)
            if max_depth(fpfile) > 0:
                max_depths.append(max_depth(fpfile))
                percent_comments.append(round(100 * (comm_counter(fpfile)/char_counter(fpfile))))
                semi_colons.append(round(100 * (semi_counter(fpfile)/line_counter(fpfile))))


## Function Length in Lines (max - 100)
## Average per File Function Length in Lines (max - 99)
## Relative Frequency (max - 66 to 68)
# I couldn't quite get the average function line to be under 100 just like the example. I used the same function in week 11 so I'm not quite sure what is different. I wasn't sure if I had to create my own function to figure out the average.
# number of function length in lines divided by total function length in lines frequency
# print(func_lines)
# print(len(func_lines))
# print(max(func_lines))
# print(func_lines)
# print(len(func_lines))
# print(max(func_lines))
# relative_func = []
# sum_function = sum(func_lines);
# for x in func_lines:
#     test = 100*(x/sum_function)
#     print(test)
#     # print(x/sum_function)
#     relative_func.append(test)

fig = plt.figure(figsize=(8,6), dpi=100)
fig.suptitle('Linux Kernel Arch=x86 Per File Code Complexity Statistics')
plt.subplots_adjust(hspace=.6)

## Comment Ratio vs Max {} Depth, r=0.0
ax2 = plt.subplot(222)
# Partially positively correlated data
slope, intercept, r, p, se = stats.linregress(max_depths, percent_comments)
subplot_title2 = "Comment Ratio vs Max {} Depth, r=" + str(round(r))
ax2.title.set_text(subplot_title2)
plt.xlabel("Maximum per File {} Indendation Depth", fontsize=7)
plt.ylabel("Percentage of Comments to Code", fontsize=7)
plt.xticks(np.arange(1,9))
plt.yticks(np.arange(0,81,10))
plt.scatter(max_depths, percent_comments, color="blue", marker="_")
def line(x):
    """line based on global slope and intercept"""
    y = intercept + slope * x
    return y
y_best_fit = list(map(line,max_depths))
print(max_depths,y_best_fit)
plt.plot(max_depths,y_best_fit, color="black")


## % Lines Ending in ';' vs. Max {} Depth, r=0.35
ax4 = plt.subplot(224)
# Partially positively correlated data
slope, intercept, r, p, se = stats.linregress(max_depths, semi_colons)
subplot4_title = "% Lines Ending in ';' vs. Max {} Depth, r=" + str(round(r, 2))
ax4.title.set_text(subplot4_title)
plt.xlabel("Maximum per File {} Indendation Depth", fontsize=7)
plt.ylabel("Percentage of Lines Ending in ';'", fontsize=7)
plt.xticks(np.arange(1,9))
plt.yticks(np.arange(0,71,10))
plt.scatter(max_depths, semi_colons, color="blue", marker="_")
y_best_fit = list(map(line,max_depths))
print(max_depths,y_best_fit)
plt.plot(max_depths,y_best_fit, color="black")

plt.show()