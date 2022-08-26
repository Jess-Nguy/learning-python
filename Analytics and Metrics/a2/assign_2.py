import matplotlib.pyplot as plt
import numpy as np
import re

datafile = 'a2_cpu_amd_intel.txt'

CPU_GEN = 5
i3 = []
i5 = []
i7 = []
i9 = []
allRequiredCPU = []
requiredBrandModifer = ["i3", "i5", "i7", "i9"]
div1Generation = ""

# Functions


def findBrandModifer(line):
    """
        finds the index location of the important brand modifer
        line - document line from reader as string
    """
    for bm in requiredBrandModifer:
        if bm in line:
            return line.find(bm)


def findGeneration(gen, brandModiferArray):
    """
        find the specific generation you want.
        gen - Generation number as int
        brandModiferArray - Has all the parsed cpu. Index contain: Clock speed, generation, cpu name, benchmark, price
    """
    filteredGen = []
    for item in brandModiferArray:
        if str(item[1]).startswith(str(CPU_GEN)):
            filteredGen += [item]

    return filteredGen


def top3Process(cpuArray):
    """
        Finds the top 3 CPU from Performance vs Cost.
        cpuArray - Has all the parsed cpu. Index contain: Clock speed, generation, cpu name, benchmark, price
    """
    top3Price = []
    top3Clock = []

    foundGen = findGeneration(CPU_GEN, cpuArray)

    foundGen.sort(key=lambda x: x[3], reverse=True)
    # Grab the first 2 best cpumark(benchmark)
    top3Price += [foundGen[0], foundGen[1]]
    top3Clock += [foundGen[0], foundGen[1]]

    foundGen.sort(key=lambda x: x[4])
    # Grab the best price
    top3Price += [foundGen[0]]

    foundGen.sort(key=lambda x: x[0], reverse=True)
    # Grab the best clock speed
    top3Clock += [foundGen[0]]

    return top3Price, top3Clock


def plotPoints(type):
    """
        Plot scatter points
        type - y point type clock(1) or price(2) as int
    """
    plt.ylabel("cpuMark")
    if len(i3) > 0:
        plt.scatter(i3[:, type], i3[:, 0], color='blue',
                    marker="^", label="i3")
    if len(i5) > 0:
        plt.scatter(i5[:, type], i5[:, 0], color='orange',
                    marker="x", label="i5")
    if len(i7) > 0:
        plt.scatter(i7[:, type], i7[:, 0], color='green',
                    marker="*", label="i7")
    if len(i9) > 0:
        plt.scatter(i9[:, type], i9[:, 0], color='red', marker="o", label="i9")


def annotateBestPoints(x, y, type, top3):
    """
        Annotate top3 arrays
        x - x points for annotation as float or int
        y - y points for annotation as int
        type -  y point type clock(0) or price(4) as int
        top3 - array of the top 3 gen for comparsion of clock or price
    """
    plt.annotate(titleAnnotate, xy=(x, y), xytext=(x, y))
    plt.annotate(top3[0][2].split()[2:][0], xy=(top3[0][type], top3[0][3]), xycoords='data', xytext=(
        x, y-200), arrowprops=dict(arrowstyle='->', linestyle='--'))
    plt.annotate(top3[1][2].split()[2:][0], xy=(top3[1][type], top3[1][3]), xycoords='data', xytext=(
        x, y-400), arrowprops=dict(arrowstyle='->', linestyle='--'))
    plt.annotate(top3[2][2].split()[2:][0], xy=(top3[2][type], top3[2][3]), xycoords='data', xytext=(
        x, y-600), arrowprops=dict(arrowstyle='->', linestyle='--'))


# Read and parse data
with open(datafile, 'r') as f:
    for line in f:
        # Skip any lines that is comment, just \n, without clock speed (hz), has NA for price, is not i3,i5,i7,i9
        if "#" in line or "\n" == line or "GHz" not in line or "NA" in line or ("i3" not in line and "i5" not in line and "i7" not in line and "i9" not in line):
            continue

        # Split line into 2 to get rid of (%)
        div1 = line[0:line.find('(')]
        div2 = line[line.find(')')+2:]

        # Get cpu bench and clock speed.
        # Generation delimitter is `-`
        if "-" in div1:
            div1Generation = div1[div1.find('-')+1:].split()
        else:
            # Generation delimitter is `space`
            div1Generation = div1[findBrandModifer(div1)+3:].split()

            # Generation has letter combined and after.
            if len(div1Generation) == 3:

                div1Generation = [div1Generation[0],
                                  div1Generation[1], div1Generation[2]]

            # Generation has letter is not combine and is before number.
            elif len(div1Generation) > 3:
                div1Generation = [div1Generation[1],
                                  div1Generation[2], div1Generation[3]]
        regGenNumber = re.findall('(\d+(?:\.\d+)?)', div1Generation[0])
        clockIndex = [i for i, elem in enumerate(
            div1Generation) if "GHz" in elem]

        regClock = re.findall('(\d+(?:\.\d+)?)', div1Generation[clockIndex[0]])
        div1Generation[2] = float(regClock[0])

        parsedData = [div1Generation[2], int(regGenNumber[0])]

        parsedData += [div1[:div1.find('@')-1]]
        # Add CPU benchmark and price.

        parsedData += div2.strip('\n').split()

        # Clean up price
        regExPrice = re.findall(
            '(\d+(?:\.\d+)?)', parsedData[len(parsedData)-1])

        if len(regExPrice) == 1:
            parsedData[len(parsedData)-1] = float(regExPrice[0])
        else:
            parsedData[len(parsedData)-1] = float(''.join(regExPrice))

        # Clean up benchmark of commas
        if "," in parsedData[len(parsedData)-2]:
            parsedData[len(
                parsedData)-2] = int(parsedData[len(parsedData)-2].replace(',', ''))

        # Clock speed, generation, cpu name, benchmark, price
        allRequiredCPU += [parsedData]

# Filter data for scatter points
for data in allRequiredCPU:

    # generate x values (Clock speed in GHz)
    x1 = data[0]
    # generate x values (Price in USD)
    x2 = data[4]
    # generate y values (cpuMark)
    y = data[3]

    # Assign each brand modifer the x1,x2,y points
    if "i3" in data[2]:
        i3 += [[int(y), float(x1), float(x2)]]
    elif "i5" in data[2]:
        i5 += [[int(y), float(x1), float(x2)]]
    elif "i7" in data[2]:
        i7 += [[int(y), float(x1), float(x2)]]
    elif "i9" in data[2]:
        i9 += [[int(y), float(x1), float(x2)]]

# Scatter plot
# Enlargen the figure size
plt.figure(figsize=(12, 8), dpi=100)
minCpumark = 500
maxCpumark = 4750
# axes for cpu & clock speed
ax1 = plt.subplot(221)
plt.axis([1, 4.5, minCpumark, maxCpumark])

# convert to np array to allow column selection for scatter
i3 = np.array(i3)
i5 = np.array(i5)
i7 = np.array(i7)
i9 = np.array(i9)

plt.xlabel("Clock Speed in GHz")
plotPoints(1)

plt.legend(loc="lower left", fontsize=8)
plt.title('Intel Single Core Performance vs. Clock Speed')

# TOP 3 - Annotation
top3Clock, top3Price = top3Process(allRequiredCPU)
print(top3Clock)
titleAnnotate = "Best " + str(CPU_GEN) + "th Gen Intel Processors"
annotateBestPoints(1.25, 4200, 0, top3Clock)

# axes for cpu & price
ax2 = plt.subplot(222)

plt.xscale("log")
xPrice = [25, 50, 100, 250, 500, 1000, 2500]
plt.xticks(xPrice, xPrice)
plt.axis([20, 2500, minCpumark, maxCpumark])

plt.xlabel("Price in USD")
plotPoints(2)
annotateBestPoints(25, 4200, 4, top3Price)
plt.title('Intel Single Core Performance vs. Price')
plt.show()
