import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib import patheffects

# Parse file for data
filename = 'a4_hamilton_climate_2021.csv'
data = []
parsedData = set()
try:
    with open(filename, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if row["Mean Temp (°C)"] != '':
                data.append({"Month": row["Month"], "Day": row["Day"], "Max Temp (°C)": row["Max Temp (°C)"],
                            "Min Temp (°C)": row["Min Temp (°C)"], "Mean Temp (°C)": row["Mean Temp (°C)"]})
except csv.Error as e:
    print("Error reading CSV file at line %s: %s" % (reader.line_num, e))
    sys.exit(-1)


fig = plt.figure(figsize=(8, 6), dpi=100)
fig.suptitle('Hamilton Temperature Statistics By Month and Day for 2021')

# Daily Average Figure
ax1 = plt.subplot(3, 1, 1)
plt.subplots_adjust(hspace=.25)
ax1.title.set_text('Daily Average')
plt.grid(axis='y',  linestyle='--')
plt.ylabel("Degree Celcius", fontsize=10)

# individual y ticks for graph
y = np.arange(-15, 30, 5)
plt.yticks(y)
plt.margins(0)
plt.xticks([])

meanData = [float(row["Mean Temp (°C)"]) for row in data]
# plot data
plt.plot(meanData, color="grey")

# Monthly Mean, Median, Min and Max Figure
ax2 = plt.subplot(3, 2, (3, 6))
ax2.title.set_text('Monthly Mean, Median, Min and Max (of Daily Average)')

x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ax2.set_ylabel("Degree Celcius", fontsize=10)
ax2.set_yticks(y)

plt.grid(linestyle='--')

# parse months for mean
monthIndex = 1
monthData = []
monthsData = []
for dic in data:
    if int(dic["Month"]) != monthIndex:
        monthsData.append(monthData)
        monthIndex += 1
        monthData = []
    else:
        monthData.append(float(dic["Mean Temp (°C)"]))
# Append 12th month
monthsData.append(monthData)
y = [np.array(month).mean() for month in monthsData]

# Labels, ticks and plots
plt.boxplot(monthsData, widths=.4,
            boxprops=dict(color='tab:grey', lw=1.5),
            whiskerprops=dict(color='tab:grey', lw=1.5),
            medianprops=dict(color='black', lw=1.2, ls='dotted'),
            capprops=dict(color='black', lw=2.5))

xIndexTicks = np.arange(1, 13)

ax2.set_xticks(xIndexTicks)
ax2.set_xticklabels(x)

plt.bar(xIndexTicks, y, align='center', width=.7, lw=1.5,
        color='white', edgecolor='grey', hatch="/")

plt.show()
