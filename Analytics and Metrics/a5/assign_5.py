import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
# from pandas import DataFrame, Series

filenames = ['a5_AMZN.csv', 'a5_GOOGL.csv', 'a5_MSFT.csv']
amzn_data = []
msft_data = []
googl_data = []
for file in filenames:
    try:
        with open(file, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=",")

            for row in reader:
                if file == 'a5_AMZN.csv':
                    amzn_data.append(
                        {"Date": row["Date"], "Open": float(row["Open"])})
                elif file == 'a5_MSFT.csv':
                    msft_data.append(
                        {"Date": row["Date"], "Open": float(row["Open"])})
                elif file == 'a5_GOOGL.csv':
                    googl_data.append(
                        {"Date": row["Date"], "Open": float(row["Open"])})
    except csv.Error as e:
        print("Error reading CSV file at line %s: %s" % (reader.line_num, e))
        sys.exit(-1)

# set the initial dimensions of the plot
fig = plt.figure(figsize=[8, 6], dpi=100)
# Collect an explicit reference to the axis object
ax = fig.add_subplot(projection='3d')
c = 25

y = ["Amazon", "Microsoft", "Google"]
z = ["100%", "110%", "120%", "130%", "140%", "150%"]
x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def split_parse_data(base, data):
    """
    This is to split and parse the data into individual arrays for x and y axis.
    data - the whole csv diction data
    base - first open stock number to determine if other days are a gain or a lose.
    """
    percents = []
    dates = []
    for val in data:
        percent = ((val["Open"] / base) * 100) - 90
        percents.append(int(round(percent)))
        date = datetime.strptime(val["Date"], "%Y-%m-%d")
        dates.append(date)
    return np.array(percents), np.array(dates)


# data percentage adjustments
amzn_percent, amzn_dates = split_parse_data(amzn_data[0]["Open"], amzn_data)
msft_percent, msft_dates = split_parse_data(msft_data[0]["Open"], msft_data)
googl_percent, googl_dates = split_parse_data(
    googl_data[0]["Open"], googl_data)


# Graphing

xy_params = [[amzn_dates, amzn_percent], [
    msft_dates, msft_percent], [googl_dates, googl_percent]]
z_idx = 1
for coordinates in xy_params:
    c += 10
    color = list(mcolors.CSS4_COLORS.items())[c]
    plt.bar(coordinates[0], coordinates[1], zs=z_idx,
            zdir='y', color=color, alpha=.8, width=6, bottom=90)
    # This should have worked but python is erroring out on pandas datetime so this is commented out for now,
    # plt.plot(coordinates[0], coordinates[1], zs=z_idx, zdir='y', linestyle="dashed", color="black")
    z_idx += 1


# X ticks Month
# set monthly locator
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# remove additional tick and shift location
ticks_loc = ax.get_xticks().tolist()
ticks_loc.remove(ticks_loc[12])
ticks_loc = [i+15 for i in ticks_loc]
ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
# set formatter
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
# set xaxis as a date type
ax.xaxis_date()


## Ticks and tickLabels
ax.set_yticks(np.arange(1, 4))
ax.set_zticks(np.arange(100, 160, 10))
ax.set_zlim3d(90, 150)
ax.set_xticklabels(x, rotation=70)
ax.set_yticklabels(y)
ax.set_zticklabels(z)


plt.title('Tech Stock Gains for 2021', fontsize=20)

ax.view_init(elev=10, azim=-65)

plt.show()
