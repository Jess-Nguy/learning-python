import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

np.random.seed(seed=1)
sales_raw=[0,1,2,4,5.5,6.7,7.5,8,7.5,7,5,2,1,0.5,0]

def tot_sales(day):
    """Return total sales in dollars for day of the year (0-364)"""
    f2 = 0
    if (day >= 0 and day < 365):
        f1 = (day % 28) / 28
        f2 = (1-f1) * sales_raw[day // 28] + f1 * sales_raw[day // 28 + 1]
    tot = 7 + f2*5 + abs(np.random.normal(2*f2,f2+10))
    return tot * 10
day = list(range(0,365))
sales = list(map(tot_sales, day))

def gen_normal(mean, sd, length_data):
    """Generate 20 points normally distributed
    Keyword arguments:
        mean -- the distribution average (default 10.0)
        sd -- the distribution standard deviation (default 1.0)
        length_data -- is the length of the random data will be
    Return Value:
        A triple incl (mean, sd, and data distribution)
    """
    assert sd>0, "Standard Deviation must be > 0"
    data = np.random.normal(mean, sd, length_data)
    return [mean, sd, data]

## Plot data
fig = plt.figure(figsize=[8,6],dpi=100)
# Title
fontsize = 28
title = 'Ice Cream Cone Sales vs. Month'
title_text_obj = plt.title(title, fontsize=fontsize)
title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

### Plot Layout
ax = plt.axes()

x_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
x = np.arange(0,365,31)
y = np.arange(0,1110,200)
plt.ylim([0,1200])
plt.yticks(y)
plt.xticks(x)
ax.set_yticklabels(y,fontweight='bold')
ax.set_xticklabels(x_labels, fontweight='bold')
plt.ylabel("Monthly Sales (Dollars)", fontsize=10)
plt.grid(axis = 'y',  linestyle = '--')

### Monthly Avg
monthly_average = []
monthly_range = np.arange(0,366,30)
# to fit exactly 365 so month will have an extra day.
extra_days = [0,2,4,6,9]
after_extra_days = [1,3,5,10]
for idx in range(0, len(monthly_range)):
    if idx < len(monthly_range) - 1:
        if idx in extra_days:
            monthly_sale = sales[monthly_range[idx]:monthly_range[idx+1]+1]
        elif idx in after_extra_days:
            monthly_sale = sales[monthly_range[idx]+1:monthly_range[idx+1]]
        else:
            monthly_sale = sales[monthly_range[idx]:monthly_range[idx+1]]
        monthly_sale = np.array(monthly_sale)
        monthly_average.append(np.mean(monthly_sale))

print(len(monthly_average))
print(monthly_average)
plt.bar(x,monthly_average, width=28,label="Monthly Avg", color="#e5e599", edgecolor="#a4a492", zorder=1)

### Daily Sales
plt.scatter(day, sales,s=np.full(len(day),10), color="red", marker="v", label="Daily Sales", zorder=2)

### Avg = $432.83
sales = np.array(sales)
whole_sales_avg = round(np.mean(sales),2)
average_legend = "Avg = $" + str(whole_sales_avg)
mean, sd, data = gen_normal(whole_sales_avg,1,len(sales))
plt.plot(data,label=average_legend, color="black", linestyle='dashdot')

### Yearly Max/Min Annotation
max_sale = round(np.max(sales),2)
x_max_idx = day[sales.argmax()]
yearly_max = "Yearly Max = $" + str(max_sale)

min_sale = round(np.min(sales),2)
x_min_idx = day[sales.argmin()]
yearly_min = "Yearly Min = $" + str(min_sale)

plt.annotate(yearly_max, xy =(x_max_idx, max_sale), xytext=(0,1120), xycoords='data', weight='bold', arrowprops=dict(arrowstyle='->', linestyle = '-'))

plt.annotate(yearly_min, xy =(x_min_idx, min_sale), xytext=(0,1050), xycoords='data', weight='bold', arrowprops=dict(arrowstyle='->', linestyle = '-'))

### Legend. Each plt plot needs a label for it to show up here.
legend_properties = {'weight':'bold'}
plt.scatter([x_max_idx,x_min_idx], [max_sale, min_sale],s=np.full(2,100),  color="red",  edgecolor="black", marker="o", label="Max/Min Sales", zorder=3)
plt.legend(loc = "upper right", fontsize=10, prop=legend_properties)

plt.show()