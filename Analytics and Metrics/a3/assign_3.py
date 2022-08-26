import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib import patheffects

# Parse file for data
filename = 'a3_2021f_coop_tech.csv'
data = []
parsedData = set()
try:
    with open(filename) as f:
        reader = csv.reader(f)
        # skip header
        next(reader)
        data = [row[0] for row in reader]

except csv.Error as e:
    print("Error reading CSV file at line %s: %s" % (reader.line_num, e))
    sys.exit(-1)

languages = {l.title() for l in data}

data = np.array(data)
# Only grab languages that appear 2 or more times.
otherCount = 0
for language in languages:
    languageIndices = [dataIndex for dataIndex in range(
        len(data)) if data[dataIndex].lower() == language.lower()]
    if len(languageIndices) >= 2:
        parsedData.add((data[languageIndices[0]], round(
            len(languageIndices)/len(data)*100)))
    else:
        otherCount += 1
parsedData.add(("Other", round(otherCount/len(data)*100)))

# Graph data into pie chart
parsedData = np.array(list(parsedData))

labels = np.array(parsedData[:, 0])

x = np.array(parsedData[:, 1])
x = x.astype(np.int64)
x, labels = zip(*sorted(zip(x, labels)))
x = np.array(x)

explode_spc = [0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

explode_spc[x.argmax()] = 0.1

cmap = plt.get_cmap("tab20c")
colours = cmap(np.array([1, 2, 5, 6, 9, 10, 12, 13, 15, 17, 18, 20]))

# Enlargen the figure size
plt.figure(figsize=(12, 8), dpi=100)


def make_autopct(x):
    """
        Make an autopct that takes the x values and adds % to it.
        Only format values that's bigger than 2.
        x - array of number for pie chart
    """
    def my_autopct(pct):
        """
            Iterate through x array values and change format.
            pct - individual values of x
        """
        total = sum(x)
        val = int(round(pct*total/100.0))
        return '{v:d}%'.format(p=pct, v=val) if val > 2 else ''
    return my_autopct


ax = plt.pie(x, explode=explode_spc, labels=labels, autopct=make_autopct(
    x), startangle=142, rotatelabels=True, colors=colours, labeldistance=1.05, pctdistance=0.80)

# Set 'Other' label rotation
for tx in ax[1]:
    if(tx.get_text() == 'Other'):
        rot = tx.get_rotation()
        tx.set_horizontalalignment('center')
        tx.set_rotation(rot+90+(1-rot//180)*180)
# Set 'Other' data text with explanation
for tx in ax[2]:
    if(tx.get_text() == str(x[x.argmax()])+'%'):
        tx.set_text(str(
            x[x.argmax()]) + '%\n\n Many Specliaized Techologies\n Were Reported Only Once')

fontsize = 18
title = 'Technologies Used by COOPs in Fall 2021'
title_text_obj = plt.title(title, fontsize=fontsize, fontweight='bold')
title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

# offset_xy -- set the 'angle' of the shadow
# patch_alpha -- setup the transparency of the shadow
offset_xy = (4, -4)
patch_alpha = .2
# customize shadow properties
pe = patheffects.withSimplePatchShadow(offset=offset_xy, alpha=patch_alpha)

plt.show()
