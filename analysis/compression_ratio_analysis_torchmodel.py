import matplotlib.pyplot as plt
import numpy as np
import os


analysis_results_dirname = os.path.join(os.curdir, "analysis_results")
analysis_result_filename = "comparison.csv"
filepath = os.path.join(analysis_results_dirname, analysis_result_filename)

categories = []
results = {}

layer_filter = ['weight']
algor_filter = ['BDI', 'FPC', 'BDI+ZV']


with open(filepath, "rt") as file:
    content = file.readlines()
    labels = content[0].split(',')[1:]

    for idx in range(len(labels)):
        labels[idx] = labels[idx].strip()

    for lbl in labels:
        if lbl not in algor_filter:
            continue
        results[lbl] = []

    for line in content[1:]:
        parsed = line.strip().split(',')

        flag = False
        for lay_fil in layer_filter:
            if lay_fil in parsed[0]:
                flag = True

        if not flag:
            continue

        categories.append(parsed[0])
        for idx, lbl in enumerate(labels):
            if lbl not in algor_filter:
                continue
            results[lbl].append(float(parsed[idx+1].strip()))

    labels = algor_filter

    print(results)

width_max = 0.8
width = width_max / len(results.keys())

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(results.items()):
    plt.bar(x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width), val, width=width, label=key)
plt.xticks(x_axis, categories, rotation=45, ha='right')
plt.ylim([0.9, 2.5])

plt.legend()
plt.tight_layout()
plt.show()
