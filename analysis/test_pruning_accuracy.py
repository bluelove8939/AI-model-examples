import os
import numpy as np
import matplotlib.pyplot as plt


filename = os.path.join(os.curdir, 'analysis_results', 'prune-linear-steps+b.csv')
categories = [30, 50, 70]
result_top1 = {
    "step5":     [],
    "step10":    [],
    "reference": [],
}
result_top5 = {
    "step5":     [],
    "step10":    [],
    "reference": [],
}


with open(filename, 'rt') as file:
    content = file.readlines()
    labels = content[0]

    parsed = [element.strip() for element in content[1].split(',')]
    result_top1["reference"] = [float(parsed[5])] * 3
    result_top5["reference"] = [float(parsed[6])] * 3

    for line in sorted(content[2:]):
        parsed = [element.strip() for element in line.split(',')]
        result_top1[f"step{parsed[4]}"].append(float(parsed[5]))
        result_top5[f"step{parsed[4]}"].append(float(parsed[6]))


width_max = 0.6
width = width_max / len(result_top1.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(result_top1.items()):
    ax1.bar(x_axis + ((idx - (len(result_top1.keys()) / 2) + 0.5) * width), val, width=width, label=key)
ax1.set_ylim([0, 100])
ax1.set_title('top1 accuracy')
ax1.set_xlabel('target pruning amount')
ax1.set_ylabel('accuracy')
plt.sca(ax1)
plt.xticks(x_axis, categories, rotation=0, ha='center')
plt.legend()
plt.tight_layout()

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(result_top5.items()):
    ax2.bar(x_axis + ((idx - (len(result_top5.keys()) / 2) + 0.5) * width), val, width=width, label=key)
ax2.set_ylim([0, 100])
ax2.set_title('top5 accuracy')
ax2.set_xlabel('target pruning amount')
plt.sca(ax2)
plt.xticks(x_axis, categories, rotation=0, ha='center')
plt.legend()
plt.tight_layout()

plt.show()
