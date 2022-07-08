import os
import torch
import parse
import numpy as np
import matplotlib.pyplot as plt


class LayerConfig(object):
    def __init__(self, model_type, target_prune_amount, batch_size, step):
        super(LayerConfig, self).__init__()

        self.model_type = model_type
        self.target_prune_amount = target_prune_amount
        self.batch_size = batch_size
        self.step = step

    def name(self, model_type=False, target_prune_amount=False, batch_size=False, step=False):
        name_element = []
        if model_type: name_element.append(self.model_type)
        if target_prune_amount: name_element.append(self.target_prune_amount)
        if batch_size: name_element.append(self.batch_size)
        if step: name_element.append(self.step)

        return '_'.join(name_element)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

dirname = os.path.join(os.curdir, 'state_dict-only_prune')
categories = [30, 50, 70]
result = {
    "step5":     [],
    "step10":    [],
    "reference": [30., 50., 70.],
}

for filename in sorted(os.listdir(dirname)):
    filepath = os.path.join(dirname, filename)
    if not os.path.isfile(filepath): continue

    state_dict = torch.load(filepath, map_location=device)
    model_name = os.path.split(filepath)[-1].split('.')[0]
    layer_config = LayerConfig(*parse.parse("{}_{}_b{}_step{}_statedict", model_name))
    key = f"{layer_config.target_prune_amount}"

    # if layer_config.step != "10":
    #     continue

    # result["reference"].append(float(layer_config.target_prune_amount))
    result[f"step{layer_config.step}"].append(0)
    layer_cnt = 0

    print(f"testing {model_name}")
    for lidx, (param_name, param) in enumerate(state_dict.items()):
        if (("weight" not in param_name) and ("bias" not in param_name)) or ("layer1" not in param_name):
            continue

        cnt, total = 0, 0
        for val in torch.flatten(param):
            if val == 0:
                cnt += 1
            total += 1

        # print(f"{param_name:50s}: {cnt}/{total}({cnt / total * 100:.4f}%)")
        result[f"step{layer_config.step}"][-1] += cnt / total * 100
        layer_cnt += 1

    result[f"step{layer_config.step}"][-1] /= layer_cnt

print(result)

width_max = 0.6
width = width_max / len(result.keys())

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(result.items()):
    plt.bar(x_axis + ((idx - (len(result.keys()) / 2) + 0.5) * width), val, width=width, label=key)
plt.xticks(x_axis, categories, rotation=0, ha='center')
plt.xlabel('target pruning amount')
plt.ylabel('calculated pruning amount')
plt.legend()
plt.tight_layout()
plt.show()
