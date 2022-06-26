import os
import torch
import torch.fx as fx
import torch.quantization.quantize_fx as quantize_fx

import torch_pretrained_model_loader as modelfile


model_path = modelfile.model_path  # path toward saved target model
model_name = modelfile.model_name  # target model name

test_dataloader = modelfile.test_dataloader  # test dataset
test = modelfile.test                        # test function
loss_fn = modelfile.loss_fn                  # loss function

model = modelfile.NetworkModel()  # generate model of modelfile
# model.load_state_dict(torch.load(os.path.join(model_path, model_name)))  # load save state_dict
quant_type = modelfile.quant_type
model.eval()                 # set model into evaluation mode
qconfig = modelfile.qconfig
qconfig_dict = {"": qconfig} # generate Qconfig

model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)  # preparation
# calibrate(model_prepared, test_dataloader)                    # calibration
model_quantized = quantize_fx.convert_fx(model_prepared)      # convert the model
model_quantized.load_state_dict(torch.load(os.path.join(model_path, model_name)))  # load save state_dict

target_model = model_quantized
prune_amount = modelfile.prune_amount  # pruning amount
output_modelname = model_name  # model name for output data
output_dirname = os.path.join(os.curdir, "torch_model_outputs", output_modelname)  # path toward saved data
os.makedirs(output_dirname, exist_ok=True)

features = {}             # dictionary containing extracted data
activation_cnt_limit = 5  # the number of intermidiate activation to save

# Define specific action while extracting data
# @torch.fx.wrap
# def save_output(name, idx, output):
#     # Extract output data
#     print(f"extracting {name}_output{idx}")
#     features[f"{name}_output{idx}"] = output
#
# def make_hook(name):
#     def extract_output(model, model_input, model_output):
#         # Calculate activation count
#         activation_cnt = 1
#         while f"{name}_output{activation_cnt}" in features: activation_cnt += 1
#         if activation_cnt > activation_cnt_limit: return
#         model_output.detach()
#         return save_output(name, idx=activation_cnt, output=model_output)
#     return extract_output
#
# Define layers to extract output
# hooks = list()
# hooks.append(target_model.layer1.register_forward_hook(make_hook(f'{output_modelname}_layer1')))
# hooks.append(target_model.layer2.register_forward_hook(make_hook(f'{output_modelname}_layer2')))
# hooks.append(target_model.layer3.register_forward_hook(make_hook(f'{output_modelname}_layer3')))
# hooks.append(target_model.layer4.register_forward_hook(make_hook(f'{output_modelname}_layer4')))
# target_model.recompile()

class OutputExtractor(fx.Interpreter):
    def __init__(self, gm):
        super(OutputExtractor, self).__init__(gm)
        self.traces = []

    def call_module(self, target, *args, **kwargs):
        if target in self.traces:
            idx = 0
            save_output_name = f"{target}_output{idx}"
            if save_output_name in features:
                idx += 1
                save_output_name = f"{target}_output{idx}"

            print(f'extracting {save_output_name}')
            features[save_output_name] = super().call_module(target, *args, **kwargs)
        return super().call_module(target, *args, **kwargs)

# Forward propagation by using test dataset provided by modelfile
# test(test_dataloader, target_model, loss_fn, max_iter=activation_cnt_limit)
iter_cnt = 0
max_iter = 5
device = modelfile.device
traced = torch.fx.symbolic_trace(target_model)

extractor = OutputExtractor(target_model)
extractor.traces.append('layer1.0.conv3')
extractor.traces.append('layer1.1.conv3')
extractor.traces.append('layer1.2.conv3')

extractor.traces.append('layer2.0.conv3')
extractor.traces.append('layer2.1.conv3')
extractor.traces.append('layer2.2.conv3')
extractor.traces.append('layer2.3.conv3')

extractor.traces.append('layer3.0.conv3')
extractor.traces.append('layer3.1.conv3')
extractor.traces.append('layer3.2.conv3')
extractor.traces.append('layer3.3.conv3')
extractor.traces.append('layer3.4.conv3')
extractor.traces.append('layer3.5.conv3')

extractor.traces.append('layer4.0.conv3')
extractor.traces.append('layer4.1.conv3')
extractor.traces.append('layer4.2.conv3')

for X, y in test_dataloader:
    if iter_cnt > max_iter:
        break
    else:
        iter_cnt += 1
    X, y = X.to(device), y.to(device)
    extractor.run(X)
    print(f'extraction iter: {iter_cnt}')

# Extracting parameters
for param_name in target_model.state_dict():
    if 'weight' in param_name:
        parsed_name = f"{output_modelname}_{param_name.replace('.', '_')}"
        try:
            print(f"extracting {parsed_name}")
            features[parsed_name] = target_model.state_dict()[param_name].int_repr().detach()
        except:
            print(f"error occurred on extracting {parsed_name}")

print(f"\n{len(features)} data extracted!")

# Saving extracted data
for layer_name in features.keys():
    torch.save(features[layer_name], os.path.join(output_dirname, f"{layer_name}"))

# for hook_handle in hooks:
#     hook_handle.remove()

target_model.graph.print_tabular()