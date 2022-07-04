import os
import argparse

import torch
import torch.fx as fx
import torch.quantization.quantize_fx as quantize_fx

import torch_pretrained_model_compress as compressed_modelfile


# Parsing the commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--prune', type=float, default=0.3)
args = parser.parse_args()


model_path = compressed_modelfile.model_path  # path toward saved target model
model_name = compressed_modelfile.model_name  # target model name

test_dataloader = compressed_modelfile.test_dataloader  # test dataset
test = compressed_modelfile.modelfile.test        # test function
loss_fn = compressed_modelfile.modelfile.loss_fn  # loss function

model = compressed_modelfile.NetworkModel()    # generate model of modelfile
quant_type = compressed_modelfile.quant_type
model.eval()                                   # set model into evaluation mode
qconfig = compressed_modelfile.qconfig
qconfig_dict = {"": qconfig}                   # generate Qconfig

model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)                       # preparation
model_quantized = quantize_fx.convert_fx(model_prepared)                           # convert the model
model_quantized.load_state_dict(torch.load(os.path.join(model_path, model_name)))  # load save state_dict

target_model = model_quantized
prune_amount = args.prune                                                          # pruning amount
output_modelname = model_name                                                      # model name for output data
output_dirname = os.path.join(os.curdir, "torch_model_outputs", output_modelname)  # path toward saved data
os.makedirs(output_dirname, exist_ok=True)

features = {}             # dictionary containing extracted data
activation_cnt_limit = 5  # the number of intermidiate activation to save

class OutputExtractor(fx.Interpreter):
    def __init__(self, gm):
        super(OutputExtractor, self).__init__(gm)
        self.traces = []

    def call_module(self, target, *args, **kwargs):
        for kw in self.traces:
            if kw in target.split('.'):
                idx = 0
                save_output_name = f"{output_modelname}_{target}_output{idx}"
                if save_output_name in features:
                    idx += 1
                    save_output_name = f"{output_modelname}_{target}_output{idx}"

                print(f'extracting {save_output_name}')
                features[save_output_name] = super().call_module(target, *args, **kwargs)
        return super().call_module(target, *args, **kwargs)

# Forward propagation by using test dataset provided by compressed_modelfile
iter_cnt = 0
max_iter = 5
device = compressed_modelfile.device
traced = torch.fx.symbolic_trace(target_model)

extractor = OutputExtractor(target_model)
extractor.traces.append('conv')
extractor.traces.append('conv1')
extractor.traces.append('conv2')
extractor.traces.append('conv3')
extractor.traces.append('conv4')
extractor.traces.append('conv5')
extractor.traces.append('fc')

for X, y in test_dataloader:
    if iter_cnt > max_iter: break
    else: iter_cnt += 1
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

with open(os.path.join(output_dirname, 'filelist.txt'), 'wt') as filelist:
    filelist.write('\n'.join([os.path.join(output_dirname, layer_name) for layer_name in features.keys()]))

target_model.graph.print_tabular()