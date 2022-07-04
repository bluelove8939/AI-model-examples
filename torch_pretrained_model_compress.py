import os
import argparse
import math

import torch
import torch.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune

# Setup warnings
import warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Parsing the commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--prune', type=float, default=0.3)
args = parser.parse_args()


'''
[1] Preparation of the datasets

Required interfaces
    training_data: defines training dataset
    test_data: defines test dataset
    train_dataloader: train dataset loader
    test_dataloader: test dataset loader
'''

import torch_pretrained_model_loader as modelfile

dataset_name = modelfile.dataset_name
target_dataset = modelfile.target_dataset
train_batch_size = modelfile.train_batch_size  # batch size (for further testing of the model)
test_batch_size = modelfile.test_batch_size    # batch size (for further testing of the model)

training_data = modelfile.training_data
test_data = modelfile.test_data

train_dataloader = modelfile.train_dataloader  # prepare dataset for training
test_dataloader = modelfile.test_dataloader    # prepare dataset for testing


'''
[2] Creation of the model

Required interfaces
    NetworkModel: structure of the model
    train: training function
    test: test function

Note
    Creation and optimization of the model only occurrs when this code is call as 'main'
'''

# Creating models
# In pytorch, model inherits nn.Module
# Structure of the model is defined inside the '__init__' method
# Forward propagation of the model is defined in 'forward' method

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Setup for model
model_type = modelfile.model_type
NetworkModel = modelfile.NetworkModel

# Generate model for fine tuning
model = modelfile.model


prune_amount = args.prune  # pruning amount

def prune_all_layers(module: torch.nn.Module, prune_amount):
    for sub_idx, sub_module in module._modules.items():
        if isinstance(sub_module, torch.nn.Module):
            # print(f"Entering layer[{sub_idx}]: {sub_module._get_name()}")
            prune_all_layers(sub_module, prune_amount)
        else:
            if hasattr(sub_module, 'weight'):
                prune.l1_unstructured(sub_module, 'weight', amount=prune_amount)
                prune.remove(sub_module, 'weight')
            if hasattr(sub_module, 'bias'):
                prune.l1_unstructured(sub_module, 'bias', amount=prune_amount)
                prune.remove(sub_module, 'bias')

if __name__ == '__main__':
    iter_amount = 10
    iter_cnt = math.ceil(prune_amount * 10)

    for iter_idx in range(iter_cnt):
        print(f"pruning iter: {iter_idx:2d}/{iter_cnt:2d}", end='')
        prune_all_layers(model, prune_amount=iter_amount)
        print('  training...')
        modelfile.train(train_dataloader, model,
                        loss_fn=modelfile.loss_fn,
                        optimizer=modelfile.optimizer,
                        max_iter=10,
                        verbose=True)

print("\npruning completed")

quant_type = 'static'
model.eval()                                                 # set model into evaluation mode
qconfig = torch.quantization.get_default_qconfig('fbgemm')  # set Qconfig
qconfig_dict = {"": qconfig}                                 # generate Qconfig

def calibrate(model, data_loader):         # calibration function
    cnt = 1
    model.eval()                           # set to evaluation mode
    with torch.no_grad():                  # do not save gradient when evaluation mode
        for image, target in data_loader:  # extract input and output data
            model(image)                   # forward propagation
            print(f'\rcalibration iter: {cnt:3d}/{len(data_loader):3d}', end='')
            cnt += 1
    print()

if __name__ == '__main__':
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)  # preparation
    calibrate(model_prepared, test_dataloader)                    # calibration
    model_quantized = quantize_fx.convert_fx(model_prepared)      # convert the model

    print('quantization completed')
    print(model_quantized)


'''
[3] Saving generated models

Required interfaces
    model_name: name of the model (state_dict)
    model_path: path toward the saved model (state_dict)

Note
    There are two ways to save model
    1) Saving with state_dict: only saves parameters of the given model
    2) Saving with pickle: actually saves python pickle of the given model

    This source code saves model with 2) by default
'''

model_name = f"{model_type}_{dataset_name}_{quant_type}_{int(prune_amount*100)}.pth"
model_path = os.path.join(os.curdir, 'torch_models')

if __name__ == '__main__':
    # saving the model
    torch.save(model_quantized.state_dict(), os.path.join(model_path, model_name))
    # torch.save(model, os.path.join(model_path, model_name))
    print('model saved')