#!/usr/bin/env python
# coding: utf-8

# Imports for checking if the srcipt actually needs to run
import json
import os
import gzip


# In[ ]:


# External arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--finn_json")
args = parser.parse_args()
path_to_json = str(args.finn_json)


# In[ ]:


finn_json_name = path_to_json.split("/")[-1]
print(f"Loading data from: {finn_json_name}")

with gzip.open(path_to_json) as json_file:
    finn_data_raw = json.load(json_file)

# check that the version is compatible
compatible_scan_script_versions = ['0.4a', '0.4b', '0.5b', '0.6b', '0.7b', '0.8b', '1.0']
last_key = list(finn_data_raw.keys())[-1]
scan_script_version = finn_data_raw[last_key]['scan_script_version']
if scan_script_version in compatible_scan_script_versions:
    print(f"Compatible Scan script version {scan_script_version} found.")
else:
    print(f"Incompatible Scan script version {scan_script_version} found. Exiting.")
    exit(1)


# check that the run was actually successfull
if scan_script_version in ['0.5b', '0.6b', '0.7b', '0.8b', '1.0']:
    if finn_data_raw['PPR optimization successfull'] == False:
        print(f"FINN run wasn't succesfull, exiting here.")
        exit()
else:
    if finn_data_raw['Finished PPR optimization'] == False:
        print(f"FINN run wasn't succesfull, exiting here.")
        exit()

# Get "best" FINN data
finn_data = finn_data_raw[str(finn_data_raw['largest PPRing max_LUT'])]

# In[ ]:


# Parameters

# Version
script_version = "0.3"

# Parameters from JSON
pruning_mode = finn_data["pruning_mode"]
target_pruning = finn_data["pruning_ratio"]
WEIGHT_BIT_WIDTH = finn_data["wbits"]
ACT_BIT_WIDTH = finn_data["abits"]
finn_folding = finn_data["folding"]
folding = list(finn_folding.values())

# Misc. parameters
output_json_folder = "finn_result_jsons/after_training/"
folder_creation_tries = 20

# Training settings
pruning_steps = 0.1
minor_pruning_epochs = 50
#minor_pruning_epochs = 2
target_pruning_epochs = 300
#target_pruning_epochs = 3
pruning_norm = 1

# Trainer arguments arguments
network = "cnv"
experiments = "."
datadir = "./data/"
batch_size = 100
num_workers = 6
lr = 0.02
weight_decay = 0
random_seed = 1
log_freq = 10



# CNV parameters
# Likely always static
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3


# In[ ]:
# check if the output file already exists
if os.path.isfile(output_json_folder + finn_json_name):
    print(f"Output file {output_json_folder + finn_json_name} already exists, exiting here.")
    exit()


# Imports for torch
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import random

# In[ ]:





# In[ ]:





# In[ ]:


from dependencies import value

from brevitas.inject import BaseInjector as Injector
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType


class CommonQuant(Injector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant):
    scaling_const = 1.0


class CommonActQuant(CommonQuant):
    min_val = -1.0
    max_val = 1.0

    


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.init as init


class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return ((x - self.running_mean) / (self.running_var + self.eps).pow(0.5)) * self.weight + self.bias


# In[ ]:


import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.core.restrict_val import RestrictValueType
#from .tensor_norm import TensorNorm
#from .common import CommonWeightQuant, CommonActQuant


# In[ ]:


class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(QuantConv2d(
                kernel_size=KERNEL_SIZE,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))

        self.linear_features.append(QuantLinear(
            in_features=LAST_FC_IN_FEATURES,
            out_features=num_classes,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


# In[ ]:


#from torch.prune import BasePruningMethod
from torch.nn.utils.prune import BasePruningMethod
from torch.nn.utils.prune import _validate_pruning_amount_init, _validate_structured_pruning, _validate_pruning_dim, _compute_nparams_toprune, _validate_pruning_amount

class CoarseSIMDStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the 
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, SIMD):
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.SIMD = SIMD

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to 
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning 
                iterations, that need to be respected after the new mask is 
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        
        # New additions
        # Number of FINN input channels (MW)
        n_channels = torch.flatten(t, start_dim=1).shape[1]
        
        #Validate that SIMD fits into the number of channels
        if not (n_channels % self.SIMD == 0):
            raise ValueError(f"n_channels={n_channels} must be divisible by SIMD={self.SIMD}")
        
        # Number of channels when seen as one unit of SIMD
        n_SIMD_channels = n_channels // self.SIMD
        
        # Compute number of units to prune: amount if int,
        # else amount * n_channels
        nparams_toprune = _compute_nparams_toprune(self.amount, n_SIMD_channels)
        nparams_tokeep = n_SIMD_channels - nparams_toprune
        
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, n_SIMD_channels)
        
        # Make the mask
        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
            return mask
        # generate a random number in [0, 1] to associate to each channel
        prob = torch.rand(n_SIMD_channels)
        # generate mask for each channel by 0ing out the channels that
        # got assigned the k = nchannels_toprune lowest values in prob
        threshold = torch.kthvalue(prob, k=nparams_toprune).values
        
        # Create a mask, which matches the Coarse grain pruning scheme
        mask = torch.zeros_like(t)
        flat_mask = torch.flatten(mask, start_dim=1)
        for i in range(n_SIMD_channels):
            if prob[i] > threshold:
                flat_mask[:, i*self.SIMD : i*self.SIMD + self.SIMD] = 1
        
        
        
        # apply the new structured mask on top of prior (potentially 
        # unstructured) mask
        mask *= default_mask.to(dtype=mask.dtype)
        return mask
        
    @classmethod
    def apply(cls, module, name, amount, SIMD):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the 
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super(CoarseSIMDStructured, cls).apply(
            module, name, SIMD=SIMD, amount=amount
        )


# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Function

class squared_hinge_loss(Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets) 
        output = 1.-predictions.mul(targets)
        output[output.le(0.)] = 0.
        loss = torch.mean(output.mul(output))
        return loss 

    @staticmethod
    def backward(ctx, grad_output):
       predictions, targets = ctx.saved_tensors
       output=1.-predictions.mul(targets)
       output[output.le(0.)]=0.
       grad_output.resize_as_(predictions).copy_(targets).mul_(-2.).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(predictions.numel())
       return grad_output, None    

class SqrHingeLoss(nn.Module):
    # Squared Hinge Loss
    def __init__(self):
        super(SqrHingeLoss, self).__init__()
    
    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)


# In[ ]:


import logging
import sys
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingEpochMeters(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()


class EvalEpochMeters(object):
    def __init__(self):
        self.model_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()


class Logger(object):

    def __init__(self, output_dir_path, dry_run):
        self.output_dir_path = output_dir_path
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)

        # Stout logging
        out_hdlr = logging.StreamHandler(sys.stdout)
        out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        out_hdlr.setLevel(logging.INFO)
        self.log.addHandler(out_hdlr)

        # Txt logging
        if not dry_run:
            file_hdlr = logging.FileHandler(os.path.join(self.output_dir_path, 'log.txt'))
            file_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            file_hdlr.setLevel(logging.INFO)
            self.log.addHandler(file_hdlr)
            self.log.propagate = False

    def info(self, arg):
        self.log.info(arg)

    def eval_batch_cli_log(self, epoch_meters, batch, tot_batches):
        self.info('Test: [{0}/{1}]\t'
                  'Model Time {model_time.val:.3f} ({model_time.avg:.3f})\t'
                  'Loss Time {loss_time.val:.3f} ({loss_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  .format(batch, tot_batches,
                          model_time=epoch_meters.model_time,
                          loss_time=epoch_meters.loss_time,
                          loss=epoch_meters.losses,
                          top1=epoch_meters.top1,
                          top5=epoch_meters.top5))

    def training_batch_cli_log(self, epoch_meters, epoch, batch, tot_batches):
        self.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                         .format(epoch, batch, tot_batches,
                                 batch_time=epoch_meters.batch_time,
                                 data_time=epoch_meters.data_time,
                                 loss=epoch_meters.losses,
                                 top1=epoch_meters.top1,
                                 top5=epoch_meters.top5))


# In[ ]:


import random
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[ ]:


class LnCoarseSIMDStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the 
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount, SIMD, n):
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.SIMD = SIMD
        self.n = n

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to 
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning 
                iterations, that need to be respected after the new mask is 
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        
        # New additions
        # Number of FINN input channels (MW)
        n_channels = torch.flatten(t, start_dim=1).shape[1]
        
        #Validate that SIMD fits into the number of channels
        if not (n_channels % self.SIMD == 0):
            raise ValueError(f"n_channels={n_channels} must be divisible by SIMD={self.SIMD}")
        
        # Number of channels when seen as one unit of SIMD
        n_SIMD_channels = n_channels // self.SIMD
        
        # Compute number of units to prune: amount if int,
        # else amount * n_channels
        nparams_toprune = _compute_nparams_toprune(self.amount, n_SIMD_channels)
        nparams_tokeep = n_SIMD_channels - nparams_toprune
        
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, n_SIMD_channels)
        
        # Make the mask
        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
            return mask
        # generate a random number in [0, 1] to associate to each channel
        prob = torch.rand(n_SIMD_channels)
        # generate mask for each channel by 0ing out the channels that
        # got assigned the k = nchannels_toprune lowest values in prob
        threshold = torch.kthvalue(prob, k=nparams_toprune).values
        
        # Calculate the requested N norm for the block size
        flat_t = torch.flatten(t, start_dim=1)
        norms_of_blocks = []
        nill_counter = 0
        for i in range(n_SIMD_channels):
            block = flat_t[:, i*self.SIMD : i*self.SIMD + self.SIMD]
            norm = torch.norm(block, p=self.n)
            norms_of_blocks.append(norm)
            
            if torch.isclose(norm, torch.tensor([0.], device=device))[0]:
                nill_counter += 1
        norms_of_blocks = torch.tensor(norms_of_blocks)
        threshold = torch.kthvalue(norms_of_blocks, k=nparams_toprune).values
        
        # Create a mask, which matches the Coarse grain pruning scheme
        mask = torch.zeros_like(t)
        flat_mask = torch.flatten(mask, start_dim=1)
        # set values of the pruning mask
        non_nill_counter = 0
        for i in range(n_SIMD_channels):
            if norms_of_blocks[i] > threshold:
                flat_mask[:, i*self.SIMD : i*self.SIMD + self.SIMD] = 1
                non_nill_counter += 1
                if non_nill_counter > nparams_tokeep:
                    raise RuntimeWarning("Missmatch between L1 threshold and number of prameters to keep.")
                    break
        
        msg = f"Num blocks already pruned: {nill_counter};\tpercentage: {100*nill_counter/n_SIMD_channels:.1f} [%]"
        print(msg)
        
        msg = f"Num blocks pruned now pruned: {n_SIMD_channels-non_nill_counter};\tpercentage: {100*(n_SIMD_channels-non_nill_counter)/n_SIMD_channels:.1f} [%]"
        print(msg)
        
        # apply the new structured mask on top of prior (potentially 
        # unstructured) mask
        mask *= default_mask.to(dtype=mask.dtype)
        return mask
        
    @classmethod
    def apply(cls, module, name, amount, SIMD, n):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the 
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super(LnCoarseSIMDStructured, cls).apply(
            module, name, SIMD=SIMD, amount=amount, n=n,
        )


# In[ ]:


class LnFineSIMDStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): CHANGE ME!
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
        enforce_FINN_compatibility  (bool, optional): Check SIMD values for compatibility with FINN.
            In particular enforce that SIMD_in % SIMD_out == 0
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount, SIMD_in, SIMD_out, n, enforce_FINN_compatibility=True):
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.SIMD_in = SIMD_in
        self.SIMD_out = SIMD_out
        self.n = n
        self.enforce_FINN_compatibility = enforce_FINN_compatibility

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to 
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning 
                iterations, that need to be respected after the new mask is 
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        
        # New additions
        # Number of FINN input channels (MW)
        n_channels = torch.flatten(t, start_dim=1).shape[1]
        
        # Verify that SIMD values are compatible
        if not (n_channels >= self.SIMD_in):
            raise ValueError(f"n_channels={n_channels} must be larger or equal to SIMD_in={self.SIMD_in}")
        if not (n_channels >= self.SIMD_out):
            raise ValueError(f"n_channels={n_channels} must be larger or equal to SIMD_out={self.SIMD_out}")
        if not (self.SIMD_in >= self.SIMD_out):
            raise ValueError(f"SIMD_in={self.SIMD_in} must be larger or equal to SIMD_out={self.SIMD_out}")
        
        # Validate that SIMD fits into the number of channels
        if not (n_channels % self.SIMD_in == 0):
            raise ValueError(f"n_channels={n_channels} must be divisible by SIMD_in={self.SIMD_in}")
        if self.enforce_FINN_compatibility:
            if not (n_channels % self.SIMD_out == 0):
                raise ValueError(f"n_channels={n_channels} must be divisible by SIMD_out={self.SIMD_out}")
                
        
        # Number of channel blocks when seen as units of one SIMD width
        n_SIMD_blocks = n_channels // self.SIMD_in
        
        # Compute number of units to prune: amount if int,
        # else amount * n_channels
        nparams_toprune = self.SIMD_in - self.SIMD_out
        nparams_tokeep = self.SIMD_in - nparams_toprune
                
        # Make the mask
        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
            return mask
        
        # Calculate the norm for each channel
        flat_t = torch.flatten(t, start_dim=1)
        norms_of_channels = []
        nill_counter = 0
        for i in range(n_channels):
            channel = flat_t[:, i : i+1]
            norm = torch.norm(channel, p=self.n)
            norms_of_channels.append(norm)
            
            if torch.isclose(norm, torch.tensor([0.], device=device))[0]:
                nill_counter += 1
        norms_of_channels = torch.tensor(norms_of_channels)
        
        # Create a mask, which matches the Fine grain pruning scheme
        mask = torch.zeros_like(t)
        flat_mask = torch.flatten(mask, start_dim=1)
        # set values of the pruning mask
        non_nill_counter = 0
        for i in range(n_SIMD_blocks):
            # In each block we check where the norm is the lowest and prune those channels
            block_norms = norms_of_channels[i*self.SIMD_in : i*self.SIMD_in + self.SIMD_in]
            threshold = torch.kthvalue(block_norms, k=nparams_toprune).values
            for j in range(self.SIMD_in):
                if block_norms[j] > threshold:
                    flat_mask[:, i*self.SIMD_in + j] = 1
                    non_nill_counter += 1
        
        # Check mask for obvious errors
        ok = 0
        nok = 0
        for k in range(flat_mask.shape[1]//self.SIMD_in):
            mask_channel_sum = torch.sum(flat_mask[0][k*self.SIMD_in: k*self.SIMD_in+self.SIMD_in])
            if torch.isclose(mask_channel_sum, torch.tensor([float(self.SIMD_out)], device=mask_channel_sum.device))[0]:
                ok += 1
            else:
                nok += 1
                raise RuntimeWarning("Missmatch SIMD_out and sum of mask summation, there is likely an error in the mask creation.")
        
        print(f"Test result: ok: {ok}, nok: {nok}")
        
        msg = f"Num cols already pruned: {nill_counter};\tpercentage: {100*nill_counter/n_channels:.1f} [%]"
        print(msg)
        
        msg = f"Num cols pruned now pruned: {n_channels-non_nill_counter};\tpercentage: {100*(n_channels-non_nill_counter)/n_channels:.1f} [%]"
        print(msg)
        
        # apply the new structured mask on top of prior (potentially 
        # unstructured) mask
        mask *= default_mask.to(dtype=mask.dtype)
        return mask
    
    
    @classmethod
    def apply(cls, module, name, SIMD_in, SIMD_out, n, enforce_FINN_compatibility):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the 
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super(LnFineSIMDStructured, cls).apply(
            module, name, SIMD_in=SIMD_in, SIMD_out=SIMD_out, n=n, enforce_FINN_compatibility=enforce_FINN_compatibility, amount=1-(SIMD_out/SIMD_in),
        )


# In[ ]:


def make_brevitas_pruning_permanent(model):
    pruned_parameter = 'weight'
    # Make pruning permanent
    for brev_module in model.conv_features:
        if isinstance(brev_module, QuantConv2d):
            # Check that there even was pruning in the first place
            buffer_names = [name for name, buffer in brev_module.named_buffers()]
            if not (pruned_parameter+'_mask') in buffer_names:
                continue
            prune.remove(brev_module, pruned_parameter)
    return model

def prune_brevitas_model(model, amount, prune_class, folding, n=1):
    if prune_class == LnCoarseSIMDStructured:
        prune_kwargs = {"name":"weight", "amount":amount, 'n':n}
        SIMD_list = [fold[1] for fold in folding]
        i = 0
        for brev_module in model.conv_features:
            if isinstance(brev_module, QuantConv2d):
                # Do the prune
                #print(brev_module.weight.shape)
                prune_kwargs["SIMD"] = SIMD_list[i]
                prune_class.apply(brev_module, **prune_kwargs)
                i += 1
        return model
    elif prune_class == LnFineSIMDStructured:
        prune_kwargs = {"name":"weight", "n":1, "enforce_FINN_compatibility": False}
        SIMD_in_list = [fold[1] for fold in folding]
        SIMD_out_list = []
        for SIMD_in in SIMD_in_list:
            SIMD_out = int(SIMD_in*(1-amount))
            #print(SIMD_in, 1-amount, SIMD_out)
            if SIMD_out == 0:
                SIMD_out = 1
            SIMD_out_list.append(SIMD_out)
        i = 0
        for j in range(len(model.conv_features)):
            brev_module = model.conv_features[j]
            if isinstance(brev_module, QuantConv2d):
                # Do the prune
                # Skip the first layer, it is too coarse
                if i==0:
                    i += 1 
                    continue
                #print(f"Pruning conv layer number: {j}")
                #print(brev_module.weight.shape)
                prune_kwargs["SIMD_in"] = SIMD_in_list[i]
                prune_kwargs["SIMD_out"] = SIMD_out_list[i]
                prune_class.apply(brev_module, **prune_kwargs)
                i += 1
        return model
    else:
        raise RuntimeError("Pruning method not supported!")


# In[ ]:


# Set pruning class
if not pruning_mode == None:
    prune_mode_dict = {
        "coarse": LnCoarseSIMDStructured,
        "fine": LnFineSIMDStructured,
    }
    try:
        prune_class = prune_mode_dict[pruning_mode]
    except KeyError:
        print(f"Pruning mode: {pruning_mode} not supported")
else:
    prune_class = "None"

# In[ ]:


# Create the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNV(10, WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, 8, 3).to(device=device)

prune_name = str(prune_class).split(".")[-1].split("'")[0]

prec_name = "_{}W{}A_prune_method_{}".format(WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, prune_name)

while(folder_creation_tries > 0):
    try:
        experiment_name = '{}{}_{}'.format(network, prec_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
        output_dir_path = os.path.join(experiments, experiment_name)

        checkpoints_dir_path = os.path.join(output_dir_path, 'checkpoints')
        os.mkdir(output_dir_path)
        os.mkdir(checkpoints_dir_path)
        break
    except FileExistsError: 
        back_off_time = random.randint(1, 10)
        print(f"Directory already exists, backing off for {back_off_time}")
        time.sleep(back_off_time)
        folder_creation_tries -= 1

logger = Logger(output_dir_path, False)

# Randomness
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Datasets
transform_to_tensor = transforms.Compose([transforms.ToTensor()])

dataset = 'CIFAR10'
num_classes = 10
if dataset == 'CIFAR10':
    train_transforms_list = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor()]
    transform_train = transforms.Compose(train_transforms_list)
    builder = CIFAR10

elif dataset == 'MNIST':
    transform_train = transform_to_tensor
    builder = MNIST
else:
    raise Exception("Dataset not supported: {}".format(dataset))

train_set = builder(root=datadir,
                    train=True,
                    download=True,
                    transform=transform_train)
test_set = builder(root=datadir,
                   train=False,
                   download=True,
                   transform=transform_to_tensor)
train_loader = DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

# Init starting values
starting_epoch = 1
best_val_acc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

# Loss function
criterion = SqrHingeLoss()
criterion = criterion.to(device=device)

# Init optimizer
optimizer = optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)

# LR scheduler
scheduler = None

print(f"output_dir_path={output_dir_path}, checkpoints_dir_path={checkpoints_dir_path}")


# In[ ]:


def eval_model(epoch=None):
    eval_meters = EvalEpochMeters()

    # switch to evaluate mode
    model.eval()
    criterion.eval()
    save_data_list = []
    
    for i, data in enumerate(test_loader):

        end = time.time()
        (input, target) = data

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # for hingeloss only
        if isinstance(criterion, SqrHingeLoss):        
            target=target.unsqueeze(1)
            target_onehot = torch.Tensor(target.size(0), num_classes).to(device, non_blocking=True)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, target, 1)
            target=target.squeeze()
            target_var = target_onehot
        else:
            target_var = target

        # compute output
        output = model(input)

        # measure model elapsed time
        eval_meters.model_time.update(time.time() - end)
        end = time.time()

        #compute loss
        loss = criterion(output, target_var)
        eval_meters.loss_time.update(time.time() - end)

        pred = output.data.argmax(1, keepdim=True)
        correct = pred.eq(target.data.view_as(pred)).sum()
        prec1 = 100. * correct.float() / input.size(0)

        _, prec5 = accuracy(output, target, topk=(1, 5))
        eval_meters.losses.update(loss.item(), input.size(0))
        eval_meters.top1.update(prec1.item(), input.size(0))
        eval_meters.top5.update(prec5.item(), input.size(0))

        #Eval batch ends
        logger.eval_batch_cli_log(eval_meters, i, len(test_loader))
        
        # Compile save data
        save_data = [time.time(), epoch, loss.item(), prec1.item(), prec5.item()]
        save_data_list.append(save_data)

    return eval_meters.top1.avg, save_data_list

def checkpoint_best(epoch, name):
        best_path = os.path.join(checkpoints_dir_path, name)
        logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': best_val_acc,
        }, best_path)


# In[ ]:


# Make pruning list
import numpy as np
pruning_amount_list = list(np.arange(0., target_pruning, pruning_steps))
pruning_amount_list.append(target_pruning)
pruning_amount_list = list(sorted(pruning_amount_list))
pruning_amount_list, len(pruning_amount_list)


# In[ ]:


# Main training loop
epoch_data = {
    "keys": ["time", "epoch", "loss", "prec1", "prec5"]
}

for p_index, pruning_amount in enumerate(pruning_amount_list):
    # Maintain data structures
    epoch_data[pruning_amount] = {
        "train": [],
        "test": []
    }
    # Apply current pruning
    if pruning_amount == None:
        logger.info(f"Not applying pruning for pruning_amount: {pruning_amount}")
    else:
        model = prune_brevitas_model(model, pruning_amount, prune_class, folding, n=pruning_norm)
        logger.info(f"Set Pruning amount to: {pruning_amount}")
    # Reset LR
    if scheduler is not None:
        scheduler.step(0)
    else:
        optimizer.param_groups[0]['lr'] = lr
        logger.info(f"Reset LR to: {lr}")
    # Reset best val
    best_val_acc = 0
        
    # Set epochs
    if p_index == (len(pruning_amount_list)-1):
        epochs = target_pruning_epochs
    elif p_index == 0:
        epochs = 2* minor_pruning_epochs
    else:
        epochs = minor_pruning_epochs
    logger.info(f"Training for {epochs} epochs.")

    for epoch in range(starting_epoch, epochs):
        # Set to training mode
        model.train()
        criterion.train()

        # Init metrics
        epoch_meters = TrainingEpochMeters()
        start_data_loading = time.time()


        for i, data in enumerate(train_loader):
            (input, target) = data
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # for hingeloss only
            if isinstance(criterion, SqrHingeLoss):
                target=target.unsqueeze(1)
                target_onehot = torch.Tensor(target.size(0), num_classes).to(device, non_blocking=True)
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target, 1)
                target=target.squeeze()
                target_var = target_onehot
            else:
                target_var = target

            # measure data loading time
            epoch_meters.data_time.update(time.time() - start_data_loading)

            # Training batch starts
            start_batch = time.time()
            output = model(input)
            loss = criterion(output, target_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.clip_weights(-1,1)

            # measure elapsed time
            epoch_meters.batch_time.update(time.time() - start_batch)

            if i % int(log_freq) == 0 or i == len(train_loader) - 1:
                prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                epoch_meters.losses.update(loss.item(), input.size(0))
                epoch_meters.top1.update(prec1.item(), input.size(0))
                epoch_meters.top5.update(prec5.item(), input.size(0))
                logger.training_batch_cli_log(epoch_meters, epoch, i, len(train_loader))
                save_data = [time.time(), epoch, loss.item(), prec1.item(), prec5.item()]
                epoch_data[pruning_amount]["train"].append(save_data)
                

            # training batch ends
            start_data_loading = time.time()

        # Set the learning rate
        if scheduler is not None:
            scheduler.step(epoch)
        else:
            # Set the learning rate
            if epoch%40==0:
                optimizer.param_groups[0]['lr'] *= 0.5

        # Perform eval
        with torch.no_grad():
            top1avg, save_data_list = eval_model(epoch)
        epoch_data[pruning_amount]["test"].extend(save_data_list)

        # checkpoint
        # Skip the actual saving as it uses up too much data
        if top1avg >= best_val_acc:
            best_val_acc = top1avg
            #checkpoint_best(epoch, f"best_pruning_amount-{pruning_amount:.3f}.tar")
        else:
            pass
            #checkpoint_best(epoch, f"checkpoint_pruning_amount-{pruning_amount:.3f}.tar")
    # save the final model
    if pruning_amount == None:
        checkpoint_best(epoch, f"checkpoint_final_pruning_amount-None.tar")
    else:
        checkpoint_best(epoch, f"checkpoint_final_pruning_amount-{pruning_amount:.3f}.tar")
    # make pruning permanent and save again
    model = make_brevitas_pruning_permanent(model)
    #checkpoint_best(epoch, f"checkpoint_final_permanently_pruned_pruning_amount-{pruning_amount:.3f}.tar")


# In[ ]:


# Compile training results
brevitas_data = {
    "pruning_settings_epoch_data": epoch_data,
    "output_dir_path": output_dir_path,
    "script_version": script_version,
    "prune_class": str(prune_class).split(".")[-1].split("'")[0],
    "pruning_amount_list": pruning_amount_list,
}

# Append to FINN data and save to disk
logger.info(f"Saving combined FINN and Brevitas results to: {output_json_folder + finn_json_name}")
finn_data["brevitas_data"] = brevitas_data
with gzip.open(output_json_folder + finn_json_name, 'wt') as outfile:
    json.dump(finn_data, outfile)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




