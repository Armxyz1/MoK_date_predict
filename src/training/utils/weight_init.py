"""
Weight initialization utilities for neural networks.

This module provides functions to initialize weights of Conv2d and Linear layers
using various initialization strategies.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


def initialize_weights(
    module: nn.Module,
    method: str = 'kaiming_normal',
    activation: str = 'relu',
    bias_init: str = 'zeros',
    bias_value: float = 0.0,
    gain: float = 1.0,
    mean: float = 0.0,
    std: float = 0.01,
    a: float = 0.0,
    b: float = 1.0,
    mode: str = 'fan_in',
    verbose: bool = False
) -> None:
    """
    Initialize weights of a neural network module.

    Args:
        module: Neural network module (or entire model) to initialize
        method: Initialization method. Options:
            - 'kaiming_normal': He normal initialization (recommended for ReLU)
            - 'kaiming_uniform': He uniform initialization
            - 'xavier_normal': Glorot normal (recommended for sigmoid/tanh)
            - 'xavier_uniform': Glorot uniform
            - 'orthogonal': Orthogonal initialization (good for RNNs, very deep nets)
            - 'normal': Normal distribution N(mean, std)
            - 'uniform': Uniform distribution U(a, b)
            - 'constant': Constant value
            - 'zeros': Initialize with zeros
            - 'ones': Initialize with ones
            - 'default': Use PyTorch's default initialization
        activation: Activation function type ('relu', 'leaky_relu', 'tanh', 'sigmoid')
            Only used for kaiming_normal and kaiming_uniform
        bias_init: Bias initialization method. Options:
            - 'zeros': Initialize bias to zero (most common)
            - 'constant': Initialize to bias_value
            - 'uniform': Uniform distribution
            - 'normal': Normal distribution
        bias_value: Value for constant bias initialization
        gain: Scaling factor for xavier and orthogonal initialization
        mean: Mean for normal distribution
        std: Standard deviation for normal distribution
        a: Lower bound for uniform distribution (or negative slope for leaky_relu)
        b: Upper bound for uniform distribution
        mode: 'fan_in' or 'fan_out' for kaiming initialization
        verbose: If True, print initialization details

    Example:
        >>> model = MoK_CNN_Predictor()
        >>> initialize_weights(model, method='kaiming_normal', activation='relu', verbose=True)
    """

    if verbose:
        print(f"\n{'='*80}")
        print(f"Weight Initialization")
        print(f"{'='*80}")
        print(f"  Method: {method}")
        print(f"  Activation: {activation}")
        print(f"  Bias init: {bias_init}")
        if method in ['kaiming_normal', 'kaiming_uniform']:
            print(f"  Mode: {mode}")
        if method in ['xavier_normal', 'xavier_uniform', 'orthogonal']:
            print(f"  Gain: {gain}")
        if method == 'normal':
            print(f"  Mean: {mean}, Std: {std}")
        if method == 'uniform':
            print(f"  Range: [{a}, {b}]")
        print(f"{'='*80}\n")

    conv_count = 0
    linear_count = 0
    bn_count = 0

    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_count += 1
            _initialize_conv_or_linear(m, method, activation, gain, mean, std, a, b, mode)
            _initialize_bias(m, bias_init, bias_value, mean, std, a, b)

        elif isinstance(m, nn.Linear):
            linear_count += 1
            _initialize_conv_or_linear(m, method, activation, gain, mean, std, a, b, mode)
            _initialize_bias(m, bias_init, bias_value, mean, std, a, b)

        elif isinstance(m, nn.BatchNorm2d):
            bn_count += 1
            # BatchNorm: weight (gamma) = 1, bias (beta) = 0
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    if verbose:
        print(f"✓ Initialized {conv_count} Conv2d layers")
        print(f"✓ Initialized {linear_count} Linear layers")
        print(f"✓ Initialized {bn_count} BatchNorm2d layers (weight=1, bias=0)")


def _initialize_conv_or_linear(
    layer: nn.Module,
    method: str,
    activation: str,
    gain: float,
    mean: float,
    std: float,
    a: float,
    b: float,
    mode: str
) -> None:
    """Initialize weights of Conv2d or Linear layer."""

    if method == 'default':
        # Use PyTorch's default initialization (do nothing)
        return

    elif method == 'kaiming_normal':
        # Determine nonlinearity parameter
        nonlinearity = 'relu' if activation == 'relu' else 'leaky_relu'
        # For leaky_relu, 'a' is the negative slope
        if activation == 'leaky_relu':
            nn.init.kaiming_normal_(layer.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)

    elif method == 'kaiming_uniform':
        nonlinearity = 'relu' if activation == 'relu' else 'leaky_relu'
        if activation == 'leaky_relu':
            nn.init.kaiming_uniform_(layer.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity=nonlinearity)

    elif method == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight, gain=gain)

    elif method == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight, gain=gain)

    elif method == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=gain)

    elif method == 'normal':
        nn.init.normal_(layer.weight, mean=mean, std=std)

    elif method == 'uniform':
        nn.init.uniform_(layer.weight, a=a, b=b)

    elif method == 'constant':
        nn.init.constant_(layer.weight, mean)  # Use 'mean' as the constant value

    elif method == 'zeros':
        nn.init.zeros_(layer.weight)

    elif method == 'ones':
        nn.init.ones_(layer.weight)

    else:
        raise ValueError(
            f"Unknown initialization method: {method}. "
            f"Supported methods: 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', "
            f"'xavier_uniform', 'orthogonal', 'normal', 'uniform', 'constant', 'zeros', 'ones', 'default'"
        )


def _initialize_bias(
    layer: nn.Module,
    bias_init: str,
    bias_value: float,
    mean: float,
    std: float,
    a: float,
    b: float
) -> None:
    """Initialize bias of Conv2d or Linear layer."""

    if layer.bias is None:
        return

    if bias_init == 'zeros':
        nn.init.zeros_(layer.bias)

    elif bias_init == 'constant':
        nn.init.constant_(layer.bias, bias_value)

    elif bias_init == 'normal':
        nn.init.normal_(layer.bias, mean=mean, std=std)

    elif bias_init == 'uniform':
        nn.init.uniform_(layer.bias, a=a, b=b)

    else:
        raise ValueError(
            f"Unknown bias initialization method: {bias_init}. "
            f"Supported methods: 'zeros', 'constant', 'normal', 'uniform'"
        )


def get_recommended_init(activation: str) -> dict:
    """
    Get recommended initialization parameters for a given activation function.

    Args:
        activation: Activation function name

    Returns:
        Dictionary with recommended initialization parameters
    """
    recommendations = {
        'relu': {
            'method': 'kaiming_normal',
            'activation': 'relu',
            'bias_init': 'zeros',
            'mode': 'fan_out'
        },
        'leaky_relu': {
            'method': 'kaiming_normal',
            'activation': 'leaky_relu',
            'bias_init': 'zeros',
            'mode': 'fan_out',
            'a': 0.01  # negative slope
        },
        'sigmoid': {
            'method': 'xavier_uniform',
            'activation': 'sigmoid',
            'bias_init': 'zeros',
            'gain': 1.0
        },
        'tanh': {
            'method': 'xavier_uniform',
            'activation': 'tanh',
            'bias_init': 'zeros',
            'gain': 5.0/3.0  # Recommended gain for tanh
        },
        'elu': {
            'method': 'kaiming_normal',
            'activation': 'relu',  # ELU uses similar init to ReLU
            'bias_init': 'zeros',
            'mode': 'fan_in'
        },
        'gelu': {
            'method': 'xavier_uniform',
            'activation': 'relu',
            'bias_init': 'zeros',
            'gain': 1.0
        }
    }

    return recommendations.get(activation.lower(), recommendations['relu'])
