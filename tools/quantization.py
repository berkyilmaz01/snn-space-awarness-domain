#!/usr/bin/env python3
"""
FPGA Quantization Script for ZCU102 Deployment

Converts SpikeSEG model to 16-bit fixed-point format for AMD/Xilinx FPGA.

Requirements from meeting:
- 16-bit fixed point precision for all computations
- Temporal: n_timesteps=10, timestep_ms=100.0 (from config.yaml)
- Binary spikes (0,1) for inference
- Layer-by-layer computation
- Target: ZCU102 FPGA with 32Mbit BRAM

Fixed-Point Formats:
- Weights: Q0.16 UNSIGNED (STDP weights are always in [0, 1])
  Range: [0, 0.99998] with precision 2^-16 ≈ 0.000015
  Note: Using unsigned gives 2x precision vs signed Q1.15
  
- Membrane/Threshold/Leak: Q5.11 SIGNED
  Range: [-16.0, 15.999] with precision 2^-11 ≈ 0.00049
  Note: IGARSS 2023 uses threshold=10.0, leak=9.0, so need Q5.11 (not Q4.12)

Neuron Parameters (from checkpoint, NOT hardcoded):
- Threshold and leak are stored as registered buffers in checkpoint
- IGARSS 2023: threshold=10.0 for all layers
- Leak is ABSOLUTE value (e.g., 9.0), NOT the factor (0.9)
- leak = leak_factor * threshold

Output Files:
- weights_conv1.bin, weights_conv2.bin, weights_conv3.bin (binary, uint16)
- weights_conv1.h, weights_conv2.h, weights_conv3.h (C headers)
- neuron_params.h (thresholds, leaks from checkpoint)
- architecture_spec.txt (documentation for hardware team)

Usage:
    python scripts/quantize_for_fpga.py --checkpoint checkpoint_best.pt --output fpga_weights/
"""

import argparse
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Try to import torch, but allow running without it for documentation generation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features disabled.")


# =============================================================================
# FIXED-POINT CONFIGURATION
# =============================================================================

@dataclass
class FixedPointConfig:
    """Fixed-point format configuration.
    
    Weight Format:
    - STDP weights are initialized as Normal(0.8, 0.01) clamped to [0, 1]
    - Using Q0.16 UNSIGNED gives 2x precision vs Q1.15 signed
    - Precision: 1/65536 ≈ 0.000015
    
    Membrane/Threshold Format:
    - IGARSS 2023 uses threshold=10.0, leak=9.0 (90% of threshold)
    - Q4.12 allows ±8, but threshold=10 needs Q5.11 for safety
    - Using Q5.11 gives range [-16, 16) with precision 0.00049
    """
    total_bits: int = 16
    
    # Weights: Q0.16 UNSIGNED for STDP weights in [0, 1]
    # 2x precision vs signed Q1.15 since weights are always non-negative
    weight_int_bits: int = 0    # No integer bits for [0, 1) range
    weight_frac_bits: int = 16  # Full 16 bits fractional
    weight_unsigned: bool = True
    
    # Membrane/Threshold/Leak: Q5.11 format (values in [-16, 16))
    # IGARSS 2023: threshold=10.0, leak=9.0 requires >8 range
    membrane_int_bits: int = 5   # Allows values up to ±16
    membrane_frac_bits: int = 11
    
    # Accumulator: Q8.24 for intermediate calculations (32-bit)
    accum_int_bits: int = 8
    accum_frac_bits: int = 24
    
    def __post_init__(self):
        assert self.weight_int_bits + self.weight_frac_bits == self.total_bits
        assert self.membrane_int_bits + self.membrane_frac_bits == self.total_bits


# =============================================================================
# QUANTIZATION FUNCTIONS  
# =============================================================================

def float_to_fixed(value: float, int_bits: int, frac_bits: int, unsigned: bool = False) -> int:
    """
    Convert float to fixed-point integer representation.
    
    Args:
        value: Float value to convert
        int_bits: Number of integer bits (excluding sign for signed, total int for unsigned)
        frac_bits: Number of fractional bits
        unsigned: If True, use unsigned representation (for STDP weights in [0,1])
    
    Returns:
        Fixed-point integer
    """
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    
    if unsigned:
        # Unsigned: range [0, 2^total_bits - 1] / scale = [0, 2^int_bits)
        max_val = (2 ** total_bits - 1) / scale
        min_val = 0.0
        max_int = 2 ** total_bits - 1
        min_int = 0
    else:
        # Signed: range [-2^(total_bits-1), 2^(total_bits-1) - 1] / scale
        max_val = (2 ** (total_bits - 1) - 1) / scale
        min_val = -(2 ** (total_bits - 1)) / scale
        max_int = 2 ** (total_bits - 1) - 1
        min_int = -(2 ** (total_bits - 1))
    
    # Clamp to representable range
    clamped = max(min_val, min(max_val, value))
    
    # Convert to fixed-point
    fixed = int(round(clamped * scale))
    
    # Saturate to bit range (handles edge cases)
    fixed = max(min_int, min(max_int, fixed))
    
    return fixed


def fixed_to_float(fixed: int, int_bits: int, frac_bits: int, unsigned: bool = False) -> float:
    """Convert fixed-point integer back to float (for verification)."""
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    
    if not unsigned:
        # Handle sign extension for negative numbers (signed)
        if fixed >= 2 ** (total_bits - 1):
            fixed -= 2 ** total_bits
    
    return fixed / scale


def quantize_array(arr: np.ndarray, int_bits: int, frac_bits: int, 
                   unsigned: bool = False) -> np.ndarray:
    """Quantize numpy array to fixed-point using vectorized operations.
    
    Optimized version using numpy vectorization instead of Python loops.
    """
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    
    if unsigned:
        # Unsigned: [0, 2^total_bits - 1]
        max_int = 2 ** total_bits - 1
        min_int = 0
        dtype = np.uint16
    else:
        # Signed: [-2^(total_bits-1), 2^(total_bits-1) - 1]
        max_int = 2 ** (total_bits - 1) - 1
        min_int = -(2 ** (total_bits - 1))
        dtype = np.int16
    
    # Vectorized quantization (much faster than np.vectorize)
    scaled = np.round(arr * scale)
    clipped = np.clip(scaled, min_int, max_int)
    return clipped.astype(dtype)


def compute_quantization_error(original: np.ndarray, quantized: np.ndarray, 
                               int_bits: int, frac_bits: int,
                               unsigned: bool = False) -> Dict:
    """Compute quantization error statistics."""
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    
    # Vectorized dequantization
    if unsigned:
        dequantized = quantized.astype(np.float64) / scale
        overflow_threshold = 2 ** total_bits / scale
        overflow_count = int(np.sum(original >= overflow_threshold) + np.sum(original < 0))
    else:
        # Handle signed conversion
        dequantized = quantized.astype(np.float64) / scale
        overflow_threshold = 2 ** (total_bits - 1) / scale
        overflow_count = int(np.sum(np.abs(original) >= overflow_threshold))
    
    error = original - dequantized
    return {
        'max_abs_error': float(np.max(np.abs(error))),
        'mean_abs_error': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'max_original': float(np.max(original)),
        'min_original': float(np.min(original)),
        'overflow_count': overflow_count
    }


# =============================================================================
# MODEL LOADING AND WEIGHT EXTRACTION
# =============================================================================

def load_checkpoint(path: str) -> Dict:
    """Load PyTorch checkpoint."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required to load checkpoint")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    print(f"  Loading from: {path}")
    print(f"  File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Verify checkpoint structure
    required_keys = ['model_state_dict']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Missing required key in checkpoint: {key}")
    
    return checkpoint


def extract_weights(checkpoint: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """Extract weights and neuron parameters from checkpoint.
    
    Returns:
        Tuple of (weights_dict, neuron_params_dict)
        - weights_dict: layer_name -> weight array
        - neuron_params_dict: layer_name -> {'threshold': float, 'leak': float, 'beta': float}
    """
    state_dict = checkpoint['model_state_dict']
    
    weights = {}
    neuron_params = {}
    
    for key, tensor in state_dict.items():
        # Remove 'encoder.' prefix if present
        clean_key = key.replace('encoder.', '')
        arr = tensor.numpy()
        
        # Extract conv weights
        if 'conv.weight' in clean_key:
            weights[clean_key] = arr
            
        # Extract neuron parameters (threshold, leak, beta)
        elif '.neuron.' in clean_key:
            # Parse layer name and param name
            # e.g., "conv1.neuron.threshold" -> layer="conv1", param="threshold"
            parts = clean_key.split('.')
            layer_name = parts[0]
            param_name = parts[-1]
            
            if layer_name not in neuron_params:
                neuron_params[layer_name] = {}
            
            # Convert 0-d array to scalar
            neuron_params[layer_name][param_name] = float(arr)
    
    # Verify we found expected layers
    expected_layers = ['conv1', 'conv2', 'conv3']
    found_weight_layers = set()
    for key in weights.keys():
        if 'conv.weight' in key:
            found_weight_layers.add(key.split('.')[0])
    
    print(f"  Found weight layers: {sorted(found_weight_layers)}")
    print(f"  Found neuron params for: {sorted(neuron_params.keys())}")
    
    # Print actual neuron parameters (critical for verification)
    for layer, params in sorted(neuron_params.items()):
        print(f"    {layer}: threshold={params.get('threshold', 'N/A'):.4f}, "
              f"leak={params.get('leak', 'N/A'):.4f}, "
              f"beta={params.get('beta', 'N/A'):.4f}")
    
    missing = set(expected_layers) - found_weight_layers
    if missing:
        print(f"  WARNING: Missing expected layers: {missing}")
    
    return weights, neuron_params


# =============================================================================
# FPGA EXPORT FUNCTIONS
# =============================================================================

def export_weights_binary(weights: np.ndarray, path: str, config: FixedPointConfig):
    """Export quantized weights as binary file for FPGA.
    
    Binary format:
    - Header: magic (4 bytes) + version (4 bytes) + ndims (4 bytes) + shape (4*ndims bytes)
    - Data: uint16 or int16 depending on config.weight_unsigned
    """
    quantized = quantize_array(weights, config.weight_int_bits, config.weight_frac_bits,
                               unsigned=config.weight_unsigned)
    
    with open(path, 'wb') as f:
        # Write header with magic number and version for verification
        f.write(struct.pack('4s', b'SSWT'))  # Magic: SpikeSeg WeighTs
        f.write(struct.pack('I', 1))         # Version 1
        
        # Shape info
        shape = weights.shape
        f.write(struct.pack('I', len(shape)))  # Number of dimensions
        for dim in shape:
            f.write(struct.pack('I', dim))
        
        # Data type flag
        f.write(struct.pack('B', 1 if config.weight_unsigned else 0))  # 1=unsigned, 0=signed
        
        # Write data as uint16 (unsigned) or int16 (signed)
        if config.weight_unsigned:
            quantized.flatten().astype('<u2').tofile(f)  # Little-endian uint16
        else:
            quantized.flatten().astype('<i2').tofile(f)  # Little-endian int16
    
    return quantized


def export_weights_c_header(weights: np.ndarray, name: str, path: str, 
                            config: FixedPointConfig):
    """Export quantized weights as C header file."""
    quantized = quantize_array(weights, config.weight_int_bits, config.weight_frac_bits,
                               unsigned=config.weight_unsigned)
    flat = quantized.flatten()
    
    # Determine C type based on signed/unsigned
    c_type = "uint16_t" if config.weight_unsigned else "int16_t"
    sign_str = "UNSIGNED" if config.weight_unsigned else "SIGNED"
    
    with open(path, 'w') as f:
        f.write(f"// Auto-generated by quantize_for_fpga.py\n")
        f.write(f"// Fixed-point format: Q{config.weight_int_bits}.{config.weight_frac_bits} ({sign_str})\n")
        f.write(f"// STDP weights are always in [0, 1], so unsigned format gives 2x precision\n")
        f.write(f"// Original shape: {weights.shape}\n")
        f.write(f"// Total elements: {len(flat)}\n")
        f.write(f"// Weight range: [{weights.min():.4f}, {weights.max():.4f}]\n\n")
        
        f.write(f"#ifndef {name.upper()}_H\n")
        f.write(f"#define {name.upper()}_H\n\n")
        
        f.write(f"#include <stdint.h>\n\n")
        
        # Shape defines
        f.write(f"#define {name.upper()}_DIMS {len(weights.shape)}\n")
        dim_names = ['OUT_CH', 'IN_CH', 'KERNEL_H', 'KERNEL_W']
        for i, dim in enumerate(weights.shape):
            dim_name = dim_names[i] if i < len(dim_names) else f"DIM{i}"
            f.write(f"#define {name.upper()}_{dim_name} {dim}\n")
        f.write(f"#define {name.upper()}_SIZE {len(flat)}\n\n")
        
        # Quantization info
        f.write(f"// Quantization: multiply by {2**config.weight_frac_bits} to convert to fixed-point\n")
        f.write(f"// De-quantization: divide by {2**config.weight_frac_bits} to recover float\n\n")
        
        # Weight array
        f.write(f"static const {c_type} {name}[{len(flat)}] = {{\n")
        
        # Write in rows of 8
        for i in range(0, len(flat), 8):
            row = flat[i:i+8]
            row_str = ", ".join(f"{x:6d}" for x in row)
            f.write(f"    {row_str},\n")
        
        f.write(f"}};\n\n")
        f.write(f"#endif // {name.upper()}_H\n")
    
    return quantized


def export_neuron_params(params: Dict, path: str, config: FixedPointConfig):
    """Export neuron parameters (threshold, leak) as C header.
    
    IMPORTANT: The 'leak' value in the checkpoint is the ABSOLUTE leak value,
    NOT the leak factor. For example, if threshold=10.0 and leak_factor=0.9,
    then leak=9.0 (not 0.9).
    
    LIF equation: V = V + I - leak (subtractive mode)
    """
    with open(path, 'w') as f:
        f.write("// Auto-generated LIF Neuron Parameters for FPGA\n")
        f.write(f"// Fixed-point format for membrane: Q{config.membrane_int_bits}.{config.membrane_frac_bits}\n")
        f.write("//\n")
        f.write("// IMPORTANT: The leak value is the ABSOLUTE leak amount per timestep,\n")
        f.write("// NOT a fractional factor. For IGARSS 2023:\n")
        f.write("//   conv1: leak = 9.0 (90% of threshold 10.0)\n")
        f.write("//   conv2: leak = 1.0 (10% of threshold 10.0)\n")
        f.write("//   conv3: leak = 0.0 (no leak)\n")
        f.write("//\n")
        f.write("// LIF equation (subtractive mode):\n")
        f.write("//   V(t) = V(t-1) + I(t) - leak\n")
        f.write("//   V(t) = max(V(t), 0)  // Clamp\n")
        f.write("//   if V(t) >= threshold: spike, V(t) = 0\n")
        f.write("//\n\n")
        
        f.write("#ifndef NEURON_PARAMS_H\n")
        f.write("#define NEURON_PARAMS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write("// Fixed-point configuration\n")
        f.write(f"#define FP_WEIGHT_INT_BITS {config.weight_int_bits}\n")
        f.write(f"#define FP_WEIGHT_FRAC_BITS {config.weight_frac_bits}\n")
        f.write(f"#define FP_WEIGHT_UNSIGNED {1 if config.weight_unsigned else 0}\n")
        f.write(f"#define FP_MEMBRANE_INT_BITS {config.membrane_int_bits}\n")
        f.write(f"#define FP_MEMBRANE_FRAC_BITS {config.membrane_frac_bits}\n")
        f.write(f"#define FP_ACCUM_INT_BITS {config.accum_int_bits}\n")
        f.write(f"#define FP_ACCUM_FRAC_BITS {config.accum_frac_bits}\n\n")
        
        f.write("// Conversion macros\n")
        f.write(f"#define FLOAT_TO_MEMBRANE_FP(x) ((int16_t)((x) * {2**config.membrane_frac_bits}))\n")
        f.write(f"#define MEMBRANE_FP_TO_FLOAT(x) ((float)(x) / {2**config.membrane_frac_bits}.0f)\n\n")
        
        # Layer parameters (sorted for consistency)
        for layer_name, layer_params in sorted(params.items()):
            f.write(f"// {layer_name} parameters (from checkpoint)\n")
            
            threshold = layer_params.get('threshold', 10.0)
            leak = layer_params.get('leak', 0.0)
            
            # Signed fixed-point for membrane/threshold
            threshold_fp = float_to_fixed(threshold, config.membrane_int_bits, 
                                          config.membrane_frac_bits, unsigned=False)
            leak_fp = float_to_fixed(leak, config.membrane_int_bits, 
                                     config.membrane_frac_bits, unsigned=False)
            
            f.write(f"#define {layer_name.upper()}_THRESHOLD {threshold_fp}  // float: {threshold:.4f}\n")
            f.write(f"#define {layer_name.upper()}_LEAK {leak_fp}  // float: {leak:.4f} (ABSOLUTE value, not factor)\n")
            
            # Also output leak factor for documentation
            if threshold > 0:
                leak_factor = leak / threshold
                f.write(f"// Leak factor: {leak_factor:.2f} (i.e., {leak_factor*100:.0f}% of threshold)\n")
            f.write("\n")
        
        f.write("#endif // NEURON_PARAMS_H\n")


def export_architecture_spec(weights: Dict, config: FixedPointConfig, path: str,
                            neuron_params: Dict = None, layer_shapes: Dict = None):
    """Export detailed architecture specification for hardware team."""
    
    # Compute ranges based on config
    if config.weight_unsigned:
        weight_max = (2**config.total_bits - 1) / 2**config.weight_frac_bits
        weight_min = 0.0
        weight_sign = "UNSIGNED"
    else:
        weight_max = (2**(config.total_bits-1) - 1) / 2**config.weight_frac_bits
        weight_min = -2**(config.total_bits-1) / 2**config.weight_frac_bits
        weight_sign = "SIGNED"
    
    membrane_max = (2**(config.total_bits-1) - 1) / 2**config.membrane_frac_bits
    membrane_min = -2**(config.total_bits-1) / 2**config.membrane_frac_bits
    
    with open(path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SPIKESEG FPGA IMPLEMENTATION SPECIFICATION\n")
        f.write("Target: AMD/Xilinx ZCU102\n")
        f.write("Generated by quantize_for_fpga.py\n")
        f.write("=" * 70 + "\n\n")
        
        # Fixed-point format
        f.write("FIXED-POINT FORMATS\n")
        f.write("-" * 70 + "\n")
        f.write(f"""
Weights:    Q{config.weight_int_bits}.{config.weight_frac_bits} (16-bit {weight_sign})
            Range: [{weight_min:.5f}, {weight_max:.5f}]
            Precision: {2**-config.weight_frac_bits:.8f}
            Note: STDP weights are always in [0, 1], so unsigned gives 2x precision
            
Membrane/Threshold/Leak: Q{config.membrane_int_bits}.{config.membrane_frac_bits} (16-bit SIGNED)
            Range: [{membrane_min:.5f}, {membrane_max:.5f}]
            Precision: {2**-config.membrane_frac_bits:.6f}
            Note: IGARSS 2023 uses threshold=10.0, leak=9.0, so need Q5.11 minimum

Accumulator: Q{config.accum_int_bits}.{config.accum_frac_bits} (32-bit SIGNED)
            Used for intermediate convolution sums
            Must handle accumulation of many weights (use saturation for overflow)
            
Spikes:     1-bit binary (0 or 1)

""")
        
        # LIF Neuron equations
        f.write("LIF NEURON EQUATIONS (Fixed-Point)\n")
        f.write("-" * 70 + "\n")
        f.write("""
For each neuron at each timestep:

1. LEAK (Subtractive mode):
   membrane = membrane - leak
   // In fixed-point: membrane_fp -= leak_fp
   
2. INTEGRATE:
   membrane = membrane + weighted_input
   // weighted_input = sum of (spike[i] * weight[i]) for all inputs
   // Since spikes are binary: weighted_input = sum of weight[i] where spike[i]=1
   
3. CLAMP:
   if (membrane < 0) membrane = 0
   // In fixed-point: if (membrane_fp < 0) membrane_fp = 0
   
4. SPIKE:
   if (membrane >= threshold):
       spike_out = 1
       membrane = 0  // Reset
   else:
       spike_out = 0

Verilog Pseudocode:
-------------------
always @(posedge clk) begin
    if (rst) begin
        membrane <= 16'd0;
        spike_out <= 1'b0;
    end else begin
        // Step 1: Leak
        membrane_leaked = membrane - LEAK;
        
        // Step 2: Integrate (weighted_input computed elsewhere)
        membrane_integrated = membrane_leaked + weighted_input;
        
        // Step 3: Clamp to positive
        membrane_clamped = (membrane_integrated[15]) ? 16'd0 : membrane_integrated;
        
        // Step 4: Spike and reset
        if (membrane_clamped >= THRESHOLD) begin
            spike_out <= 1'b1;
            membrane <= 16'd0;
        end else begin
            spike_out <= 1'b0;
            membrane <= membrane_clamped;
        end
    end
end

""")
        
        # Convolution computation
        f.write("SPIKE-DRIVEN CONVOLUTION\n")
        f.write("-" * 70 + "\n")
        f.write("""
For spike-based inference, convolution simplifies to:

output[y][x] = sum over all (ky, kx, c_in) where input_spike[y+ky][x+kx][c_in] == 1:
               weight[c_out][c_in][ky][kx]

Key insight: Since input spikes are BINARY (0 or 1):
- If spike = 0: contribute nothing
- If spike = 1: contribute weight value

This means: NO MULTIPLICATIONS NEEDED!
Only additions of weights where input spike is 1.

Hardware Implementation:
- For each output position (y, x, c_out):
  - Initialize accumulator = 0
  - For each kernel position (ky, kx, c_in):
    - If input_spike[y+ky][x+kx][c_in] == 1:
      - accumulator += weight[c_out][c_in][ky][kx]
  - Pass accumulator to LIF neuron

""")
        
        # Layer specifications - use ACTUAL shapes from checkpoint
        f.write("LAYER SPECIFICATIONS (from checkpoint)\n")
        f.write("-" * 70 + "\n")
        
        # Build layer info from actual weight shapes
        conv_layers = []
        for key, w in weights.items():
            if 'conv.weight' in key:
                layer_name = key.split('.')[0]
                out_ch, in_ch, kh, kw = w.shape
                conv_layers.append((layer_name, in_ch, out_ch, kh, kw))
        
        # Sort by layer name
        conv_layers.sort(key=lambda x: x[0])
        
        for layer_name, in_ch, out_ch, kh, kw in conv_layers:
            f.write(f"\n{layer_name.upper()}:\n")
            f.write(f"  Type: Spiking Convolution + LIF\n")
            f.write(f"  Input channels: {in_ch} (FROM CHECKPOINT - verify this matches your config!)\n")
            f.write(f"  Output channels: {out_ch}\n")
            f.write(f"  Kernel size: {kh}x{kw}\n")
            f.write(f"  Weight count: {out_ch * in_ch * kh * kw}\n")
            f.write(f"  Weight memory: {out_ch * in_ch * kh * kw * 2} bytes (uint16)\n")
            
            # Add neuron params if available
            if neuron_params and layer_name in neuron_params:
                params = neuron_params[layer_name]
                threshold = params.get('threshold', 'N/A')
                leak = params.get('leak', 'N/A')
                f.write(f"  Threshold: {threshold} (ABSOLUTE value)\n")
                f.write(f"  Leak: {leak} (ABSOLUTE value, NOT factor!)\n")
                if isinstance(threshold, (int, float)) and threshold > 0 and isinstance(leak, (int, float)):
                    leak_factor = leak / threshold
                    f.write(f"  Leak as factor: {leak_factor:.2f} ({leak_factor*100:.0f}% of threshold)\n")
        
        # Add pooling layers (these don't have weights but are in the architecture)
        f.write(f"\nPOOL1:\n")
        f.write(f"  Type: Max Pooling (spike-preserving)\n")
        f.write(f"  Kernel: 2x2, Stride: 2\n")
        f.write(f"  Note: For spike pooling, OR operation across window\n")
        
        f.write(f"\nPOOL2:\n")
        f.write(f"  Type: Max Pooling (spike-preserving)\n")
        f.write(f"  Kernel: 2x2, Stride: 2\n")
        
        # Memory requirements
        f.write("\n\nMEMORY REQUIREMENTS\n")
        f.write("-" * 70 + "\n")
        
        total_params = 0
        for name, w in weights.items():
            if 'conv.weight' in name:
                params = np.prod(w.shape)
                total_params += params
                f.write(f"{name}: {w.shape} = {params} params = {params * 2} bytes\n")
        
        f.write(f"\nTotal weight parameters: {total_params}\n")
        f.write(f"Total weight memory: {total_params * 2} bytes ({total_params * 2 / 1024:.2f} KB)\n")
        f.write(f"BRAM available (ZCU102): 32 Mbits = 4 MB\n")
        f.write(f"Fits in BRAM: {'YES' if total_params * 2 < 4 * 1024 * 1024 else 'NO'}\n")
        
        # Temporal configuration (from config.yaml)
        f.write("\n\nTEMPORAL CONFIGURATION (from config.yaml)\n")
        f.write("-" * 70 + "\n")
        f.write("""
Timesteps (bins): 10 (n_timesteps from config.yaml)
Time per step: 100 ms (timestep_ms from config.yaml)  
Total window: 1000 ms (1 second)

Note: The original docstring incorrectly said "4 bins, 25ms each, 100ms total".
      Always verify against config.yaml for your specific training configuration.

Processing order (sequential - lower memory):
  For each temporal bin t = 0, 1, ..., 9:
    For each layer l = conv1, pool1, conv2, pool2, conv3:
      Process all spatial positions
      LIF neurons accumulate across bins (membrane persists)
      
Alternative (bin-parallel - higher throughput):
  For each layer:
    Process all 10 bins in parallel
    Requires 10x the membrane storage
    Better for pipelining

IMPORTANT: Membrane state persists across timesteps within a sample.
           Reset membrane to 0 at the START of each new sample/sequence.

""")


def export_verilog_lif(path: str, config: FixedPointConfig):
    """Export Verilog module for LIF neuron with saturation handling.
    
    Uses %-style formatting to avoid conflicts with Verilog's brace syntax
    for concatenation and replication. Braces { } are left as-is.
    """
    
    weight_sign = 'UNSIGNED' if config.weight_unsigned else 'SIGNED'
    
    # Calculate example values for testbench
    threshold_10_fp = int(10.0 * 2**config.membrane_frac_bits)
    leak_9_fp = int(9.0 * 2**config.membrane_frac_bits)
    input_10_fp = int(10.0 * 2**config.accum_frac_bits)
    weight_08_fp = int(0.8 * 2**config.weight_frac_bits)
    
    # Fixed-point ranges
    if config.weight_unsigned:
        weight_max = (2**config.total_bits - 1) / 2**config.weight_frac_bits
    else:
        weight_max = (2**(config.total_bits-1) - 1) / 2**config.weight_frac_bits
    membrane_min = -2**(config.total_bits-1) / 2**config.membrane_frac_bits
    membrane_max = (2**(config.total_bits-1) - 1) / 2**config.membrane_frac_bits
    
    # Use %-style formatting: %(name)s for strings, %(name)d for ints, %(name).Nf for floats
    # Verilog braces { } are left untouched
    verilog_template = r'''// =============================================================================
// LIF Neuron Module for SpikeSEG FPGA Implementation
// Auto-generated by quantize_for_fpga.py
// Target: AMD/Xilinx ZCU102
// =============================================================================
// 
// Fixed-point formats (from config):
//   - Weights: Q%(weight_int_bits)d.%(weight_frac_bits)d (%(weight_sign)s)
//   - Membrane, Threshold, Leak: Q%(membrane_int_bits)d.%(membrane_frac_bits)d (SIGNED)
//   - Weighted Input: Q%(accum_int_bits)d.%(accum_frac_bits)d (32-bit signed accumulator)
//   - Spikes: 1-bit binary
//
// LIF Equation (Subtractive mode):
//   V(t) = V(t-1) + I(t) - leak
//   V(t) = max(V(t), 0)           // Clamp to non-negative
//   V(t) = min(V(t), V_MAX)       // Saturate to prevent overflow
//   if V(t) >= threshold: spike = 1, V(t) = 0
//
// IMPORTANT: leak is ABSOLUTE value (e.g., 9.0), NOT decay factor (0.9)
//
// =============================================================================

module lif_neuron #(
    parameter MEMBRANE_WIDTH = 16,
    parameter ACCUM_WIDTH = 32,
    parameter MEMBRANE_FRAC_BITS = %(membrane_frac_bits)d,
    parameter ACCUM_FRAC_BITS = %(accum_frac_bits)d
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               enable,
    input  wire signed [ACCUM_WIDTH-1:0]      weighted_input,
    input  wire signed [MEMBRANE_WIDTH-1:0]   threshold,
    input  wire signed [MEMBRANE_WIDTH-1:0]   leak,
    output reg                                spike_out,
    output wire signed [MEMBRANE_WIDTH-1:0]   membrane_out
);

    // Constants for saturation
    localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MAX = {1'b0, {(MEMBRANE_WIDTH-1){1'b1}}};
    localparam signed [MEMBRANE_WIDTH-1:0] MEMBRANE_MIN = {MEMBRANE_WIDTH{1'b0}};
    
    // Membrane potential (persists across timesteps)
    reg signed [MEMBRANE_WIDTH-1:0] membrane;
    
    // Extended width for intermediate calculations
    localparam EXTENDED_WIDTH = MEMBRANE_WIDTH + 4;
    
    wire signed [EXTENDED_WIDTH-1:0] membrane_extended;
    wire signed [EXTENDED_WIDTH-1:0] leak_extended;
    wire signed [EXTENDED_WIDTH-1:0] input_scaled;
    wire signed [EXTENDED_WIDTH-1:0] membrane_after_ops;
    wire signed [MEMBRANE_WIDTH-1:0] membrane_saturated;
    wire threshold_exceeded;
    wire negative_overflow;
    wire positive_overflow;

    // Scale weighted_input from accumulator format to membrane format
    wire signed [ACCUM_WIDTH-1:0] input_shifted;
    assign input_shifted = weighted_input >>> (ACCUM_FRAC_BITS - MEMBRANE_FRAC_BITS);
    
    // Saturate input to extended width
    wire input_overflow_pos = (input_shifted[ACCUM_WIDTH-1] == 0) && (|input_shifted[ACCUM_WIDTH-2:EXTENDED_WIDTH-1]);
    wire input_overflow_neg = (input_shifted[ACCUM_WIDTH-1] == 1) && (~&input_shifted[ACCUM_WIDTH-2:EXTENDED_WIDTH-1]);
    
    assign input_scaled = input_overflow_pos ? {1'b0, {(EXTENDED_WIDTH-1){1'b1}}} :
                          input_overflow_neg ? {1'b1, {(EXTENDED_WIDTH-1){1'b0}}} :
                          input_shifted[EXTENDED_WIDTH-1:0];

    // Sign-extend membrane and leak
    assign membrane_extended = {{(EXTENDED_WIDTH-MEMBRANE_WIDTH){membrane[MEMBRANE_WIDTH-1]}}, membrane};
    assign leak_extended = {{(EXTENDED_WIDTH-MEMBRANE_WIDTH){leak[MEMBRANE_WIDTH-1]}}, leak};

    // LIF dynamics: V = V + I - leak
    assign membrane_after_ops = membrane_extended + input_scaled - leak_extended;

    // Detect overflow
    assign negative_overflow = membrane_after_ops[EXTENDED_WIDTH-1];
    assign positive_overflow = (~membrane_after_ops[EXTENDED_WIDTH-1]) && 
                               (|membrane_after_ops[EXTENDED_WIDTH-2:MEMBRANE_WIDTH-1]);

    // Saturate: clamp to [0, MEMBRANE_MAX]
    assign membrane_saturated = negative_overflow ? MEMBRANE_MIN :
                                positive_overflow ? MEMBRANE_MAX :
                                membrane_after_ops[MEMBRANE_WIDTH-1:0];

    assign threshold_exceeded = (membrane_saturated >= threshold);
    assign membrane_out = membrane;

    always @(posedge clk) begin
        if (rst) begin
            membrane <= {MEMBRANE_WIDTH{1'b0}};
            spike_out <= 1'b0;
        end 
        else if (enable) begin
            if (threshold_exceeded) begin
                spike_out <= 1'b1;
                membrane <= {MEMBRANE_WIDTH{1'b0}};
            end 
            else begin
                spike_out <= 1'b0;
                membrane <= membrane_saturated;
            end
        end
        else begin
            spike_out <= 1'b0;
        end
    end

endmodule


// =============================================================================
// Spike-Driven Convolution Unit
// =============================================================================
// For binary spikes: output = sum of weights where input_spike = 1
// No multiplications needed!

module spike_conv_unit #(
    parameter KERNEL_SIZE = 5,
    parameter INPUT_CHANNELS = 1,
    parameter WEIGHT_WIDTH = 16,
    parameter ACCUM_WIDTH = 32,
    parameter WEIGHT_UNSIGNED = 1
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [INPUT_CHANNELS*KERNEL_SIZE*KERNEL_SIZE-1:0] input_spikes,
    input  wire [WEIGHT_WIDTH-1:0] weights [0:INPUT_CHANNELS*KERNEL_SIZE*KERNEL_SIZE-1],
    output reg signed [ACCUM_WIDTH-1:0] result,
    output reg done,
    output reg overflow_flag
);

    localparam TOTAL_WEIGHTS = INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    localparam IDX_WIDTH = $clog2(TOTAL_WEIGHTS + 1);
    localparam signed [ACCUM_WIDTH-1:0] ACCUM_MAX = {1'b0, {(ACCUM_WIDTH-1){1'b1}}};
    
    reg [IDX_WIDTH-1:0] idx;
    reg signed [ACCUM_WIDTH-1:0] accumulator;
    
    localparam IDLE = 2'd0, ACCUMULATE = 2'd1, FINISH = 2'd2;
    reg [1:0] state;

    wire signed [ACCUM_WIDTH-1:0] weight_extended;
    generate
        if (WEIGHT_UNSIGNED)
            assign weight_extended = {{(ACCUM_WIDTH-WEIGHT_WIDTH){1'b0}}, weights[idx]};
        else
            assign weight_extended = {{(ACCUM_WIDTH-WEIGHT_WIDTH){weights[idx][WEIGHT_WIDTH-1]}}, weights[idx]};
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            accumulator <= 0;
            result <= 0;
            done <= 0;
            idx <= 0;
            overflow_flag <= 0;
        end
        else case (state)
            IDLE: begin
                done <= 0;
                overflow_flag <= 0;
                if (start) begin
                    accumulator <= 0;
                    idx <= 0;
                    state <= ACCUMULATE;
                end
            end
            ACCUMULATE: begin
                if (input_spikes[idx]) begin
                    if (accumulator > ACCUM_MAX - weight_extended) begin
                        accumulator <= ACCUM_MAX;
                        overflow_flag <= 1;
                    end else
                        accumulator <= accumulator + weight_extended;
                end
                if (idx == TOTAL_WEIGHTS - 1)
                    state <= FINISH;
                else
                    idx <= idx + 1;
            end
            FINISH: begin
                result <= accumulator;
                done <= 1;
                state <= IDLE;
            end
        endcase
    end

endmodule


// =============================================================================
// Testbench for LIF Neuron
// =============================================================================

`timescale 1ns / 1ps

module lif_neuron_tb;
    
    localparam MEMBRANE_FRAC_BITS = %(membrane_frac_bits)d;
    localparam ACCUM_FRAC_BITS = %(accum_frac_bits)d;
    
    reg clk, rst, enable;
    reg signed [31:0] weighted_input;
    reg signed [15:0] threshold, leak;
    wire spike_out;
    wire signed [15:0] membrane_out;
    
    lif_neuron #(.MEMBRANE_FRAC_BITS(MEMBRANE_FRAC_BITS), .ACCUM_FRAC_BITS(ACCUM_FRAC_BITS)) dut (
        .clk(clk), .rst(rst), .enable(enable),
        .weighted_input(weighted_input), .threshold(threshold), .leak(leak),
        .spike_out(spike_out), .membrane_out(membrane_out)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    initial begin
        $dumpfile("lif_neuron_tb.vcd");
        $dumpvars(0, lif_neuron_tb);
        
        rst = 1; enable = 0; weighted_input = 0;
        threshold = 16'd%(threshold_fp)d;  // %(threshold_float).1f in Q%(membrane_int_bits)d.%(membrane_frac_bits)d
        leak = 16'd%(leak_fp)d;            // %(leak_float).1f in Q%(membrane_int_bits)d.%(membrane_frac_bits)d
        
        #20; rst = 0; #10; enable = 1;
        
        // Test: input above threshold should spike
        weighted_input = 32'd%(input_fp)d;  // %(input_float).1f in Q%(accum_int_bits)d.%(accum_frac_bits)d
        repeat(15) @(posedge clk);
        
        weighted_input = 0;
        repeat(5) @(posedge clk);
        
        $finish;
    end

endmodule


// =============================================================================
// Fixed-Point Reference
// =============================================================================
// Weights Q%(weight_int_bits)d.%(weight_frac_bits)d (%(weight_sign)s): range [0, %(weight_max).5f], precision %(weight_precision).8f
// Membrane Q%(membrane_int_bits)d.%(membrane_frac_bits)d (SIGNED): range [%(membrane_min).5f, %(membrane_max).5f], precision %(membrane_precision).6f
// Example: threshold %(threshold_float).1f = %(threshold_fp)d, leak %(leak_float).1f = %(leak_fp)d, weight 0.8 = %(weight_08_fp)d
// =============================================================================
'''
    
    # Format the template with actual values using % operator
    verilog_content = verilog_template % {
        'weight_int_bits': config.weight_int_bits,
        'weight_frac_bits': config.weight_frac_bits,
        'weight_sign': weight_sign,
        'membrane_int_bits': config.membrane_int_bits,
        'membrane_frac_bits': config.membrane_frac_bits,
        'accum_int_bits': config.accum_int_bits,
        'accum_frac_bits': config.accum_frac_bits,
        'threshold_fp': threshold_10_fp,
        'threshold_float': 10.0,
        'leak_fp': leak_9_fp,
        'leak_float': 9.0,
        'input_fp': input_10_fp,
        'input_float': 10.0,
        'weight_max': weight_max,
        'weight_precision': 2**-config.weight_frac_bits,
        'membrane_min': membrane_min,
        'membrane_max': membrane_max,
        'membrane_precision': 2**-config.membrane_frac_bits,
        'weight_08_fp': weight_08_fp,
    }
    
    with open(path, 'w') as f:
        f.write(verilog_content)


# =============================================================================
# MAIN QUANTIZATION PIPELINE
# =============================================================================

def quantize_model(checkpoint_path: str, output_dir: str, config: FixedPointConfig = None):
    """Main quantization pipeline."""
    
    if config is None:
        config = FixedPointConfig()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SPIKESEG FPGA QUANTIZATION")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"\nFixed-point config:")
    weight_type = "UNSIGNED" if config.weight_unsigned else "SIGNED"
    print(f"  Weights: Q{config.weight_int_bits}.{config.weight_frac_bits} ({weight_type})")
    print(f"  Membrane: Q{config.membrane_int_bits}.{config.membrane_frac_bits} (SIGNED)")
    
    # Load checkpoint
    print("\n[1/6] Loading checkpoint...")
    checkpoint = load_checkpoint(checkpoint_path)
    weights, neuron_params_from_ckpt = extract_weights(checkpoint)
    
    # Count total parameters (only conv weights)
    total_params = sum(weights[k].size for k in weights if 'conv.weight' in k)
    print(f"  Found {len([k for k in weights if 'conv.weight' in k])} weight tensors")
    print(f"  Total parameters: {total_params:,}")
    
    # Extract and quantize each layer
    print("\n[2/6] Quantizing weights...")
    quantized_weights = {}
    error_stats = {}
    layer_shapes = {}  # Store shapes for architecture spec
    
    for key, w in weights.items():
        if 'conv.weight' in key:
            layer_name = key.split('.')[0]
            layer_shapes[layer_name] = w.shape
            print(f"  {layer_name}: {w.shape} (out_ch={w.shape[0]}, in_ch={w.shape[1]}, kernel={w.shape[2]}x{w.shape[3]})")
            print(f"    Weight range: [{w.min():.4f}, {w.max():.4f}]")
            
            # Quantize with unsigned flag for STDP weights
            q = quantize_array(w, config.weight_int_bits, config.weight_frac_bits,
                              unsigned=config.weight_unsigned)
            quantized_weights[layer_name] = q
            
            # Error analysis
            stats = compute_quantization_error(w, q, config.weight_int_bits, 
                                               config.weight_frac_bits,
                                               unsigned=config.weight_unsigned)
            error_stats[layer_name] = stats
            print(f"    Max error: {stats['max_abs_error']:.6f}")
            print(f"    RMSE: {stats['rmse']:.6f}")
            if stats['overflow_count'] > 0:
                print(f"    WARNING: {stats['overflow_count']} values overflow!")
    
    # Export binary files
    print("\n[3/6] Exporting binary files...")
    for layer_name, q in quantized_weights.items():
        bin_path = output_path / f"weights_{layer_name}.bin"
        export_weights_binary(weights[f"{layer_name}.conv.weight"], str(bin_path), config)
        print(f"  {bin_path.name}: {os.path.getsize(bin_path)} bytes")
    
    # Export C headers
    print("\n[4/6] Exporting C headers...")
    for layer_name, q in quantized_weights.items():
        h_path = output_path / f"weights_{layer_name}.h"
        export_weights_c_header(weights[f"{layer_name}.conv.weight"], 
                                f"weights_{layer_name}", str(h_path), config)
        print(f"  {h_path.name}")
    
    # Export neuron parameters FROM CHECKPOINT (not hardcoded!)
    print("\n[5/6] Exporting neuron parameters...")
    if neuron_params_from_ckpt:
        print("  Using parameters FROM CHECKPOINT (not hardcoded):")
        for layer, params in sorted(neuron_params_from_ckpt.items()):
            threshold = params.get('threshold', 'N/A')
            leak = params.get('leak', 'N/A')
            print(f"    {layer}: threshold={threshold}, leak={leak} (ABSOLUTE value)")
        export_neuron_params(neuron_params_from_ckpt, str(output_path / "neuron_params.h"), config)
    else:
        print("  WARNING: No neuron parameters found in checkpoint!")
        print("  Using config.yaml defaults (threshold=0.1)")
        # Fallback to config.yaml defaults (low thresholds for sparse EBSSA data)
        neuron_params_from_ckpt = {
            'conv1': {'threshold': 0.1, 'leak': 0.09},   # 90% of threshold
            'conv2': {'threshold': 0.1, 'leak': 0.01},   # 10% of threshold
            'conv3': {'threshold': 0.1, 'leak': 0.0},    # No leak
        }
        export_neuron_params(neuron_params_from_ckpt, str(output_path / "neuron_params.h"), config)
    print(f"  neuron_params.h")
    
    # Export architecture spec with actual layer shapes
    print("\n[6/6] Generating architecture documentation...")
    export_architecture_spec(weights, config, str(output_path / "architecture_spec.txt"),
                            neuron_params=neuron_params_from_ckpt,
                            layer_shapes=layer_shapes)
    print(f"  architecture_spec.txt")
    
    # Export Verilog LIF module
    export_verilog_lif(str(output_path / "lif_neuron.v"), config)
    print(f"  lif_neuron.v")
    
    # Summary
    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)
    
    # Correct size calculation
    total_weight_bytes = sum(os.path.getsize(output_path / f"weights_{name}.bin") 
                             for name in quantized_weights.keys())
    # Binary files include header, so actual weight bytes = file_size - header
    # Header format: magic (4) + version (4) + ndims (4) + shape (4*ndims) + dtype_flag (1)
    # For 4D convolution weights: 4 + 4 + 4 + 16 + 1 = 29 bytes
    header_size_per_file = 4 + 4 + 4 + 4 * 4 + 1  # 29 bytes for 4D tensor
    total_header_bytes = header_size_per_file * len(quantized_weights)
    actual_weight_bytes = total_weight_bytes - total_header_bytes
    
    # FP32 would be 4 bytes per param, uint16 is 2 bytes
    fp32_size = total_params * 4
    uint16_size = total_params * 2
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Quantized size: {uint16_size:,} bytes ({uint16_size/1024:.2f} KB)")
    print(f"Original FP32 size: {fp32_size:,} bytes ({fp32_size/1024:.2f} KB)")
    print(f"Compression ratio: {fp32_size/uint16_size:.1f}x")
    
    print("\nQuantization errors:")
    for layer, stats in error_stats.items():
        print(f"  {layer}: RMSE={stats['rmse']:.6f}, Max={stats['max_abs_error']:.6f}")
    
    print(f"\nOutput files in: {output_dir}/")
    for f in sorted(output_path.iterdir()):
        print(f"  {f.name}")
    
    return quantized_weights, error_stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantize SpikeSEG model for ZCU102 FPGA deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (Q0.16 unsigned weights, Q5.11 membrane):
  python scripts/quantize_for_fpga.py -c checkpoint_best.pt
  
  # Custom format (Q1.15 signed weights, Q4.12 membrane):
  python scripts/quantize_for_fpga.py -c checkpoint.pt --signed-weights --weight-frac-bits 15
  
  # Specify output directory:
  python scripts/quantize_for_fpga.py -c checkpoint.pt -o output/fpga/

Fixed-Point Formats:
  Weights (default Q0.16 unsigned): Optimal for STDP weights in [0, 1]
  Membrane (default Q5.11 signed):  Allows threshold up to 16.0 (IGARSS uses 10.0)
"""
    )
    parser.add_argument(
        '--checkpoint', '-c',
        default='checkpoint_best.pt',
        help='Path to model checkpoint (default: checkpoint_best.pt in current dir)'
    )
    parser.add_argument(
        '--output', '-o',
        default='fpga_weights',
        help='Output directory for quantized weights (default: fpga_weights/)'
    )
    parser.add_argument(
        '--weight-frac-bits',
        type=int,
        default=16,
        help='Fractional bits for weights (default: 16 for Q0.16 unsigned)'
    )
    parser.add_argument(
        '--membrane-frac-bits',
        type=int,
        default=11,
        help='Fractional bits for membrane (default: 11 for Q5.11, allows threshold up to 16.0)'
    )
    parser.add_argument(
        '--signed-weights',
        action='store_true',
        help='Use signed weights (Q1.15) instead of unsigned (Q0.16). Not recommended for STDP.'
    )
    
    args = parser.parse_args()
    
    # Build config
    weight_unsigned = not args.signed_weights
    config = FixedPointConfig(
        weight_frac_bits=args.weight_frac_bits,
        weight_int_bits=16 - args.weight_frac_bits,
        weight_unsigned=weight_unsigned,
        membrane_frac_bits=args.membrane_frac_bits,
        membrane_int_bits=16 - args.membrane_frac_bits,
    )
    
    quantize_model(args.checkpoint, args.output, config)


if __name__ == '__main__':
    main()
