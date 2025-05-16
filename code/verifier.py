
import argparse
import torch
import torch.nn as nn
from networks import FullyConnected, Conv, Normalization
from typing import Tuple, Optional

# Device and input size constants
DEVICE = 'cpu'
INPUT_SIZE = 28

#-----------------------------------------------------------------------------------------------
# USE batch_verify1.py TO CHECK ALL THE TESTS CASES TOGETHER AND FIND MISMATCH FROM gt.txt...
# usage : python batch_verify1.py
#-----------------------------------------------------------------------------------------------

class Bounds:
    """
    A simple container to hold lower and upper bounds (as torch.Tensors).
    """
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        # lb: Lower bound tensor...
        self.lb = lb  
        # ub: Upper bound tensor...
        self.ub = ub  

def pgd_attack(model: nn.Module, x: torch.Tensor, y: int, 
               eps: float = 0.1, alpha: float = 0.01, 
               iters: int = 100) -> Tuple[torch.Tensor, bool]:
    """
    Performs a Projected Gradient Descent (PGD) attack on the input 'x' to generate 
    an adversarial example within an L_infinity ε-ball.
    
    Parameters:
        model (nn.Module): The neural network model (should return logits)...
        x (torch.Tensor): Input image tensor...
        y (int): True label for the input...
        eps (float): Maximum allowed perturbation (ε)...
        alpha (float): Step size for each PGD iteration....
        iters (int): Number of PGD iterations...
    
    Returns:
        Tuple[torch.Tensor, bool]: A tuple where the first element is the adversarial 
        image and the second is a boolean indicating whether the attack succeeded 
        (i.e. the final prediction is not equal to y)...
    """
    # Print initial PGD attack parameters.
    print(f"\n🚀 Starting PGD Attack (ε={eps:.4f}, α={alpha:.4f}, iters={iters})")
    # Print the initial model prediction.
    print(f"Initial prediction: {model(x).argmax().item()}")
    
    # Clone input x to create adversarial input (x_adv) and original copy.
    x_adv = x.clone().detach().requires_grad_(True)
    orig = x.clone().detach()
    
    # Iterate for a fixed number of iterations.
    for i in range(iters):
        # Forward pass: compute logits from current adversarial input.
        logits = model(x_adv)
        # Get current prediction.
        pred = logits.argmax().item()
        # Compute the loss (cross-entropy between logits and true label).
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([y]).to(x.device))
        
        # Zero the gradients in the model.
        model.zero_grad()
        # If x_adv already has a gradient, zero it to avoid accumulation.
        if x_adv.grad is not None:
            x_adv.grad.data.zero_()
        # Backward pass: compute gradients of loss w.r.t. x_adv.
        loss.backward()
        
        with torch.no_grad():
            # Get the computed gradient.
            grad = x_adv.grad.data
            # Compute the norm of the gradient (for debugging).
            grad_norm = grad.norm().item()
            
            # Print debugging information every 5 iterations or on the last iteration.
            if i % 5 == 0 or i == iters - 1:
                print(f"\nIter {i+1}/{iters}:")
                print(f"  Current pred: {pred} (true: {y})")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Grad norm: {grad_norm:.4f}")
                print(f"  Input range: [{x_adv.min().item():.4f}, {x_adv.max().item():.4f}]")
                print(f"  Δ from original: {(x_adv - orig).abs().max().item():.4f}")
            
            # Perform the PGD step: move x_adv in the direction of the sign of the gradient.
            x_adv = x_adv + alpha * grad.sign()
            
            # Project the updated x_adv back into the ε-ball around the original input:
            # It ensures that the maximum perturbation per pixel does not exceed eps.
            x_adv = torch.max(torch.min(x_adv, orig + eps), orig - eps)
            # Also, ensure that the perturbed image stays within the valid pixel range [0, 1].
            x_adv = torch.clamp(x_adv, 0, 1)
            # Detach x_adv from the current computation graph and enable gradient computation for next iteration.
            x_adv = x_adv.detach().requires_grad_(True)
    
    # After all iterations, evaluate the final prediction.
    final_pred = model(x_adv).argmax().item()
    # Determine if the attack was successful (i.e., if the final prediction is not the true label)...
    success = final_pred != y
    print(f"\n🔥 PGD Attack {'SUCCEEDED' if success else 'FAILED'}")
    print(f"  Final prediction: {final_pred} (true: {y})")
    print(f"  Max perturbation: {(x_adv - orig).abs().max().item():.4f}")
    print(f"  Final input range: [{x_adv.min().item():.4f}, {x_adv.max().item():.4f}]")
    
    # Return the final adversarial image and whether the attack succeeded.
    return x_adv.detach(), success

def verify_with_pgd(net: nn.Module, x: torch.Tensor, y: int, eps: float) -> bool:
    """
    Runs a PGD attack and returns True if no adversarial example is found (i.e., the network is robust).
    """
    _, success = pgd_attack(net, x, y, eps=eps, alpha=0.01, iters=100)
    return not success  # Returns True if PGD fails (network is robust), False if adversary is found.

def basic_interval_analysis(net: nn.Module, inputs: torch.Tensor, eps: float, 
                          true_label: int) -> Tuple[Bounds, Optional[bool]]:
    """
    Computes output bounds using simple interval propagation (IBP).
    Returns:
      - A Bounds object containing lower and upper bounds.
      - An optional boolean:
            True if the bounds prove robustness,
            False if they prove non-robustness,
            None if the result is uncertain.
    """
    # Compute the initial bounds by perturbing the input by eps and clamping to [0, 1].
    lb = torch.clamp(inputs - eps, min=0., max=1.)
    ub = torch.clamp(inputs + eps, min=0., max=1.)
    bounds = Bounds(lb, ub)

    with torch.no_grad():
        for layer in net.layers:
            if isinstance(layer, nn.Linear):
                # For Linear layers, use weight splitting.
                pos_w = torch.clamp(layer.weight, min=0)
                neg_w = torch.clamp(layer.weight, max=0)
                # Compute lower bound with positive part of weights applied to lb and negative part to ub.
                lb = torch.matmul(bounds.lb.reshape(bounds.lb.size(0), -1), pos_w.t()) + \
                     torch.matmul(bounds.ub.reshape(bounds.ub.size(0), -1), neg_w.t())
                # Similarly, compute upper bound.
                ub = torch.matmul(bounds.ub.reshape(bounds.ub.size(0), -1), pos_w.t()) + \
                     torch.matmul(bounds.lb.reshape(bounds.lb.size(0), -1), neg_w.t())
                if layer.bias is not None:
                    lb += layer.bias
                    ub += layer.bias
                bounds = Bounds(lb, ub)
            elif isinstance(layer, nn.ReLU):
                # For ReLU, the output is 0 for negative inputs, so take max(bound, 0).
                new_lb = torch.maximum(bounds.lb, torch.zeros_like(bounds.lb))
                new_ub = torch.maximum(bounds.ub, torch.zeros_like(bounds.ub))
                bounds = Bounds(new_lb, new_ub)
            elif isinstance(layer, nn.Conv2d):
                # For convolutional layers, use your custom bound propagation.
                bounds.lb, bounds.ub = optimize_conv_bounds(bounds.lb, bounds.ub, layer)
            elif isinstance(layer, Normalization):
                # For normalization, adjust bounds using mean and sigma.
                mean, sigma = layer.mean, layer.sigma
                bounds.lb = (bounds.lb - mean) / sigma
                bounds.ub = (bounds.ub - mean) / sigma
            elif isinstance(layer, nn.Flatten):
                # Flatten the bounds...
                bounds.lb = bounds.lb.reshape(bounds.lb.size(0), -1)
                bounds.ub = bounds.ub.reshape(bounds.ub.size(0), -1)

    # Check robustness: compare the true label's lower bound to the maximum upper bound among other classes.
    true_lb = bounds.lb[0, true_label]
    other_ub = torch.max(bounds.ub[0, torch.arange(bounds.ub.size(1)) != true_label])
    
    if true_lb > other_ub + 1e-5:
        return bounds, True  # Verified robust..
    if true_lb < other_ub - 1e-5:
        return bounds, False  # Verified non-robust..
    return bounds, None  # Inconclusive

def optimize_relu_bounds(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes tighter bounds for a ReLU activation using dynamic slope adaptation.
    
    For neurons where the pre-activation interval is entirely nonnegative (active) or nonpositive (inactive),
    the bounds remain as-is (or 0 for inactive). For ambiguous (crossing) neurons,
    an adaptive slope (λ) is computed and used to tighten the upper bound.
    """
    new_lb = torch.zeros_like(lb)
    new_ub = torch.zeros_like(ub)
    
    # For neurons that are definitely active, keep the original bounds.
    active = lb >= 0
    new_lb[active] = lb[active]
    new_ub[active] = ub[active]
    
    # For neurons that are definitely inactive, output 0.
    inactive = ub <= 0
    new_lb[inactive] = 0.0
    new_ub[inactive] = 0.0
    
    # For ambiguous neurons (crossing), compute a dynamic slope.
    crossing = (lb < 0) & (ub > 0)
    if torch.any(crossing):
        # Extract the bounds for crossing neurons.
        ub_cross = ub[crossing]
        lb_cross = lb[crossing]
        
        # Compute the base slope (u/(u-l)) with a safeguard against division by zero.
        denominator = ub_cross - lb_cross
        safe_denom = torch.where(denominator.abs() < 1e-12, torch.ones_like(denominator) * 1e-12, denominator)
        base_alpha = ub_cross / safe_denom
        
        # Optionally, adjust the slope further using a heuristic based on the neuron’s position...?
        position_factor = ub_cross / (ub_cross - lb_cross)
        dynamic_alpha = 0.2 + 0.3 * position_factor  # For example, this may range between 0.2 and 0.5.can change if needed.. but works fine now..
        
        # Choose the smaller (more conservative) value between base and dynamic.
        alpha = torch.minimum(base_alpha, dynamic_alpha)
        
        # For ambiguous neurons, maintain a lower bound of 0 for soundness, and tighten the upper bound.
        new_lb[crossing] = 0.0
        new_ub[crossing] = alpha * ub_cross
    
    return new_lb, new_ub


def optimize_linear_bounds(lb: torch.Tensor, ub: torch.Tensor, 
                         weight: torch.Tensor, bias: torch.Tensor, 
                         is_final_layer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced linear bound propagation with multi-stage tightening"""
    # Flatten if needed..
    if len(lb.shape) > 2:
        lb = lb.reshape(lb.size(0), -1)
        ub = ub.reshape(ub.size(0), -1)

    # Basic interval arithmetic
    pos_w = torch.clamp(weight, min=0)
    neg_w = torch.clamp(weight, max=0)
    new_lb = torch.matmul(lb, pos_w.t()) + torch.matmul(ub, neg_w.t())
    new_ub = torch.matmul(ub, pos_w.t()) + torch.matmul(lb, neg_w.t())

    # Add bias if present..
    if bias is not None:
        new_lb += bias
        new_ub += bias

    # Multi-stage bound tightening
    if not is_final_layer:
        width = new_ub - new_lb
        avg_width = width.mean().item()
        
        # Stage 1: Moderate tightening for very wide bounds
        wide_mask = width > 100
        if torch.any(wide_mask):
            midpoint = (new_lb + new_ub) / 2
            new_lb = torch.where(wide_mask, midpoint - width/5, new_lb)
            new_ub = torch.where(wide_mask, midpoint + width/5, new_ub)
        
        # Stage 2: Gentle tightening for moderately wide bounds
        moderate_mask = (width > 30) & (width <= 50)
        if torch.any(moderate_mask):
            midpoint = (new_lb + new_ub) / 2
            new_lb = torch.where(moderate_mask, midpoint - width/3, new_lb)
            new_ub = torch.where(moderate_mask, midpoint + width/3, new_ub)
    
    return new_lb, new_ub
def optimize_conv_bounds(lb, ub, layer):
    """
    compute precise convolutional bounds with optimized bound tightening...
    """
    weight = layer.weight
    bias = layer.bias.data if layer.bias is not None else None
    
    # extract convolution parameters
    stride = layer.stride[0]
    padding = layer.padding[0]
    kernel_size = layer.kernel_size[0]
    
    # handle padding more precisely
    if padding > 0:
        pad = nn.ZeroPad2d(padding)
        padded_lb = pad(lb)
        padded_ub = pad(ub)
        
        # Optimize bounds for padded regions (known to be zero)
        padded_lb[:, :, :padding, :] = 0
        padded_lb[:, :, -padding:, :] = 0
        padded_ub[:, :, :padding, :] = 0
        padded_ub[:, :, -padding:, :] = 0
        
        padded_lb[:, :, :, :padding] = 0
        padded_lb[:, :, :, -padding:] = 0
        padded_ub[:, :, :, :padding] = 0
        padded_ub[:, :, :, -padding:] = 0
    else:
        padded_lb = lb
        padded_ub = ub
    
    # compute output dimensions
    n, c, h, w = padded_lb.shape
    out_h = (h - kernel_size) // stride + 1  # removed padding from calculation since we already padded
    out_w = (w - kernel_size) // stride + 1
    
    # efficient bound computation using optimized im2col
    lb_unf = nn.functional.unfold(padded_lb, kernel_size, stride=stride, padding=0)
    ub_unf = nn.functional.unfold(padded_ub, kernel_size, stride=stride, padding=0)
    
    # reshape weight for efficient computation
    in_channels = weight.size(1)
    out_channels = weight.size(0)
    weight_matrix = weight.view(out_channels, -1)
    pos_w = torch.clamp(weight_matrix, min=0)
    neg_w = torch.clamp(weight_matrix, max=0)
    
    # compute bounds
    new_lb = torch.matmul(pos_w, lb_unf) + torch.matmul(neg_w, ub_unf)
    new_ub = torch.matmul(pos_w, ub_unf) + torch.matmul(neg_w, lb_unf)
    
    # handle small intervals
    diff_unf = ub_unf - lb_unf
    small_interval_mask = diff_unf < 1e-4
    
    if torch.any(small_interval_mask):
        midpoint = (lb_unf + ub_unf) / 2
        mid_val = torch.matmul(weight_matrix, midpoint)
        small_interval_mask_reshaped = small_interval_mask.any(dim=1).view(1, -1)
        small_interval_mask_expanded = small_interval_mask_reshaped.expand_as(new_lb)
        new_lb = torch.where(small_interval_mask_expanded, mid_val, new_lb)
        new_ub = torch.where(small_interval_mask_expanded, mid_val, new_ub)
    
    # reshape the output: [out_channels, batch_size * out_h * out_w] -> [batch_size, out_channels, out_h, out_w]
    new_lb = new_lb.view(out_channels, n, out_h, out_w).permute(1, 0, 2, 3)
    new_ub = new_ub.view(out_channels, n, out_h, out_w).permute(1, 0, 2, 3)
    
    if bias is not None:
        bias_term = bias.view(1, -1, 1, 1)
        new_lb += bias_term
        new_ub += bias_term
    
    return new_lb, new_ub

def get_bounds_affine(net: nn.Module, inputs: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes output bounds using affine (DeepPoly-style) propagation.
    
    The input is first clamped (with perturbation eps) and flattened to a row vector x of shape (1, d).
     initialize the affine form f(x)= x*M + b with M = I (d x d) and b = 0 (1 x d).
    Then, for each layer in net.layers, we update M and b as follows:
      - For a Linear layer (with output: x @ weight.t() + bias), update:
            M <- M @ layer.weight.t()
            b <- b @ layer.weight.t() + layer.bias.unsqueeze(0)
      - For a ReLU layer, compute pre-activation bounds from the current affine form and then
        for each neuron j where the bound is ambiguous (l[j] < 0 < u[j]), compute an adaptive slope λ
        (using λ = clamp(u[j] / (u[j]-l[j]), 0.01, 0.99)) and update the j-th column of M and b.
      - For Normalization, adjust the bias b...
      - Flatten layers do not change the affine form...
    
    Finally, the function computes the final lower and upper bounds from the affine form.
    
    Returns:
       final_lb, final_ub: both of shape (1, n), where n is the number of output neurons.
    """
    # Compute input bounds and flatten to a row vector
    x_lb = torch.clamp(inputs - eps, 0, 1).view(1, -1)  # shape: (1, d)
    x_ub = torch.clamp(inputs + eps, 0, 1).view(1, -1)  # shape: (1, d)
    d = x_lb.shape[1]  # e.g. 784 for MNIST
    
    # Initialize affine form f(x)= x*M + b with M = Identity and b = 0
    M = torch.eye(d, device=x_lb.device)       # shape: (d, d)
    b = torch.zeros(1, d, device=x_lb.device)    # shape: (1, d)
    
    for layer in net.layers:
        if isinstance(layer, nn.Linear):
            # Linear layer: out = x @ layer.weight.t() + layer.bias
            M = M @ layer.weight.t()   # new M: (d, n)
            b = b @ layer.weight.t() + layer.bias.unsqueeze(0)  # new b: (1, n)
        elif isinstance(layer, nn.ReLU):
            # Compute pre-activation bounds for each neuron
            A_pos = torch.clamp(M, min=0)   # shape: (d, n)
            A_neg = torch.clamp(M, max=0)   # shape: (d, n)
            # Using a row-vector formulation: we compute l and u as 1D tensors of length n.
            # Note: we multiply x_lb (shape (1, d)) transposed so that the multiplication is over the d dimension.
            l = (A_pos * x_lb.transpose(0, 1)).sum(dim=0) + (A_neg * x_ub.transpose(0, 1)).sum(dim=0) + b.squeeze(0)
            u = (A_pos * x_ub.transpose(0, 1)).sum(dim=0) + (A_neg * x_lb.transpose(0, 1)).sum(dim=0) + b.squeeze(0)
            # Identify ambiguous neurons (where l < 0 < u)
            mask = (l < 0) & (u > 0)
            mask = mask.view(-1)  # Ensure it's a 1D boolean tensor
            if mask.any():
                lambda_val = torch.zeros_like(l)
                denom = u[mask] - l[mask]
                safe_denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom)*1e-12, denom)
                # Compute base λ and clamp it to [0.01, 0.99]
                lambda_val[mask] = torch.clamp(u[mask] / safe_denom, 0.01, 0.99)
                # Optionally, apply a conservative factor (e.g. multiply by 0.9) to tighten the bound:
                conservative_factor = 0.9
                lambda_val[mask] = lambda_val[mask] * conservative_factor
                # Update the affine form for each ambiguous neuron j
                for j in range(M.shape[1]):
                    if mask[j].item():
                        M[:, j] = lambda_val[j] * M[:, j]
                        b[0, j] = lambda_val[j] * b[0, j] - lambda_val[j] * l[j]
            # For definite neurons (active or inactive), leave M and b unchanged.
        elif isinstance(layer, Normalization):
            mean, sigma = layer.mean, layer.sigma
            b = (b - mean) / sigma
        elif isinstance(layer, nn.Flatten):
            # Flatten does not change the affine form.
            pass
        else:
            continue
    
    # Final interval computation from the affine form:
    A_pos = torch.clamp(M, min=0)
    A_neg = torch.clamp(M, max=0)
    final_lb = (A_pos * x_lb.transpose(0, 1)).sum(dim=0) + (A_neg * x_ub.transpose(0, 1)).sum(dim=0) + b.squeeze(0)
    final_ub = (A_pos * x_ub.transpose(0, 1)).sum(dim=0) + (A_neg * x_lb.transpose(0, 1)).sum(dim=0) + b.squeeze(0)
    
    final_lb = final_lb.view(1, -1)
    final_ub = final_ub.view(1, -1)
    return final_lb, final_ub

def advanced_verification(net: nn.Module, inputs: torch.Tensor, 
                          eps: float, true_label: int) -> bool:
    """
    Advanced verification that combines IBP-based propagation and affine (DeepPoly) bounds.
    Returns True if the combined bounds verify robustness, False otherwise.
    """
    # === IBP-based propagation (using your existing optimize functions) ===
    lb_ibp = torch.clamp(inputs - eps, min=0., max=1.)
    ub_ibp = torch.clamp(inputs + eps, min=0., max=1.)
    bounds_ibp = Bounds(lb_ibp, ub_ibp)
    
    with torch.no_grad():
        for i, layer in enumerate(net.layers):
            if isinstance(layer, nn.ReLU):
                bounds_ibp.lb, bounds_ibp.ub = optimize_relu_bounds(bounds_ibp.lb, bounds_ibp.ub)
            elif isinstance(layer, nn.Linear):
                is_final = (i == len(net.layers) - 1)
                if len(bounds_ibp.lb.shape) > 2:
                    lb_input = bounds_ibp.lb.reshape(bounds_ibp.lb.size(0), -1)
                    ub_input = bounds_ibp.ub.reshape(bounds_ibp.ub.size(0), -1)
                else:
                    lb_input = bounds_ibp.lb
                    ub_input = bounds_ibp.ub
                bounds_ibp.lb, bounds_ibp.ub = optimize_linear_bounds(lb_input, ub_input,
                                                                       layer.weight, layer.bias,
                                                                       is_final_layer=is_final)
            elif isinstance(layer, nn.Conv2d):
                bounds_ibp.lb, bounds_ibp.ub = optimize_conv_bounds(bounds_ibp.lb, bounds_ibp.ub, layer)
            elif isinstance(layer, Normalization):
                mean, sigma = layer.mean, layer.sigma
                bounds_ibp.lb = (bounds_ibp.lb - mean) / sigma
                bounds_ibp.ub = (bounds_ibp.ub - mean) / sigma
            elif isinstance(layer, nn.Flatten):
                bounds_ibp.lb = bounds_ibp.lb.reshape(bounds_ibp.lb.size(0), -1)
                bounds_ibp.ub = bounds_ibp.ub.reshape(bounds_ibp.ub.size(0), -1)
    
    # === Affine (DeepPoly) propagation ===
    lb_aff, ub_aff = get_bounds_affine(net, inputs, eps)
    
    # === Combine the bounds by intersecting them ===
    combined_lb = torch.max(bounds_ibp.lb, lb_aff)
    combined_ub = torch.min(bounds_ibp.ub, ub_aff)
    
    # Debug printouts
    print("\nCombined Bounds:")
    print(f"  True label {true_label} lower bound: {combined_lb[0, true_label]:.4f}")
    # Compute max upper bound among all classes other than true_label
    other_indices = [i for i in range(combined_ub.size(1)) if i != true_label]
    max_other_ub = torch.max(combined_ub[0, torch.tensor(other_indices)])
    print(f"  Max upper bound among other classes: {max_other_ub:.4f}")
    
    # Compute an adaptive margin (e.g., based on average width)
    avg_width = (combined_ub - combined_lb).mean().item()
    margin = max(1e-4, 0.002 * avg_width)  # Tune this multiplier as needed
    print(f"  Adaptive margin: {margin:.4f}")
    
    true_lb = combined_lb[0, true_label]
    return true_lb > max_other_ub + margin


def analyze(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    """Multi-stage verification pipeline"""
    
    # Stage 1: Fast empirical check
    if verify_with_pgd(net, inputs, true_label, eps):
        return True
    
    # Stage 2: Basic interval analysis
    bounds, basic_result = basic_interval_analysis(net, inputs, eps, true_label)
    if basic_result is not None:
        return basic_result
    
    # Stage 3: Advanced verification
    return advanced_verification(net, inputs, eps, true_label)

def main():
    parser = argparse.ArgumentParser(description='Neural network verification')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    # Load network
    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    
    # Run empirical attacks first
    print("\n🧪 Running Empirical Attacks:")
    pgd_image, pgd_success = pgd_attack(net, inputs, true_label, eps)
    print(f"PGD attack {'succeeded' if pgd_success else 'failed'}")
    
    # Run formal verification
    print("\n🔍 Running Formal Verification:")
    is_verified = analyze(net, inputs, eps, true_label)
    
    print("\n🎯 Final Result:")
    print('verified' if is_verified else 'not verified')

if __name__ == '__main__':
    main()