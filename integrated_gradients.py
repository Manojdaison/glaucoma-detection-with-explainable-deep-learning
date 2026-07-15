"""
integrated_gradients.py
=======================
Integrated Gradients for attribution analysis.
Reference: Sundararajan et al., ICML 2017
"""
import numpy as np
import tensorflow as tf

def integrated_gradients(model, input_img, baseline=None, n_steps=50):
    """
    Compute Integrated Gradients attribution.
    
    IG(x) = (x - x') * integral_0^1 of grad_input(baseline + t*(x - baseline))
    """
    if baseline is None:
        baseline = np.zeros_like(input_img)
    
    # Interpolation path from baseline to input
    alphas = np.linspace(0, 1, n_steps)
    interpolated_inputs = [
        baseline + alpha * (input_img - baseline) for alpha in alphas
    ]
    interpolated_inputs = np.concatenate(interpolated_inputs, axis=0)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)
    
    grads = tape.gradient(predictions, interpolated_inputs)
    
    # Integrate gradients
    avg_grads = np.mean(grads.numpy(), axis=0)
    integrated_grads = (input_img - baseline) * avg_grads
    
    return integrated_grads.astype(np.float32)