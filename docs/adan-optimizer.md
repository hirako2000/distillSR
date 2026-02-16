# Adan Optimizer

_with [Schedule-Free Training](https://en.wikipedia.org/wiki/Automated_planning_and_scheduling)_

The Adan optimizer extends Adam with a third momentum buffer and incorporates schedule-free training dynamics, for faster convergence and improved stability for super-resolution tasks. This implementation is adapted from NeoSR and integrated into the training pipeline as an alternative to AdamW. Kudo to the NeoSR team.

## Some math...

[Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) (Adaptive Moment Estimation) maintains two momentum buffers. 

First moment estimate of gradients and second moment estimate of squared gradients. 

Adan introduces a third buffer tracking the difference between successive gradients, enabling Nesterov-like acceleration without requiring future gradient computation.

The update rules follow:

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t - g_{t-1})
```

```math
n_t = \beta_3 n_{t-1} + (1 - \beta_3) (g_t + \beta_2 (g_t - g_{t-1}))
```

Where g_t is the current gradient, and β₁, β₂, β₃ are momentum coefficients set to 0.98, 0.92, and 0.99 respectively.

## Schedule-Free Training

The schedule-free variant removes the need for learning rate scheduling by maintaining an additional set of slow weights that track the fast weights updated by the optimizer. This approach provides the benefits of averaging without requiring separate evaluation runs or checkpoint selection.

The slow weights z_t are updated as:

```math
z_{t+1} = (1 - c_k) z_t + c_k \theta_t
```

Where θ_t are the fast weights and c_k is a dynamically computed combination coefficient based on training progress. At evaluation time, the slow weights are used instead of the fast weights, producing better generalization.

## Integration

When enabled in the configuration, the Adan optimizer replaces AdamW with the same learning rate and weight decay parameters. The optimizer automatically handles warmup steps and transitions between training and evaluation modes. 

During training, fast weights update normally while slow weights track their moving average. At validation checkpoints, the model automatically switches to slow weights for evaluation before reverting to fast weights for continued training.

## Configuration

The optimizer is enabled by setting `optimizer: adan_sf` in [train_config.yaml](../configs/train_config.yaml). Parameters include learning rate, weight decay, warmup steps, and the three momentum coefficients.

For more details on loss functions that work well with this optimizer, read [loss-functions](./loss-functions.md).