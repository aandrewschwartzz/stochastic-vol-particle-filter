# Stochastic Volatility Particle Filter

Implementation of particle Gibbs sampling for stochastic volatility models based on ["Particle Gibbs for Likelihood-Free Inference of State Space Models with Application to Stochastic Volatility"](https://arxiv.org/abs/2312.13044) by Hou and Wong (2023).

## Key Modification

This implementation includes a modification to stabilize numerical degeneracy in the ABC-CAPF (Approximate Bayesian Computation - Conditional Auxiliary Particle Filter) algorithm:

- **Original paper**: Uses Gaussian kernel for the ABC approximation
- **This implementation**: Uses **Cauchy kernel** instead to mitigate numerical instability

The Cauchy kernel modification is implemented in `particle_filter/algo/ABC_CAPF.py` (lines 90-95):

```python
# Cauchy kernel to mitigate numerical instability
K_epsilon = 1 / (1 + (discrepancy**2 / self.epsilon**2))
```

## Structure

- `particle_filter/algo/` - Core ABC-CAPF and particle Gibbs implementations
- `particle_filter/algo/cython_code/` - Cython-optimized components for performance
- `particle_filter/Z_estimation/` - Maximum likelihood estimation for auxiliary variable parameters
- `particle_filter/run_functions/` - Helper functions for running experiments

## Requirements

See `requirements.txt` for dependencies. Main requirements include:
- NumPy
- SciPy (for levy_stable distributions)
- Cython (for optimized code)

## Usage

```python
python particle_filter/run.py
```

## Citation

If you use this code, please cite the original paper:
```
@article{hou2023particle,
  title={Particle Gibbs for Likelihood-Free Inference of State Space Models with Application to Stochastic Volatility},
  author={Hou, Zhaoran and Wong, Samuel W.K.},
  journal={arXiv preprint arXiv:2312.13044},
  year={2023}
}
```

## License

This implementation is for research purposes. Please refer to the original paper for theoretical details. Also, there's a lot of leftover code from different implementations.
