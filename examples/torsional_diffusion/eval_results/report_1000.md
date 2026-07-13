# Torsional-diffusion eval - 2026-07-12 16:47

- **method** GFN2-xTB - force tol 0.02 eV/A - 100 mols/split - K = 2*n_crest samples/mol
- **runs** crest, rdkit_t1 - gap baseline **crest**
- units: AMR / relax_shift in A, init_force in eV/A, Cov & frac_converged in [0,1], n_iter = L-BFGS steps.
- gap = `run - baseline`. Lower-is-better (AMR, init_force, n_iter): **+ = worse**. Higher-is-better (Cov, frac_converged): **- = worse**.

## train

### Raw (pre-relaxation)

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| amr_r | 0.011 | 0.099 |
| amr_p | 0.062 | 0.227 |
| cov_r_0.5 | 0.997 | 0.986 |
| cov_p_0.5 | 0.963 | 0.889 |
| cov_r_0.75 | 1.000 | 1.000 |
| cov_p_0.75 | 0.980 | 0.962 |
| cov_r_1.0 | 1.000 | 1.000 |
| cov_p_1.0 | 0.984 | 0.977 |
| cov_r_1.25 | 1.000 | 1.000 |
| cov_p_1.25 | 1.000 | 0.996 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| amr_r | +0.087 |
| amr_p | +0.165 |
| cov_r_0.5 | -0.010 |
| cov_p_0.5 | -0.074 |
| cov_r_0.75 | +0.000 |
| cov_p_0.75 | -0.017 |
| cov_r_1.0 | +0.000 |
| cov_p_1.0 | -0.007 |
| cov_r_1.25 | +0.000 |
| cov_p_1.25 | -0.003 |

### Relaxed (GFN2-xTB)

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| relaxed_amr_r | 0.027 | 0.045 |
| relaxed_amr_p | 0.087 | 0.139 |
| relax_shift | 0.054 | 0.187 |
| relaxed_cov_r_0.5 | 0.996 | 0.985 |
| relaxed_cov_p_0.5 | 0.938 | 0.884 |
| relaxed_cov_r_0.75 | 1.000 | 1.000 |
| relaxed_cov_p_0.75 | 0.970 | 0.965 |
| relaxed_cov_r_1.0 | 1.000 | 1.000 |
| relaxed_cov_p_1.0 | 0.982 | 0.976 |
| relaxed_cov_r_1.25 | 1.000 | 1.000 |
| relaxed_cov_p_1.25 | 0.999 | 0.996 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| relaxed_amr_r | +0.018 |
| relaxed_amr_p | +0.052 |
| relax_shift | +0.134 |
| relaxed_cov_r_0.5 | -0.011 |
| relaxed_cov_p_0.5 | -0.054 |
| relaxed_cov_r_0.75 | -0.000 |
| relaxed_cov_p_0.75 | -0.005 |
| relaxed_cov_r_1.0 | +0.000 |
| relaxed_cov_p_1.0 | -0.005 |
| relaxed_cov_r_1.25 | +0.000 |
| relaxed_cov_p_1.25 | -0.003 |

### Relaxation cost / validity

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| init_force | 0.510 | 3.854 |
| n_iter | 16.817 | 61.106 |
| frac_converged | 1.000 | 1.000 |
| prune_factor | 80.637 | 22.214 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| init_force | +3.344 |
| n_iter | +44.289 |
| frac_converged | +0.000 |
| prune_factor | -58.423 |

## val

### Raw (pre-relaxation)

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| amr_r | 0.005 | 0.101 |
| amr_p | 0.029 | 0.224 |
| cov_r_0.5 | 0.999 | 0.998 |
| cov_p_0.5 | 0.990 | 0.899 |
| cov_r_0.75 | 1.000 | 1.000 |
| cov_p_0.75 | 0.997 | 0.971 |
| cov_r_1.0 | 1.000 | 1.000 |
| cov_p_1.0 | 0.998 | 0.989 |
| cov_r_1.25 | 1.000 | 1.000 |
| cov_p_1.25 | 1.000 | 0.999 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| amr_r | +0.096 |
| amr_p | +0.195 |
| cov_r_0.5 | -0.001 |
| cov_p_0.5 | -0.091 |
| cov_r_0.75 | +0.000 |
| cov_p_0.75 | -0.026 |
| cov_r_1.0 | +0.000 |
| cov_p_1.0 | -0.009 |
| cov_r_1.25 | +0.000 |
| cov_p_1.25 | -0.001 |

### Relaxed (GFN2-xTB)

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| relaxed_amr_r | 0.020 | 0.035 |
| relaxed_amr_p | 0.042 | 0.111 |
| relax_shift | 0.044 | 0.207 |
| relaxed_cov_r_0.5 | 0.999 | 0.991 |
| relaxed_cov_p_0.5 | 0.973 | 0.918 |
| relaxed_cov_r_0.75 | 1.000 | 1.000 |
| relaxed_cov_p_0.75 | 0.995 | 0.973 |
| relaxed_cov_r_1.0 | 1.000 | 1.000 |
| relaxed_cov_p_1.0 | 0.997 | 0.986 |
| relaxed_cov_r_1.25 | 1.000 | 1.000 |
| relaxed_cov_p_1.25 | 1.000 | 0.999 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| relaxed_amr_r | +0.015 |
| relaxed_amr_p | +0.069 |
| relax_shift | +0.163 |
| relaxed_cov_r_0.5 | -0.008 |
| relaxed_cov_p_0.5 | -0.055 |
| relaxed_cov_r_0.75 | +0.000 |
| relaxed_cov_p_0.75 | -0.021 |
| relaxed_cov_r_1.0 | +0.000 |
| relaxed_cov_p_1.0 | -0.011 |
| relaxed_cov_r_1.25 | +0.000 |
| relaxed_cov_p_1.25 | -0.001 |

### Relaxation cost / validity

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| init_force | 0.161 | 3.934 |
| n_iter | 14.103 | 58.713 |
| frac_converged | 1.000 | 1.000 |
| prune_factor | 93.768 | 23.430 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| init_force | +3.773 |
| n_iter | +44.610 |
| frac_converged | +0.000 |
| prune_factor | -70.338 |

## test

### Raw (pre-relaxation)

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| amr_r | 0.004 | 0.091 |
| amr_p | 0.031 | 0.201 |
| cov_r_0.5 | 1.000 | 0.997 |
| cov_p_0.5 | 0.989 | 0.919 |
| cov_r_0.75 | 1.000 | 1.000 |
| cov_p_0.75 | 0.999 | 0.993 |
| cov_r_1.0 | 1.000 | 1.000 |
| cov_p_1.0 | 1.000 | 1.000 |
| cov_r_1.25 | 1.000 | 1.000 |
| cov_p_1.25 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| amr_r | +0.086 |
| amr_p | +0.170 |
| cov_r_0.5 | -0.003 |
| cov_p_0.5 | -0.070 |
| cov_r_0.75 | +0.000 |
| cov_p_0.75 | -0.006 |
| cov_r_1.0 | +0.000 |
| cov_p_1.0 | -0.000 |
| cov_r_1.25 | +0.000 |
| cov_p_1.25 | +0.000 |

### Relaxed (GFN2-xTB)

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| relaxed_amr_r | 0.020 | 0.029 |
| relaxed_amr_p | 0.047 | 0.087 |
| relax_shift | 0.035 | 0.189 |
| relaxed_cov_r_0.5 | 1.000 | 0.995 |
| relaxed_cov_p_0.5 | 0.970 | 0.930 |
| relaxed_cov_r_0.75 | 1.000 | 1.000 |
| relaxed_cov_p_0.75 | 1.000 | 0.998 |
| relaxed_cov_r_1.0 | 1.000 | 1.000 |
| relaxed_cov_p_1.0 | 1.000 | 1.000 |
| relaxed_cov_r_1.25 | 1.000 | 1.000 |
| relaxed_cov_p_1.25 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| relaxed_amr_r | +0.009 |
| relaxed_amr_p | +0.039 |
| relax_shift | +0.153 |
| relaxed_cov_r_0.5 | -0.005 |
| relaxed_cov_p_0.5 | -0.040 |
| relaxed_cov_r_0.75 | +0.000 |
| relaxed_cov_p_0.75 | -0.002 |
| relaxed_cov_r_1.0 | +0.000 |
| relaxed_cov_p_1.0 | -0.000 |
| relaxed_cov_r_1.25 | +0.000 |
| relaxed_cov_p_1.25 | +0.000 |

### Relaxation cost / validity

| metric | crest | rdkit_t1 |
| --- | --- | --- |
| init_force | 0.217 | 4.010 |
| n_iter | 13.451 | 60.113 |
| frac_converged | 1.000 | 1.000 |
| prune_factor | 93.462 | 23.534 |

_Gap vs crest:_

| metric | rdkit_t1 |
| --- | --- |
| init_force | +3.794 |
| n_iter | +46.662 |
| frac_converged | +0.000 |
| prune_factor | -69.928 |
