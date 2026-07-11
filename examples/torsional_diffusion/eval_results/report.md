# Torsional-diffusion eval - 2026-07-08 00:22

- **method** GFN2-xTB - force tol 0.02 eV/A - 100 mols/split - K = 2*n_crest samples/mol
- **runs** crest, rdkit_t1, rdkit_t2 - gap baseline **crest**
- units: AMR / relax_shift in A, init_force in eV/A, Cov & frac_converged in [0,1], n_iter = L-BFGS steps.
- gap = `run - baseline`. Lower-is-better (AMR, init_force, n_iter): **+ = worse**. Higher-is-better (Cov, frac_converged): **- = worse**.

## train

### Raw (pre-relaxation)

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| amr_r | 0.032 | 0.166 | 0.179 |
| amr_p | 0.053 | 0.218 | 0.217 |
| cov_r_0.5 | 0.989 | 0.945 | 0.943 |
| cov_p_0.5 | 0.977 | 0.893 | 0.906 |
| cov_r_0.75 | 0.998 | 0.996 | 0.982 |
| cov_p_0.75 | 0.984 | 0.970 | 0.968 |
| cov_r_1.0 | 1.000 | 1.000 | 0.989 |
| cov_p_1.0 | 0.990 | 0.984 | 0.980 |
| cov_r_1.25 | 1.000 | 1.000 | 1.000 |
| cov_p_1.25 | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| amr_r | +0.134 | +0.147 |
| amr_p | +0.165 | +0.164 |
| cov_r_0.5 | -0.045 | -0.047 |
| cov_p_0.5 | -0.085 | -0.071 |
| cov_r_0.75 | -0.002 | -0.016 |
| cov_p_0.75 | -0.015 | -0.017 |
| cov_r_1.0 | +0.000 | -0.011 |
| cov_p_1.0 | -0.006 | -0.010 |
| cov_r_1.25 | +0.000 | +0.000 |
| cov_p_1.25 | +0.000 | +0.000 |

### Relaxed (GFN2-xTB)

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| relaxed_amr_r | 0.031 | 0.084 | 0.104 |
| relaxed_amr_p | 0.046 | 0.118 | 0.122 |
| relax_shift | 0.017 | 0.156 | 0.156 |
| relaxed_cov_r_0.5 | 0.989 | 0.948 | 0.943 |
| relaxed_cov_p_0.5 | 0.979 | 0.906 | 0.921 |
| relaxed_cov_r_0.75 | 0.998 | 0.998 | 0.976 |
| relaxed_cov_p_0.75 | 0.984 | 0.973 | 0.967 |
| relaxed_cov_r_1.0 | 1.000 | 1.000 | 0.989 |
| relaxed_cov_p_1.0 | 0.990 | 0.987 | 0.982 |
| relaxed_cov_r_1.25 | 1.000 | 1.000 | 1.000 |
| relaxed_cov_p_1.25 | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| relaxed_amr_r | +0.053 | +0.073 |
| relaxed_amr_p | +0.072 | +0.076 |
| relax_shift | +0.139 | +0.139 |
| relaxed_cov_r_0.5 | -0.041 | -0.047 |
| relaxed_cov_p_0.5 | -0.073 | -0.058 |
| relaxed_cov_r_0.75 | +0.000 | -0.022 |
| relaxed_cov_p_0.75 | -0.011 | -0.017 |
| relaxed_cov_r_1.0 | +0.000 | -0.011 |
| relaxed_cov_p_1.0 | -0.003 | -0.008 |
| relaxed_cov_r_1.25 | +0.000 | +0.000 |
| relaxed_cov_p_1.25 | +0.000 | +0.000 |

### Relaxation cost / validity

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| init_force | 0.111 | 3.954 | 3.977 |
| n_iter | 9.623 | 56.655 | 58.296 |
| frac_converged | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| init_force | +3.843 | +3.866 |
| n_iter | +47.032 | +48.673 |
| frac_converged | +0.000 | +0.000 |

## val

### Raw (pre-relaxation)

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| amr_r | 0.023 | 0.189 | 0.211 |
| amr_p | 0.029 | 0.221 | 0.231 |
| cov_r_0.5 | 0.991 | 0.933 | 0.912 |
| cov_p_0.5 | 0.987 | 0.902 | 0.902 |
| cov_r_0.75 | 0.992 | 0.990 | 0.971 |
| cov_p_0.75 | 0.998 | 0.979 | 0.972 |
| cov_r_1.0 | 1.000 | 0.994 | 0.991 |
| cov_p_1.0 | 0.998 | 0.995 | 0.986 |
| cov_r_1.25 | 1.000 | 1.000 | 1.000 |
| cov_p_1.25 | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| amr_r | +0.166 | +0.188 |
| amr_p | +0.192 | +0.201 |
| cov_r_0.5 | -0.057 | -0.078 |
| cov_p_0.5 | -0.085 | -0.085 |
| cov_r_0.75 | -0.001 | -0.021 |
| cov_p_0.75 | -0.019 | -0.026 |
| cov_r_1.0 | -0.006 | -0.009 |
| cov_p_1.0 | -0.004 | -0.012 |
| cov_r_1.25 | +0.000 | +0.000 |
| cov_p_1.25 | +0.000 | +0.000 |

### Relaxed (GFN2-xTB)

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| relaxed_amr_r | 0.022 | 0.079 | 0.099 |
| relaxed_amr_p | 0.022 | 0.088 | 0.107 |
| relax_shift | 0.013 | 0.197 | 0.184 |
| relaxed_cov_r_0.5 | 0.990 | 0.956 | 0.937 |
| relaxed_cov_p_0.5 | 0.990 | 0.948 | 0.934 |
| relaxed_cov_r_0.75 | 0.992 | 0.989 | 0.969 |
| relaxed_cov_p_0.75 | 0.998 | 0.981 | 0.969 |
| relaxed_cov_r_1.0 | 1.000 | 0.993 | 0.991 |
| relaxed_cov_p_1.0 | 0.998 | 0.993 | 0.982 |
| relaxed_cov_r_1.25 | 1.000 | 1.000 | 1.000 |
| relaxed_cov_p_1.25 | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| relaxed_amr_r | +0.056 | +0.077 |
| relaxed_amr_p | +0.066 | +0.085 |
| relax_shift | +0.184 | +0.171 |
| relaxed_cov_r_0.5 | -0.035 | -0.053 |
| relaxed_cov_p_0.5 | -0.042 | -0.056 |
| relaxed_cov_r_0.75 | -0.003 | -0.022 |
| relaxed_cov_p_0.75 | -0.017 | -0.029 |
| relaxed_cov_r_1.0 | -0.007 | -0.009 |
| relaxed_cov_p_1.0 | -0.005 | -0.016 |
| relaxed_cov_r_1.25 | +0.000 | +0.000 |
| relaxed_cov_p_1.25 | +0.000 | +0.000 |

### Relaxation cost / validity

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| init_force | 0.088 | 3.837 | 3.777 |
| n_iter | 7.572 | 57.530 | 56.023 |
| frac_converged | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| init_force | +3.749 | +3.689 |
| n_iter | +49.959 | +48.451 |
| frac_converged | +0.000 | +0.000 |

## test

### Raw (pre-relaxation)

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| amr_r | 0.022 | 0.156 | 0.183 |
| amr_p | 0.033 | 0.203 | 0.203 |
| cov_r_0.5 | 0.996 | 0.973 | 0.954 |
| cov_p_0.5 | 0.993 | 0.919 | 0.932 |
| cov_r_0.75 | 0.998 | 1.000 | 0.988 |
| cov_p_0.75 | 0.998 | 0.996 | 0.995 |
| cov_r_1.0 | 1.000 | 1.000 | 0.998 |
| cov_p_1.0 | 0.999 | 1.000 | 0.999 |
| cov_r_1.25 | 1.000 | 1.000 | 1.000 |
| cov_p_1.25 | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| amr_r | +0.133 | +0.160 |
| amr_p | +0.170 | +0.170 |
| cov_r_0.5 | -0.023 | -0.043 |
| cov_p_0.5 | -0.074 | -0.061 |
| cov_r_0.75 | +0.002 | -0.010 |
| cov_p_0.75 | -0.001 | -0.002 |
| cov_r_1.0 | +0.000 | -0.002 |
| cov_p_1.0 | +0.001 | -0.000 |
| cov_r_1.25 | +0.000 | +0.000 |
| cov_p_1.25 | +0.000 | +0.000 |

### Relaxed (GFN2-xTB)

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| relaxed_amr_r | 0.018 | 0.059 | 0.078 |
| relaxed_amr_p | 0.026 | 0.073 | 0.071 |
| relax_shift | 0.015 | 0.176 | 0.174 |
| relaxed_cov_r_0.5 | 0.995 | 0.975 | 0.955 |
| relaxed_cov_p_0.5 | 0.991 | 0.948 | 0.952 |
| relaxed_cov_r_0.75 | 0.998 | 0.998 | 0.989 |
| relaxed_cov_p_0.75 | 0.999 | 1.000 | 0.997 |
| relaxed_cov_r_1.0 | 1.000 | 1.000 | 0.998 |
| relaxed_cov_p_1.0 | 0.999 | 1.000 | 0.999 |
| relaxed_cov_r_1.25 | 1.000 | 1.000 | 1.000 |
| relaxed_cov_p_1.25 | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| relaxed_amr_r | +0.041 | +0.060 |
| relaxed_amr_p | +0.047 | +0.045 |
| relax_shift | +0.161 | +0.160 |
| relaxed_cov_r_0.5 | -0.020 | -0.041 |
| relaxed_cov_p_0.5 | -0.043 | -0.039 |
| relaxed_cov_r_0.75 | -0.000 | -0.009 |
| relaxed_cov_p_0.75 | +0.001 | -0.002 |
| relaxed_cov_r_1.0 | +0.000 | -0.002 |
| relaxed_cov_p_1.0 | +0.001 | -0.000 |
| relaxed_cov_r_1.25 | +0.000 | +0.000 |
| relaxed_cov_p_1.25 | +0.000 | +0.000 |

### Relaxation cost / validity

| metric | crest | rdkit_t1 | rdkit_t2 |
| --- | --- | --- | --- |
| init_force | 0.110 | 4.072 | 3.900 |
| n_iter | 8.602 | 58.080 | 57.142 |
| frac_converged | 1.000 | 1.000 | 1.000 |

_Gap vs crest:_

| metric | rdkit_t1 | rdkit_t2 |
| --- | --- | --- |
| init_force | +3.963 | +3.790 |
| n_iter | +49.478 | +48.540 |
| frac_converged | -0.000 | +0.000 |
