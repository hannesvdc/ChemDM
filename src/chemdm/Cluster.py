import numpy as np

from chemdm.geometry import kabsch_align_numpy

def rmsd_clustering( Z : np.ndarray, 
                    conformers : list[np.ndarray], 
                    rmsd_tol : float = 0.5
                    ) -> tuple[list[np.ndarray], list[int], list[int]]:
    if len(conformers) == 0:
        return conformers, list(), list()
    idx = (Z != 1)
    
    initial_conformer = conformers[0] - np.mean( conformers[0], axis=0, keepdims=True )
    optimal_conformers = [ initial_conformer ]
    indices = [ 0 ]
    cluster_size = [ 1 ]
    for n in range(1, len(conformers)):
        x_conf = conformers[n] - np.mean( conformers[n], axis=0, keepdims=True )

        is_new = True
        for k in range( len(optimal_conformers) ):
            reference_conformer = optimal_conformers[k]

            # Center and Kabsch align
            x_conf_aligned = kabsch_align_numpy( x_conf, reference_conformer, Z )

            # Compute the per-atom RMSD. Ignore hydrogens
            rmsd = np.sqrt(np.mean(np.sum( (x_conf_aligned[idx,:] - reference_conformer[idx,:])**2, axis=1 ) ))
            
            if rmsd <= rmsd_tol:
                is_new = False
                cluster_size[ k ] += 1
                break

        if is_new:
            optimal_conformers.append( x_conf )
            indices.append( n )
            cluster_size.append( 1 )

    return optimal_conformers, indices, cluster_size