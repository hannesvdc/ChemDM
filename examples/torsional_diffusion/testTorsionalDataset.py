
from pathlib import Path
import torch as pt

from chemdm.TorsionalDataset import TorsionalDataset
from chemdm.TorsionalDiffusionSampling import collate_torsional

if __name__ == '__main__':
    """ Simple testing routine. """
    from dotenv import load_dotenv
    load_dotenv()
    import os
    qm9_folder = Path( os.environ["QM9_FOLDER"] )

    data_folder = qm9_folder.parent / "parsed" / "train.pt"
    ds = TorsionalDataset( data_folder )
    data1 = ds[1001]
    data2 = ds[2]
    collated = collate_torsional( [data1, data2] )

    print( "ex1 N:", int(data1["mol"].Z.numel()), "  ex2 N:", int(data2["mol"].Z.numel()) )
    print( "batch N_total:", int(collated["mol"].Z.numel()) )
    print( "batch m_total:", int(collated["rotatable_bonds"].shape[0]) )
    print( "molecule_id unique:", pt.unique(collated["mol"].molecule_id).tolist() )