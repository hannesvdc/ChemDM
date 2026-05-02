import numpy as np

from dataclasses import dataclass, field

@dataclass
class LSQLogger:
    history: list = field(default_factory=list)
    pending: dict | None = None
    eval_count: int = 0
    iter_count: int = 0
    converged: bool = False

    def observe(self, E_np: np.ndarray, 
                      F_neb: np.ndarray) -> None:
        """Called inside residual evaluation. Stores only the latest evaluation."""
        F_rms_i = np.sqrt((F_neb**2).mean(axis=(1, 2)))

        rel_E = E_np - E_np[0]

        self.pending = { "eval": self.eval_count,
                         "cost": 0.5 * float(np.sum(F_neb.reshape(-1) ** 2)),
                         "max_force_rms": float(F_rms_i.max()),
                         "mean_force_rms": float(F_rms_i.mean()),
                         "barrier_eV": float(rel_E.max()),
        }

        self.eval_count += 1

    def commit(self):
        """Called by least_squares callback once per accepted iteration."""
        if self.pending is None:
            return

        row = { "iter": self.iter_count, **self.pending }
        self.history.append(row)
        self.iter_count += 1

        print( f"Iter {row['iter']:5d}: "
               f"eval {row['eval']:5d}, "
               f"maxF {row['max_force_rms']:.6e}, "
               f"meanF {row['mean_force_rms']:.6e}, "
               f"barrier {row['barrier_eV']:.6f} eV, "
               f"cost {row['cost']:.6e}" )