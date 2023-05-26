# Qcrew crewmate

### Installation

```bash
pip install crewmate
```

### Usage

```python
from crewmate.utils import quick_wigner_qctrl
```

### Examples

Quick Wigner function using QCTRL

```python
import numpy as np
from qctrl import Qctrl
from crewmate.utils import quick_wigner_qctrl

# Start a Boulder Opal session.
qctrl = Qctrl(organization="yvonne-gaos-team")

# Cavity state
c_dim = 3
cavity_state = np.array([0, 1, 0])
# Qubit state
q_dim = 2
qubit_state = np.array([1, 0])

# Get total state of the system
psi = np.kron([cavity_state], [qubit_state])[0]

# Plot Wigner function of the cavity state
quick_wigner_qctrl(qctrl, psi, c_dim, q_dim)
```
