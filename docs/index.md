# Qcrew crewmate

### Installation

```bash
pip install crewmate
```

### Usage

```python
from crewmate.utils import quick_wigner
```

### Examples

Quickly plot Wigner functions.

```python
# import crewmate functions
from crewmate.utils import quick_wigner

# Define system dimensions
q_dim = 2
c_dim = 5

# Define qubit and cavity states
g = [1,0]
fock1 = [0,1,0,0,0]
psi = np.kron(g, fock1)

# Plot Wigner of the state in the cavity
quick_wigner(psi, [q_dim, c_dim])
```
