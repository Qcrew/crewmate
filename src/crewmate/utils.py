import numpy as np


def find_highest_populated_state(state: np.array, tolerance: float = 1e-6) -> int:
    """Find the highest Fock state that is populated for more than the specified tolerance.

    Parameters
    ----------
    state : np.array
        Quantum state in the Fock basis.
    tolerance : float, optional
        Population tolerance, by default 1e-6

    Returns
    -------
    int
        Number of the highest populated-enough state.

    Raises
    ------
    Exception
        If couldn't find a populated-enough state.

    Examples
    --------
    >>> find_highest_populated_state([0.99, 0.01, 0])
    1
    State |1> is the highest populated state, considering the default tolerance of 1e-6.

    >>> find_highest_populated_state([0.99, 0.01, 0], tolerance=0.1)
    0
    State |0> is the highest populated state, considering a tolerance of 0.1.
    """
    # Highest occupied state above tolerance
    highest = -1
    state_abs = np.abs(state) ** 2
    for n in range(len(state)-1, -1, -1):
        if state_abs[n] > tolerance:
            highest = n
            break
    if (highest == -1 or highest == len(state)-1 and state_abs[highest] > tolerance):
        message = ("Couldn't find a Fock state below the target population tolerance.\n" +
                   "Consider increasing the size of the Hilber space.\n")
        if (highest > -1):
            message += (
                f"Target population tolerance: {tolerance}\n" +
                f"Found: {state_abs[highest]} for Fock state |{highest}>"
            )
        raise Exception(message)
    elif (highest == len(state)-1):
        print(f"WARNING: The highest populated state coincides with the system's dimension ({highest}).\n" +
              "There could be higher occupied states that aren't being accounted for.")
    return highest
