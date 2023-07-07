from typing import Union
import pickle

import numpy as np
from scipy.linalg import block_diag

import matplotlib.pyplot as plt
from matplotlib import colors

import qutip as qt


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
    In this case state |1> is the highest populated state, considering the default tolerance of 1e-6.
    >>> find_highest_populated_state([0.99, 0.01, 0])
    1

    Now state |0> is the highest populated state, considering a tolerance of 0.1.
    >>> find_highest_populated_state([0.99, 0.01, 0], tolerance=0.1)
    0
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


def cd(alphas: np.array, c_dim: int) -> np.array:
    """Get the conditional displacement matrix in a qubit-cavity system.
    Qubit space dimension is assumed to be 2.

    Parameters
    ----------
    alphas : np.array
        Array containing the values of the displacements
        (e.g. [-1,1] -> if qubit in |g> displace by -1, if qubit in |e> displace by 1).
    c_dim : int
        Dimension of the cavity Hilbert space.

    Returns
    -------
    np.array
        Conditional displacement matrix.

    Examples
    --------
    Create a conditional displacement gate for a qubit-cavity system of dimension 2 x 15.
    >>> cd([-1,1], 15)
    """
    D_gates = [qt.displace(c_dim, alpha) for alpha in alphas]
    return block_diag(*D_gates)


def ecd(alpha: float, chi: float, wait_time: float, c_dim: int) -> np.array:
    """Get ECD gate matrix in a qubit-cavity system,
    assuming the qubit space dimension = 2.
    ECD parameter beta = 2j * alpha * sin(chi * wait_time/2)

    Parameters
    ----------
    alpha : float
        alpha of the first displacement gate ECD sequence.
    chi : float
        Dispersive coupling strength in rad*Hz.
    wait_time : float
        Time between the first 2 displacement gates in the ECD sequence in nanoseconds.
    c_dim : int
        Dimension of the cavity Hilbert space.

    Returns
    -------
    np.array
        ECD gate matrix.
    """
    beta = 2j * alpha * np.sin(chi * wait_time/2)
    D_gates = [qt.displace(c_dim, a) for a in [-beta, beta]]
    return np.flip(block_diag(*D_gates), 0)


def quick_wigner(
    psi: np.array,
    dims: Union[int, np.array, list],
    title: str = "Wigner",
    trace_idx: int = 1,
    x_lim: Union[np.array, list] = [-3, 3],
    y_lim: Union[np.array, list] = [-3, 3],
    pixel_count: int = 200,
    contour_levels: int = 100
) -> None:
    """Plot the Wigner function of psi. By default the system is assumed to be qubit-cavity.

    Parameters
    ----------
    psi : np.array
        State to plot.
    dims : np.arrayorlistorint
        Dimensions of components of the system.
        (e.g. 2 dimensional qubit and 15 dimensional cavity -> dims = [2,15])
    title : str, optional
        Plot title, by default "Wigner"
    trace_idx : int, optional
        The index of the system element that you want to plot, by default 1.
        (e.g. consider a qubit and a cavity with dimensions [2,15],
        if you want to plot the cavity use trance_idx=1)
    x_lim : np.arrayorlist, optional
        Limit on the x axis, by default [-3, 3]
    y_lim : np.arrayorlist, optional
        Limit on the y axis, by default [-3, 3]
    pixel_count : int, optional
        Number of pixels, by default 200
    contour_levels : int, optional
        Number of contour levels, by default 100

    Examples
    --------
    Plot the Wigner function of the cavity.
    psi = |g>|0>, considering qubit dimension = 2 and cavity dimension = 3.
    >>> quick_wigner([1,0,0,0,0,0], [2,3])

    Plot the Wigner function of the qubit.
    psi = |g>|0>, considering qubit dimension = 2 and cavity dimension = 3.
    >>> quick_wigner([1,0,0,0,0,0], [2,3], trace_idx=0)
    """
    if len(x_lim) != 2:
        raise print(f"Length of x_lim must be 2, but got {len(x_lim)}.")
    if len(y_lim) != 2:
        raise print(f"Length of y_lim must be 2, but got {len(y_lim)}.")
    if isinstance(dims, int):
        dims = [dims]
        trace_idx = 0

    x = np.linspace(x_lim[0], x_lim[1], pixel_count)
    y = np.linspace(y_lim[0], y_lim[1], pixel_count)
    state = qt.Qobj(np.array(psi), dims=[dims, [1, 1]])
    state_wigner = qt.wigner(state.ptrace(trace_idx), x, y)
    # Plot
    fig = plt.figure(figsize=(5, 4), dpi=100)
    state_max = np.max(np.abs(np.array(state_wigner)))
    color_norm = colors.TwoSlopeNorm(
        vmin=-state_max, vcenter=0., vmax=state_max)
    contour = plt.contourf(
        x, y, state_wigner, contour_levels, cmap="bwr_r", norm=color_norm)
    fig.colorbar(contour)
    plt.title(title)
    plt.show()


def save_drives_csv(drives: list[np.array], file_name: str) -> None:
    """Save drives as complex numbers in a csv file.

    Parameters
    ----------
    drives : list
        A list containing the drives. Each drive must be a np.array.
        IMPORTANT: if you want to save only one drive specify it as [drive]
    file_name : str
        File name.
    """
    if len(drives) > 5:
        print("WARNING: you are trying to save more than 5 drives" +
              "If you are trying to save just one drive you should specify it as [drive]")
    if file_name.endswith('.csv'):
        file_name = file_name[0:-4]
    np.savetxt(f"{file_name}.csv", drives, delimiter=",")


def read_drives_csv(file_name: str) -> list[np.array]:
    """Read drives from csv file.

    Parameters
    ----------
    file_name : str
        File name.

    Returns
    -------
    np.array
        Drive list.
    """
    if file_name.endswith('.csv'):
        file_name = file_name[0:-4]
    return np.genfromtxt(f"{file_name}.csv", dtype="complex", delimiter=",")


def save_object(obj, file_name: str):
    """Save a python object to a file.

    Parameters
    ----------
    obj : any
        Object to save.
    file_name : str
        File name, including path if needed.
    """
    with open(f'{file_name}', 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def read_object(file_name: str):
    """Read python object from a file

    Parameters
    ----------
    file_name : str
        File name, including path if needed.

    Returns
    -------
    any
        Object
    """
    with open(f'{file_name}', 'rb') as inp:
        return pickle.load(inp)
