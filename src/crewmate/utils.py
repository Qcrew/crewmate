import numpy as np
import qctrl
import qctrlvisualizer
import matplotlib.pyplot as plt


def quick_wigner_qctrl(qctrl: qctrl.Qctrl, psi: np.array, c_dim: int, q_dim: int, title="Wigner"):
    """Plot the Wigner function of the state in the cavity starting from the full state (qubit x cavity).

    Parameters
    ----------
    qctrl : qctrl.Qctrl
        qctrl session reference
    psi : np.array
        state in the form qubit x cavity
    c_dim : int
        dimension of the cavity Hilbert space
    q_dim : int
        dimension of the qubit Hilbert space
    title : str, optional
        plot title, by default "Wigner"

    Examples
    --------
    Consider psi = |cavity> x |qubit> = [1,0,0,0,0,0]
    >>> quick_wigner_qctrl(qctrl, [1,0,0,0,0,0], 3, 2)
    """

    graph = qctrl.create_graph()

    # Cavity state
    final_cavity_state = graph.partial_trace(
        graph.outer_product(psi, psi),
        [c_dim, q_dim],
        1,
    )
    # Wigner
    wigner_range = 3
    wigner_density = 128
    # Axes for wigner plot
    position = np.linspace(-wigner_range, wigner_range, wigner_density)
    momentum = np.linspace(-wigner_range, wigner_range, wigner_density)
    # Wigner transform
    graph.wigner_transform(
        final_cavity_state, position, momentum, name="wigner")

    # Simulate system
    simulation = qctrl.functions.calculate_graph(
        graph=graph,
        output_node_names=["wigner"]
    )

    qctrlvisualizer.plot_wigner_function(
        simulation.output["wigner"]["value"], position, momentum)
    plt.suptitle(title)
    plt.show()


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
