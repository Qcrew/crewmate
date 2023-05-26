import numpy as np
import qctrlcommons


def define_ladder_operators(graph: qctrlcommons.data_types.Graph, c_dim: int = 1, q_dim: int = 1) -> dict[str, qctrlcommons.node.node_data.Tensor]:
    """Generate the ladder operators for the oscillator and the qubit

    Parameters
    ----------
    graph : qctrl.graphs.Graph
        QCTR graph
    c_dim : int
        Oscillator Hilbert space dimension
    q_dim : int
        Qubit Hilbert space dimension

    Returns
    -------
    { [a, ad, q, qd]: qctrlcommons.node.node_data.Tensor }
        Dictionary containing the ladder operators
    """
    ladder_operators = {
        'a': np.eye(1),
        'ad': np.eye(1),
        'q': np.eye(1),
        'qd': np.eye(1)
    }

    # Oscillator
    if (c_dim > 1):
        a = graph.kronecker_product_list([
            graph.annihilation_operator(c_dim),
            np.eye(q_dim),
        ])

        ad = graph.kronecker_product_list([
            graph.creation_operator(c_dim),
            np.eye(q_dim),
        ])
        ladder_operators['a'] = a
        ladder_operators['ad'] = ad

    # Qubit
    if (q_dim > 1):
        q = graph.kronecker_product_list([
            np.eye(c_dim),
            graph.annihilation_operator(q_dim),
        ])

        qd = graph.kronecker_product_list([
            np.eye(c_dim),
            graph.creation_operator(q_dim),
        ])
        ladder_operators['q'] = q
        ladder_operators['qd'] = qd

    return ladder_operators


# FIXME: super inefficient implementation forced by qctrl. Should find a workaround.
def displacement(graph: qctrlcommons.data_types.Graph, alpha: float, c_dim: int) -> qctrlcommons.node.node_data.Tensor:
    """Generate the matrix associated with the displacement operator D(alpha)

    Parameters
    ----------
    graph : qctrlcommons.data_types.Graph
        QCTRL graph
    alpha : float
        displacement parameter
    c_dim : int
        dimension of the Hilber space

    Returns
    -------
    qctrlcommons.node.node_data.Tensor
        QCTRL tensor containing the operator
    """
    a = graph.annihilation_operator(c_dim)
    ad = graph.creation_operator(c_dim)
    D = graph.multiply(1, np.eye(c_dim))
    term = np.eye(c_dim)
    for i in range(1, 30):
        term = -graph.sum(graph.outer_product(1/i *
                          (alpha * ad - graph.conjugate(alpha) * a), term), 0)
        D = graph.add(D, term)
    return D


def build_snap_sequence_params(graph: qctrlcommons.data_types.Graph, c_dim: int, seq_length: int, D_var: qctrlcommons.node.node_data.Tensor, S_var: qctrlcommons.node.node_data.Tensor, verbose: bool = False) -> list:
    """Bundle the displacement and SNAP parameters in the format used by the apply_D_SNAP_D_sequence function.

    Parameters
    ----------
    graph : qctrlcommons.data_types.Graph
        QCTRL graph
    c_dim : int
        dimension of the Hilbert space
    seq_length : int
        length of the D-SNAP-D sequence (e.g. D-SNAP-D = 3, D-SNAP-D-D-SNAP-D = 6)
    D_var : qctrlcommons.node.node_data.Tensor
        displacement parameters
    S_var : qctrlcommons.node.node_data.Tensor
        SNAP parameters as a 1D flat array
    verbose: bool
        If true prints the gate sequence

    Returns
    -------
    list
        D_SNAP_D parameters
    """
    if ((seq_length-1) % 2 != 0):
        raise Exception(
            f"Invalid sequence length: {seq_length}. It must hold (seq_length-1) % 2 = 0.")
    S_num = int((seq_length-1)/2)
    S_dim = int(S_var.shape[0] / S_num)
    if (S_dim > c_dim):
        raise Exception(
            f"The dimension of SNAP parameters theta ({S_dim}) cannot be greater than Hilber space dimension c_dim ({c_dim})")
    S_fill = np.zeros(c_dim-S_dim)
    params = []
    i, d, s = 0, 0, 0
    for i in range(seq_length):
        if i % 2 == 0:
            if verbose:
                print(f"{i}\tD-{d}")
            params.append(D_var[d])
            d += 1
        else:
            idx = S_dim*s
            params.append(graph.concatenate([S_var[idx:idx+S_dim], S_fill], 0))
            if verbose:
                print(f"{i}\tS-{s}  {S_var[idx:idx+S_dim].shape}")
            s += 1
    return params


def apply_D_SNAP_sequence(graph: qctrlcommons.data_types.Graph, params: np.array or list, c_dim: int, psi_init: qctrlcommons.node.node_data.Tensor or np.array) -> qctrlcommons.node.node_data.Tensor:
    """Apply an interleaved sequence of displacement and SNAP gates (D-SNAP-D...) to psi_init.

    Parameters
    ----------
    graph : qctrlcommons.data_types.Graph
        QCTRL graph
    params : np.array | list
        sequence parameters. Use build_sequence_params to create this list.
    c_dim : int
        dimension of the Hilber space
    psi_init : qctrlcommons.node.node_data.Tensor | np.array
        _description_

    Returns
    -------
    qctrlcommons.node.node_data.Tensor
        State after the application of the gates.
    """
    if ((len(params)-1) % 2 != 0):
        raise Exception(
            "Invalid params vector. It must hold (len(params)-1) % 2 = 0.")
    I = np.eye(c_dim)
    state = psi_init[:, None]
    for i in range(len(params)):
        gate = I
        if i % 2 == 0:
            gate = displacement(graph, params[i], c_dim)
        else:
            gate = I * graph.exp(1j*params[i])
        state = gate @ state
    return state[:, 0]


def define_bandwidth_drive(
    graph: qctrlcommons.data_types.Graph,
    segment_count: int,
    duration: int,
    maximum: float,
    cutoff_frequency: float
) -> qctrlcommons.node.node_data.Pwc:
    """Define an optimizable complex pwc signal limited by the cutoff_frequency.

    Parameters
    ----------
    graph : qctrlcommons.data_types.Graph
        QCTRL graph
    segment_count : int
        number of segments per pulse
    duration : int
        pulse duration [s]
    maximum : float
        drive maximum
    cutoff_frequency : float
        maximum frequency allowed by the drive

    Returns
    -------
    qctrlcommons.node.node_data.Pwc
        QCTRL complex optimizable pwc signal
    """
    # Define raw drive
    raw_drive = graph.utils.complex_optimizable_pwc_signal(
        segment_count=segment_count,
        duration=duration,
        maximum=maximum,
    )
    # Apply sinc cut to raw drives
    drive = graph.utils.filter_and_resample_pwc(
        pwc=raw_drive,
        cutoff_frequency=cutoff_frequency,
        segment_count=segment_count
    )
    return drive
