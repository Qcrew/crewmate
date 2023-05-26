import qctrlcommons


def compute_neumann_cost_qctrl(
    graph: qctrlcommons.data_types.Graph,
    duration: float,
    max_amplitude: float,
    drive: qctrlcommons.node.node_data.Pwc
) -> float:
    """Compute Neumann boundary cost. Part of ensuring that the drive has a sufficiently small bandwidth
    is forcing the pulse start and end at zero. So the Neumann boundary cost is proportional to the sum of
    the amplidutes of the pulse in its first and last point.

    Parameters
    ----------
    graph : qctrlcommons.data_types.Graph
        QCTRL graph
    duration : int
        pulse float [s]
    max_amplitude: float
        maximum drive amplitude [rad.Hz]
    drive : qctrlcommons.node.node_data.Pwc
        QCTRL drive

    Returns
    -------
    float
        Neumann cost
    """
    duration -= 1e-12  # Make is within the bounds of the pulse
    start_end_times = [0, duration]
    start_end_cavity_amp = graph.sample_pwc(drive, start_end_times)
    start_end_cavity_amp = graph.abs(start_end_cavity_amp) ** 2
    neumann_cost = graph.sum(start_end_cavity_amp)
    neumann_cost = neumann_cost / (2 * max_amplitude ** 2)  # Normalization
    neumann_cost.name = "Neumann boundary cost"
    return neumann_cost
