


def _calculate_dfof_mean_of_percentile(traces: np.ndarray,
                                       frame_rate: float = 30.0,
                                       in_place: bool = False,
                                       offset: float = 0.0,
                                       external_reference: Optional[np.ndarray] = None
                                       ) -> np.ndarray:
    baseline = np.nanmean(sliding_window(traces, int(frame_rate * 30), np.nanpercentile, q=8, axis=-1), axis=0)
    if not in_place:
        dfof = np.zeros_like(traces)
    else:
        dfof = traces.copy()
    for neuron in range(dfof.shape[0]):
        dfof[neuron, :] = (traces[neuron, :] - baseline[neuron]) / baseline[neuron]
    return dfof
