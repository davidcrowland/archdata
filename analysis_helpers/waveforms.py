# Waveform stats
import numpy as np

def get_waveforms_stats(spike_chs_wfs,indices_session,no_session):
    # get basic waveform statistics
    # feed in waveforms for this cluster and session indices

    maxima_wf = {}
    mean_wf = {}
    std_wf = {}
    for ch in xrange(4):
        mean_wf[ch] = np.mean(spike_chs_wfs[ch][indices_session[no_session]:indices_session[no_session+1]],axis=0)
        std_wf[ch] = np.std(spike_chs_wfs[ch][indices_session[no_session]:indices_session[no_session+1]],axis=0)
        maxima_wf[ch] = np.max(mean_wf[ch])
    return mean_wf, std_wf, maxima_wf

print('Loaded analysis helpers: Waveforms')
