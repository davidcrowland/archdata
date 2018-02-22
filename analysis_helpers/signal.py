# general signal functions ...
import sys
import numpy as np
from scipy.signal import convolve2d
import h5py
from matplotlib import pyplot as plt
import math
import warnings


# 2D convolution using the convolution's FFT property
def conv2(a,b):
    ma,na = a.shape
    mb,nb = b.shape
    return np.fft.ifft2(np.fft.fft2(a,[2*ma-1,2*na-1])*np.fft.fft2(b,[2*mb-1,2*nb-1]))

# compute a normalized 2D cross correlation using convolutions
# this will give the same output as matlab, albeit in row-major order
def normxcorr2(b,a):
    c = convolve2d(a,np.flipud(np.fliplr(b)))
    a = convolve2d(a**2, np.ones(b.shape))
    b = np.sum(b.flatten()**2)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            c = c/np.sqrt(a*b)
        except Warning as e:
            c = np.zeros((c.shape[0],c.shape[1]))
    return c

def calc_ISI(spiket_tracking_session):
  # takes care of inter stimulus interval stats

  ISI_stats = {}
  hist_ISI, bin_edges_ISI = np.histogram(np.diff(spiket_tracking_session.index*1000),bins=50,range=(0,50))
  # fraction of spikes with ISI values >= 2 and < 10 ms
  ISI_stats['percent_bursts'] = 100*np.sum(hist_ISI[2:10])/float(len(spiket_tracking_session))
  # contamination with spikes in the first 2 ms
  ISI_stats['ISI_contam'] = np.sum(hist_ISI[:2])
  ISI_stats['ISI_contam_perc'] = 100*np.sum(hist_ISI[:2])/float(len(spiket_tracking_session))

  return hist_ISI, bin_edges_ISI, ISI_stats


def calc_autocorr_idxs(good_clusters,correlograms,theta_peak_win=[100,140],theta_trough_win=[50,70], burst_peak_win1=[0,10], burst_baseline_win1=[40,50], burst_peak_win2=[3,6], burst_baseline_win2=[150,250], downsample=5.):
    '''
    Calculate theta index as (peak-trough) / (peak+trough), whereby peak and trough are the averages over windows
    in the spiketime autocorrelation of that neuron.
    A downsampling of 5 (standard) will give 5ms windows (more robust than 1ms windows for sparse autocorrs).
    Windows are adapted from BNT code (thetaModulationIndex.m)

    Calculate burst index1 as described in Buzsaki paper:
    Control of timing, rate and bursts of hippocampal place cells by dendritic and somatic inhibition - 2012 Nature Neuro
    http://www.nature.com/neuro/journal/v15/n5/full/nn.3077.html

    Calculate burst index2 as described in Buzsaki paper:
    Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells
    http://dx.doi.org/10.1016/j.neuron.2016.12.011

    '''

    theta_idxs = {}
    burst_idxs1 = {}
    burst_idxs2 = {}

    for cluster in good_clusters:
        correlogram = np.array(correlograms[cluster,cluster],dtype=float)
        correlogram = np.array(correlogram[(len(correlogram)/2):],dtype=float) # take half of it

        # Calculate burst idx1
        # it is silently assumed that these are 1 ms bins - don't change that (make it compatible with Buzsaki paper)
        burst_peak1 = np.max(correlogram[burst_peak_win1[0]:burst_peak_win1[1]])
        burst_baseline1 = np.mean(correlogram[burst_baseline_win1[0]:burst_baseline_win1[1]])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                if burst_peak1 >= burst_baseline1:
                    burst_idx1 = (burst_peak1-burst_baseline1)/burst_peak1
                else:
                    burst_idx1 = (burst_peak1-burst_baseline1)/burst_baseline1
            except Warning as e:
                burst_idx1 = np.nan

        burst_idxs1[cluster] = burst_idx1

        # Calculate burst idx2
        # it is silently assumed that these are 1 ms bins - don't change that (make it compatible with Buzsaki paper)
        burst_peak2 = np.mean(correlogram[burst_peak_win2[0]:burst_peak_win2[1]])
        burst_baseline2 = np.mean(correlogram[burst_baseline_win2[0]:burst_baseline_win2[1]])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                burst_idx2 = burst_peak2/burst_baseline2
            except Warning as e:
                burst_idx2 = np.nan

        burst_idxs2[cluster] = burst_idx2

        # downsample the correlogram (1-> 5 ms bins)
        pad_size = math.ceil(float(correlogram.size)/downsample)*downsample - correlogram.size
        correlogram = np.append(correlogram, np.zeros(int(pad_size))*np.NaN)
        correlogram = np.sum(correlogram.reshape(-1,int(downsample)),axis=1)

        peak =  np.mean(correlogram[int(theta_peak_win[0]/downsample)-1:int(theta_peak_win[1]/downsample)-1])
        trough = np.mean(correlogram[int(theta_trough_win[0]/downsample)-1:int(theta_trough_win[1]/downsample)-1])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                th_index = (peak-trough) / (peak+trough)
            except Warning as e:
                th_index = np.nan

        theta_idxs[cluster] = th_index

    return theta_idxs,burst_idxs1,burst_idxs2

def get_all_crosscorrelations(filename,spiketimes,good_clusters,sample_rate,session_no=[],basenames=[], time_stamps=[], ncorrbins=500,corrbin=0.001,show_fig=True):
    '''
    Calculate crosscorrelations between all pairs of neurons.
    Code copied / adapted from the official Klustaviewa repository
    '''

    spiketimes = spiketimes/sample_rate

    with h5py.File(filename, mode="r") as f:
        labels = np.array(f['/channel_groups/1/spikes/clusters/main'][:])

    spiketimes_good = np.array([st for no,st in enumerate(spiketimes) if labels[no] in good_clusters],dtype=float)
    label_good = np.array([lab for lab in labels if lab in good_clusters])

    # check if there are session labels.
    # if so, calculate crosscorrs and indices for only this session, not over all sessions
    calc = True

    if not session_no==[]:
        sys.stdout.write('Calculating crosscorrs for: {}  ... '.format(basenames[session_no]))
        indices_session = [0] # start with zero!
        for no_session, session in enumerate(basenames):
            no_session+=1
            idx = np.argmax(spiketimes_good > (time_stamps[no_session]/sample_rate))
            indices_session.append(idx)
        indices_session[-1] = len(spiketimes_good)-1 # exchange last value with end index
        # shorten:
        spiketimes_good = spiketimes_good[indices_session[session_no]:indices_session[session_no+1]]
        label_good = label_good[indices_session[session_no]:indices_session[session_no+1]]
        # safety check:
    for clus_ in good_clusters:
        if clus_ not in label_good:
            print('Label for {} not found in current crosscorr initialization. Skipping.'.format(clus_))
            calc = False

    if calc:
        correlograms = compute_correlograms(spiketimes_good, label_good, ncorrbins=ncorrbins, corrbin=corrbin,sample_rate = sample_rate)
        print('Computed spiketime crosscorrelations.')

        # calculate theta indices from autocorrelations:
        theta_idxs,burst_idxs1,burst_idxs2 = calc_autocorr_idxs(good_clusters,correlograms,theta_peak_win=[100,140],theta_trough_win=[50,70], burst_peak_win1=[0,10], burst_baseline_win1=[40,50], burst_peak_win2=[3,6], burst_baseline_win2=[150,250], downsample=5.)

    else:
        theta_idxs,burst_idxs1,burst_idxs2 = np.nan,np.nan,np.nan
        correlograms = []

    if show_fig and calc:
        rows_subplt = (len(good_clusters)+[0 if np.mod(len(good_clusters),2)==0 else 1][0])/2
        height_fig = int((len(good_clusters)/7.))
        if height_fig == 0: height_fig = .5

        fig = plt.figure(figsize=(12,10*height_fig))

        counter= 0
        for cluster in good_clusters:
                counter +=1
                ax1 = fig.add_subplot(rows_subplt, 2,counter)
                ax1.bar(np.arange(ncorrbins+1) - ncorrbins/2, correlograms[cluster,cluster], width=1, ec='none',color='black',alpha=.85);
                ax1.set_title('{}, theta idx: {:.2f}, burst idx1: {:.2f}, burst idx2: {:.2f}'.format(cluster,theta_idxs[cluster],burst_idxs1[cluster],burst_idxs2[cluster]),fontsize=10)
                ax1.set_xlim(-ncorrbins/2,ncorrbins/2)
                ax1.yaxis.set_visible(False)
                ticklines = ax1.get_xticklines()
                for line in ticklines:
                    line.set_linewidth(1)
                    line.set_alpha(.2)

        fig.subplots_adjust(wspace=0.14, hspace=0.47)
        plt.show()
    return correlograms, theta_idxs,burst_idxs1,burst_idxs2

### KLUSTAVIEWA SOURCE CODE :
from .ccg import correlograms


def compute_correlograms(spiketimes,
                         clusters,
                         clusters_to_update=None,
                         ncorrbins=None,
                         corrbin=None,
                         sample_rate=None,
                         ):

    if ncorrbins is None:
        ncorrbins = NCORRBINS_DEFAULT
    if corrbin is None:
        corrbin = CORRBIN_DEFAULT

    # Sort spiketimes for the computation of the CCG.
    ind = np.argsort(spiketimes)
    spiketimes = spiketimes[ind]
    clusters = clusters[ind]

    window_size = corrbin * ncorrbins

    # unique clusters
    clusters_unique = np.unique(clusters)

    # clusters to update
    if clusters_to_update is None:
        clusters_to_update = clusters_unique

    # Select requested clusters.
    ind = np.in1d(clusters, clusters_to_update)
    spiketimes = spiketimes[ind]
    clusters = clusters[ind]

    assert spiketimes.shape == clusters.shape
    assert np.all(np.in1d(clusters, clusters_to_update))
    assert sample_rate > 0.
    assert 0 < corrbin < window_size

    C = correlograms(spiketimes,
                     clusters,
                     cluster_ids=clusters_to_update,
                     sample_rate=sample_rate,
                     bin_size=corrbin,
                     window_size=window_size,
                     )
    dic = {(c0, c1): C[i, j, :]
           for i, c0 in enumerate(clusters_to_update)
           for j, c1 in enumerate(clusters_to_update)}
    return dic


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
NCORRBINS_DEFAULT = 101
CORRBIN_DEFAULT = .001


# -----------------------------------------------------------------------------
# Computing one correlogram
# -----------------------------------------------------------------------------
def compute_one_correlogram(spikes0, spikes1, ncorrbins, corrbin):
    clusters = np.hstack((np.zeros(len(spikes0), dtype=np.int32),
                          np.ones(len(spikes1), dtype=np.int32)))
    spikes = np.hstack((spikes0, spikes1))
    # Indices sorting the union of spikes0 and spikes1.
    indices = np.argsort(spikes)
    C = compute_correlograms(spikes[indices], clusters[indices],
        ncorrbins=ncorrbins, corrbin=corrbin)
    return C[0, 1]


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
def get_baselines(sizes, duration, corrbin):
    baselines = (sizes.reshape((-1, 1)) * sizes.reshape((1, -1))
                    * corrbin / (duration))
    return baselines

# Utility functions
def excerpt_step(nsamples, nexcerpts=None, excerpt_size=None):
    step = max((nsamples - excerpt_size) // (nexcerpts - 1),
               excerpt_size)
    return step

def excerpts(nsamples, nexcerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    step = excerpt_step(nsamples,
                        nexcerpts=nexcerpts,
                        excerpt_size=excerpt_size)
    for i in range(nexcerpts):
        start = i * step
        if start >= nsamples:
            break
        end = min(start + excerpt_size, nsamples)
        yield start, end

def get_excerpts(data, nexcerpts=None, excerpt_size=None):
    nsamples = data.shape[0]
    return np.concatenate([data[start:end,...]
                          for (start, end) in excerpts(nsamples,
                                                       nexcerpts=nexcerpts,
                                                       excerpt_size=excerpt_size)],
                          axis=-1)

print('Loaded analysis helpers: Signal')
