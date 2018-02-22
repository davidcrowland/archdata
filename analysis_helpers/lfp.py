# Work with LFP data
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.signal import spectrogram,welch,detrend,butter,filtfilt,hilbert
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import circmean,circvar

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns

from analysis_helpers.general import find_nearest
from analysis_helpers.spatial_maps import resultant_vector_length, rayleigh, _complex_mean

# filter functions:
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs #nyquist frequency
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calc_PSD_spectrogram(eeg_df_session,spec_max_sec,sample_rate_eeg,speed_cutoff,theta_range, export_folder_lfp,filename, session,plotting):
    # basic setup:
    Fs = float(sample_rate_eeg)
    t = np.arange(0,len(eeg_df_session)/Fs,1/Fs) # time array

    # get theta peak:
    f, psd = welch(eeg_df_session['eeg_mean'][eeg_df_session['speed']>speed_cutoff], sample_rate_eeg, nperseg=sample_rate_eeg*30)
    theta_peak = f[np.argmax(f==theta_range[0])+np.argmax(psd[(f>=theta_range[0]) & (f<=theta_range[1])])]
    print('Theta peak: {:.2f} Hz'.format(theta_peak))

    # continue only if plotting = True
    if plotting:
        # get PSDs
        f_speed, psd_speed = welch(eeg_df_session['eeg_mean'][eeg_df_session['speed']>speed_cutoff], sample_rate_eeg, nperseg=sample_rate_eeg*4)
        f, psd = welch(eeg_df_session['eeg_mean'], sample_rate_eeg, nperseg=sample_rate_eeg*4)
        f_slow, psd_slow = welch(eeg_df_session['eeg_mean'][eeg_df_session['speed']<speed_cutoff], sample_rate_eeg, nperseg=sample_rate_eeg*4)

        # Plot the power spectrum
        # figure properties:
        sns.set()
        sns.set(font_scale=1.1)
        sns.set_style('white')
        figure = plt.figure(figsize=(5,8))
        ax1 = plt.subplot2grid((16,1), (0,0),rowspan=10)

        ax1.semilogy(f,psd,'k',label='all',lw=3)
        ax1.semilogy(f_speed,psd_speed,'k',label='> {:.1f} cm/s'.format(speed_cutoff),linestyle='-',lw=1.5,alpha=.75)
        ax1.semilogy(f_slow,psd_slow,'k',alpha=.5, label='< {:.1f} cm/s'.format(speed_cutoff),lw=2.5)

        ax1.legend(loc='best')
        ax1.axvline(x=theta_peak,color='k',alpha=.75,linestyle='--',lw=1.5)
        ax1.set_xlim((2,45))
        ax1.set_ylim(.8*np.min(psd[(f<45)&(f>2)]),1.2*np.max(psd[f<45]))
        info_bx_theta = ax1.text(3,np.min(psd[(f<45)&(f>2)]),'Theta: {:.2f} Hz'.format(theta_peak),fontsize=12)
        info_bx_theta.set_bbox(dict(color='white', alpha=0.75))

        ax1.set_ylabel('Power [$uV^{2}/Hz$]')
        ax1.set_xlabel('Frequency [Hz]')

        samp_spec = range(0,int(sample_rate_eeg*spec_max_sec)) # take XX seconds only (otherwise too comput. intensive)
        # create spectrogram over raw EEG signal:
        f, t_spec, x_spec = spectrogram(eeg_df_session['eeg_mean'].iloc[samp_spec], fs=int(Fs), window='hanning', nperseg=int(Fs), noverlap=int(Fs-1), mode='psd')
        fmax = 45
        x_mesh, y_mesh = np.meshgrid(t_spec, f[f<fmax])
        ax2 = plt.subplot2grid((16,1), (11,0),rowspan=4)

        cmesh = ax2.pcolormesh(x_mesh+t[samp_spec[0]],y_mesh, np.log10(x_spec[f<fmax]), cmap=cm.viridis,vmin=[np.min(np.log10(x_spec[f<fmax])) if np.max(np.log10(x_spec[f<fmax])) <= 1 else 0][0]) # adjust vmin to see more color range
        ax2.set_ylabel('Frequency [Hz]',x=-0.8)
        ax2.set_xlabel('Time [s]')

        sns.despine(top=True,bottom=True,left=True)
        figure.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=1.3)
        #plt.show()

        # save the figure
        figure.savefig(export_folder_lfp + "/" + "_".join(filename.split("/")[-4:-2]) + '_' + session +'.png',
                    bbox_inches='tight',pad_inches=0,dpi=150)


        plt.close('all')
    return theta_peak

def create_eeg_session(eeg_df_session,sample_rate_eeg,theta_range):
    '''
    Creates new EEG DataFrame from session eeg DataFrame, since the latter is a lice
    of the DataFrame spanning all sessions.
    Performs Butterworth filtering in the theta range (defined in params file)
    and Hilbert transform on that filtered data.

    Returns eeg DataFrame with:
        - time as index (session time in s),
        - session eeg (eeg_mean),
        - butterworth filtered eeg_mean (butter),
        - hilbert transform of filtered data (hilb_angle)
        - amplitude / envelope of filtered data (amp)

    '''

    # play with EEG dataframe... make a new one first because you are dealing with a slice
    # call the new dataframe "eeg"
    eeg = pd.DataFrame(eeg_df_session.eeg_mean.values,columns=['eeg_mean'])
    eeg['time'] = eeg_df_session.index.values
    eeg.set_index('time', drop=True, append=False, inplace=True, verify_integrity=False)

    # Butterworth filter:
    eeg['butter'] = butter_bandpass_filter(eeg.eeg_mean,theta_range[0],theta_range[1],sample_rate_eeg)
    # Hilbert transform of theta filtered signal:
    hilbert_filt = hilbert(eeg.butter)
    eeg['hilb_angle'] = np.angle(hilbert_filt)
    eeg['amp'] = np.abs(hilbert_filt)

    # some statistics useful for filtering:
    median_eeg = np.median(eeg.amp)
    std_eeg = np.std(eeg.amp)
    return eeg,median_eeg,std_eeg

def create_spiket_eeg(spiket_tracking_session,eeg):
    '''
    Creates new spiketime EEG DataFrame out of the session EEG DataFrame and
    the session spiketime DataFrame.
    Looks up the corresponding EEG values for every spike time.

    Returns spiket_eeg DataFrame with:
        - spike times as time index in s
        - speed_filtered (from original spiket_tracking_session DF)
        - eeg_mean
        - butter (butterworth filtered in theta range)
        - hilb_angle (hilbert angle of butterworth filtered eeg_mean) - range [0,2*pi]
        - amp (envelope/amplitude hilbert tranform of butterworth filtered eeg_mean)

    Rayleigh's test of non-uniformity (p and z statistics)

    '''
    # take care of the spike times ...
    # cast it into its own DataFrame as well
    spiket_eeg = pd.DataFrame(spiket_tracking_session.speed_filtered.values,columns=['speed_filtered']) # in case we want to filter for speed
    spiket_eeg['time'] = spiket_tracking_session.index.values
    spiket_eeg.set_index('time', drop=True, append=False, inplace=True, verify_integrity=False)

    # create the additional columns:
    spiket_eeg['eeg_mean'] = np.zeros(len(spiket_eeg))
    spiket_eeg['butter'] = np.zeros(len(spiket_eeg))
    spiket_eeg['hilb_angle'] = np.zeros(len(spiket_eeg))
    spiket_eeg['amp'] = np.zeros(len(spiket_eeg))

    # lookup spike time - eeg:
    for counter,spike_time in enumerate(spiket_eeg.index.values): # spike times for this cluster
        idx = find_nearest(eeg.index.values,spike_time)
        spiket_eeg['eeg_mean'].values[counter] = eeg['eeg_mean'].values[idx]
        spiket_eeg['butter'].values[counter] = eeg['butter'].values[idx]
        spiket_eeg['hilb_angle'].values[counter] = eeg['hilb_angle'].values[idx]
        spiket_eeg['amp'].values[counter] = eeg['amp'].values[idx]
    # transform to 0,2*pi range
    spiket_eeg['hilb_angle'] = (spiket_eeg['hilb_angle'].values + 2*np.pi) % (2*np.pi)

    # perform rayleigh uniformity test on hilbert angles
    rayleigh_p,rayleigh_z = rayleigh(spiket_eeg.hilb_angle)

    return spiket_eeg,rayleigh_p,rayleigh_z

def stats_phase_tuning(eeg_df_session,spiket_tracking_session,sample_rate_eeg,theta_range):
    '''
    Requires output from create_eeg_session() and create_spiket_eeg()

    Creates smoothed angular histogram of phase angles and
    calculates statistics over those angles.
    If Rayleigh p (uniformity test) is > 0.05 calculations are skipped.
    In a last step the spike triggered LFP (average) is calculated.

    Returns:
        - bins_angle_center: Center of angular histogram of phase angles
        - hist_angle_smooth: Savitzky Golay smoothed angular histogram
        - phase_stats: Mean vector length (MVL), angular mean and variance of
                        phase coupling
        - spike triggered LFP

    '''
    eeg,median_eeg,std_eeg = create_eeg_session(eeg_df_session,sample_rate_eeg,theta_range)
    spiket_eeg,rayleigh_p,rayleigh_z = create_spiket_eeg(spiket_tracking_session,eeg)

    phase_stats={}

    hist_angle, bins_angle = np.histogram(spiket_eeg.hilb_angle[spiket_eeg.amp > (median_eeg+.5*std_eeg)],bins=120,range=(0,2*np.pi),density=True)
    bin_half_width = abs((bins_angle[1]-bins_angle[0])/2)
    bins_angle_center = (bins_angle + bin_half_width)[:-1]
    # smooth phase curve:
    hist_angle_for_smooth = np.tile(hist_angle, 3)
    #filter out infs before applying savgol filter:
    idx = np.isfinite(hist_angle_for_smooth)
    if np.sum(idx) > 0:
        hist_angle_smooth = gaussian_filter1d(hist_angle_for_smooth[idx], 2, mode='nearest')
        hist_angle_smooth = hist_angle_smooth[len(hist_angle):2*len(hist_angle)]
        phase_stats['MVL'],phase_stats['mean'], phase_stats['var'], phase_stats['std'] = resultant_vector_length(bins_angle_center,hist_angle_smooth)
    else:
        print('Not enough values found to calculate phase tuning.')
        phase_stats['MVL'] = np.nan; phase_stats['mean'] = np.nan; phase_stats['var'] = np.nan
        bins_angle_center,hist_angle_smooth = np.nan,np.nan

    # get spike triggered LFP as DataFrame (averages!):
    spike_trig_LFP_df = spike_trig_LFP(eeg,spiket_eeg,sample_rate_eeg,median_eeg,std_eeg,window=100)

    return bins_angle_center,hist_angle_smooth,phase_stats,rayleigh_p,spike_trig_LFP_df


def spike_trig_LFP(eeg,spiket_eeg,sample_rate_eeg,median_eeg,std_eeg,window):
    '''
    Calculates spike triggered LFP.
    Input:
        - eeg (session eeg DataFrame)
        - spiket_eeg (session spiketime eeg DataFrame)
        - sample_rate_eeg
        - median and standard dev (of amplitude of hilbert transform)
        - window (how many samples should be evaluated)

    Returns:
        -
    '''
    # transform spike times to EEG samples ...
    first_sample = int(eeg.index.values[0] * sample_rate_eeg)
    spiket_ints = (spiket_eeg.index.values*sample_rate_eeg).astype(int) - first_sample

    # initialize spike triggered LFP average:
    spike_trig_LFP = np.zeros((len(spiket_ints),window*2))
    spike_trig_LFP_strong = np.zeros(window*2)
    # also save hilbert angle ...
    hilbert_angle = np.zeros((len(spiket_ints),window*2))

    for no,spike in enumerate(spiket_ints):
        if (spike-window>0) and (spike+window<len(eeg)):
            spike_trig_LFP[no,:] = eeg.eeg_mean.iloc[spike-window:spike+window]
            hilbert_angle[no,:] = eeg.hilb_angle.iloc[spike-window:spike+window]

            if np.mean(eeg.amp.iloc[spike-window:spike+window]) > (median_eeg+.5*std_eeg):
                spike_trig_LFP_strong = np.vstack((spike_trig_LFP_strong, eeg.eeg_mean.iloc[spike-window:spike+window]))

    spike_trig_LFP_avg = np.mean(spike_trig_LFP,axis=0)
    spike_trig_LFP_strong_avg = np.mean(spike_trig_LFP_strong,axis=0)
    hilbert_angle_avg = np.mean(hilbert_angle,axis=0)

    spike_trig_LFP_df = pd.DataFrame(spike_trig_LFP_avg, columns=['spike_trig_LFP_avg'])
    spike_trig_LFP_df['spike_trig_LFP_strong_avg'] = spike_trig_LFP_strong_avg
    spike_trig_LFP_df['hilbert_angle_avg'] = hilbert_angle_avg

    return spike_trig_LFP_df


print('Loaded analysis helpers: LFP')
