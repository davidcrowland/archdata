# General stuff:
import sys
import os
import numpy as np
import math
import pandas as pd
from datetime import date
from tqdm import tqdm_notebook
from scipy.stats import pearsonr

# Plotting:
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


### PLOTTING FUNCTIONS #################################################################################

def fake_ratemap(nbins=40):
    fake_ratemap = np.ones((nbins,nbins))
    masked_ratemap = np.ma.masked_where(fake_ratemap == 1, fake_ratemap)
    return masked_ratemap

def plot_comparisons(scores_df,score,ylabel,separator,figure_size):
    figure = plt.figure(figsize=figure_size,facecolor='w')
    sns.set(font_scale=1.)
    sns.set_style('white')

    ax1 = figure.add_subplot(2,1,1)
    # Draw a nested boxplot to show bills by day and sex
    ax1.axhline(y=0,zorder=0,color='k')
    sns.violinplot(x=separator, y=score, hue='excited',data=scores_df, palette="Greys",split=True,
                bw=.3,axes=ax1)
    sns.stripplot(x=separator, y=score, hue='excited',data=scores_df, palette="Reds",split=True,
                axes=ax1,jitter=.1,alpha=.5)
    ax1.set_title('{}'.format(ylabel),y=1.05)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('')

    handles, labels = ax1.get_legend_handles_labels()
    l = plt.legend(handles[2:4], labels[2:4],loc=1, borderaxespad=0.)

    # do the same for inhibited cells
    ax2 = figure.add_subplot(2,1,2)
    # Draw a nested boxplot to show bills by day and sex
    ax2.axhline(y=0,zorder=0,color='k')
    sns.violinplot(x=separator, y=score, hue='inhibited',data=scores_df, palette="Greys",split=True,
                bw=.3,axes=ax2)
    sns.stripplot(x=separator, y=score, hue='inhibited',data=scores_df, palette="Blues",split=True,
                axes=ax2,jitter=.1,alpha=.5)
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel('')

    handles, labels = ax2.get_legend_handles_labels()
    l = plt.legend(handles[2:4], labels[2:4], loc=1, borderaxespad=0.)
    sns.despine(offset=0, trim=True)
    plt.show()


# draw spikeplots
def draw_path_spike_plots(df,column_names,number_plots,speed_cutoff=5.,offset=0,plot_hd=False, clr='r'):

    '''
    Input:
    - dataframe (df)
    - column names: [tracking_column_name,spiketimes_tracking_column_name]
    - number_plots
    - speed_cutoff (if np.nan: skipped!)
    - offset
    - plot_hd (boolean)
    '''
    figure = plt.figure(figsize=(25,25),facecolor='w')
    for row in tqdm_notebook(np.arange(number_plots)):
        row = row + offset

        current_tracking = df.iloc[row][column_names[0]]
        current_spiket_tracking = df.iloc[row][column_names[1]]

        if np.isnan(speed_cutoff):
            tracking_x = current_tracking['correct_x_inter']
            tracking_y = current_tracking['correct_y_inter']
            spiket_tracking_x = current_spiket_tracking['correct_x_inter']
            spiket_tracking_y = current_spiket_tracking['correct_y_inter']
            if plot_hd: head_angle_ = (current_spiket_tracking.head_angle.values/(2*np.pi))*255
            if row == 0 + offset: print('No speed filter active!')

        else:
            tracking_x = current_tracking['correct_x_inter'][current_tracking['speed_filtered']>speed_cutoff]
            tracking_y = current_tracking['correct_y_inter'][current_tracking['speed_filtered']>speed_cutoff]
            spiket_tracking_x = current_spiket_tracking['correct_x_inter'][(current_spiket_tracking['speed_filtered']>speed_cutoff)]
            spiket_tracking_y = current_spiket_tracking['correct_y_inter'][(current_spiket_tracking['speed_filtered']>speed_cutoff)]
            if plot_hd: head_angle_ = (current_spiket_tracking.head_angle[(current_spiket_tracking['speed_filtered']>speed_cutoff)].values/(2*np.pi))*255

        try:
            ax = figure.add_subplot(5,5,row-offset+1)

            if not plot_hd:
                ax.plot(tracking_x,tracking_y,color=[.6, .6, .6],linewidth=.7,zorder=1)
                ax.scatter(spiket_tracking_x,spiket_tracking_y, color=clr,s=10,alpha=1, zorder=2)
            else:
                ax.plot(tracking_x,tracking_y,color=[.8, .8, .8],linewidth=.7,zorder=1)
                ax.scatter(spiket_tracking_x,spiket_tracking_y,c=head_angle_,s=15,alpha=.8, zorder=4,cmap='jet',lw=0)

        except TypeError as err:
            ax.axis('off');ax.axis('equal')

            continue

        #ax.set_title('{} spikes > {} cm/s'.format(len(current_spiket_tracking.index[current_spiket_tracking['speed_filtered']>speed_cutoff]),
        #                                           speed_cutoff),y=-0.1, fontsize=10)
        ax.set_title('C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no,
        df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")),fontsize=10)

        ax.axis('off');ax.axis('equal')
        ax.set_xlim(np.min(current_tracking['correct_x_inter']),np.max(current_tracking['correct_x_inter']))
        ax.set_ylim(np.min(current_tracking['correct_y_inter']),np.max(current_tracking['correct_y_inter']))
        ax.invert_yaxis()

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.12)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(r'N:\davidcr\stellate paper images\python output\putstellate_pathplot.svg', format='svg', dpi=1000)
    print('Generating plot ...')
    plt.show()


def draw_ratemaps(ratemaps_cells,column_name,number_plots,offset, cmp = 'viridis'):
    figure = plt.figure(figsize=(17,17),facecolor='w')
    df = ratemaps_cells.reset_index()
    for row in tqdm_notebook(np.arange(number_plots)):
        row = row + offset
        ax = figure.add_subplot(5,5,row-offset+1)

        current_ratemap = df[column_name].iloc[row]
        peakRate = np.around(df['peak_rate_bnt'].iloc[row],decimals = 1)
        try:
            ax.imshow(current_ratemap.T, cmap=cmp, interpolation='nearest')
        except AttributeError as err:
            current_ratemap = fake_ratemap()
            ax.imshow(current_ratemap, cmap=cmr, interpolation='nearest')

        ax.axis('off');ax.axis('equal')
        ax.set_xlim(0,current_ratemap.shape[0])
        ax.set_ylim(0,current_ratemap.shape[1])
        ax.invert_yaxis()
        ax.set_title('{} C{} T{} | {} {}'.format(df.iloc[row].animal_id,df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].session_ts.strftime("%b %d, %Y"),peakRate),fontsize=10)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.12)
    print('Generating plot ...')
    plt.savefig(r'N:\davidcr\stellate paper images\python output\putstellate_grids.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.show()

def draw_auto_corrs(df,number_plots,offset):
    figure = plt.figure(figsize=(25,15),facecolor='w')

    for row in tqdm_notebook(range(number_plots)):
        row = row + offset
        sns.set_style('whitegrid')

        ax = figure.add_subplot(5,5,row-offset+1)
        sns.set(font_scale=1.4)

        #a4.yaxis.set_visible(False)
        try:
            ax.bar(np.arange(501) - 250, df.iloc[row].st_autocorr, width=1.001, ec='none',color='black',alpha=.85);
        except ValueError as err:
            continue

        ax.bar(np.arange(501) - 250, df.iloc[row].st_autocorr, width=1.001, ec='none',color='black',alpha=.85);
        ax.set_xlim(-250,250)
        [y1,y2] = ax.get_ylim()
        ax.set_ylim(0,1.1*y2)
        ax.axes.get_yaxis().set_visible(False)
        ax.grid(color='k', linestyle=':', linewidth=1,alpha=.6)
        sns.despine(left=True, bottom=True, right=True,top=True)
        ax.set_title('C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")),fontsize=14,y=1.1)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.8)

    print('Generating plot ...')
    plt.show()


def draw_tuning_curves(df,column_names,number_plots,offset,zero_location,clr='k'):
    figure = plt.figure(figsize=(17,17),facecolor='w')
    df = df.reset_index()
    sns.set_style('darkgrid')

    for row in tqdm_notebook(np.arange(number_plots)):
        row = row + offset
        ax = figure.add_subplot(5,5,row-offset+1,projection='polar')

        bins_angle_center = df.iloc[row][column_names[0]]
        hist_angle_smooth = df.iloc[row][column_names[1]]

        try:
            ax.set_theta_zero_location(zero_location)
            ax.plot(bins_angle_center, hist_angle_smooth,lw=2,color=clr,alpha=.85)
            ax.set_yticklabels([])
            ax.spines['polar'].set_visible(False)

        except AttributeError as err:
            continue
        except ValueError as err:
            print('Not found: C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")))

        ax.set_title('C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")),fontsize=10,y=1.1)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
    print('Generating plot ...')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(r'N:\davidcr\stellate paper images\python output\stellate_grids_theta_tuning.svg', format='svg', dpi=1000, bbox_inches='tight')


    plt.show()


def draw_spiket_LFP(df,number_plots,offset,sample_rate=250.):
    figure = plt.figure(figsize=(17,17),facecolor='w')
    df = df.reset_index()
    sns.set_style('whitegrid')

    for row in tqdm_notebook(np.arange(number_plots)):
        row = row + offset
        ax = figure.add_subplot(5,5,row-offset+1)
        try:
            time_st_lfp = np.arange(df.spike_trig_lfp[row].shape[0])/float(sample_rate)
            time_spike_trig_lfp = time_st_lfp - time_st_lfp[-1]/2

            ax.plot(time_spike_trig_lfp,df.spike_trig_lfp[row].spike_trig_LFP_avg,color='k',alpha=.85, lw=3)
            ax.plot(time_spike_trig_lfp,df.spike_trig_lfp[row].spike_trig_LFP_strong_avg,color='k',alpha=.4, lw=2.5)
            #ax4.plot(time_spike_trig_lfp,base_dataframe.spike_trig_lfp[0].hilbert_angle_avg*20,color='r',alpha=.3, lw=2)
            ylim = ax.get_ylim()
            ax.set_title('-0.2     0.2'.format(),y=-0.17,fontsize=18)
            ax.axvline(x=0,color='k',linestyle=':',lw=1.5)
            ax.set_xlim(-0.2,0.2)
            ax.set_ylim(-50,50)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            sns.despine(left=True)

        except AttributeError as err:
            continue
        except ValueError as err:
            print('Not found: C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")))

        ax.set_title('C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")),fontsize=10,y=1.1)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
    print('Generating plot ...')
    plt.show()

def draw_waveforms(df, column_name, number_plots, offset):
    figure = plt.figure(figsize=(17,17),facecolor='w')
    df = df.reset_index()
    sns.set_style('white')
    sns.set_palette("PuBuGn_d")
    for row in tqdm_notebook(np.arange(number_plots)):
        row = row + offset
        ax = figure.add_subplot(5,5,row-offset+1)
        try:
            ax.plot(df[column_name].iloc[row][0],lw=2.5)
            ax.plot(df[column_name].iloc[row][1],lw=2.5)
            ax.plot(df[column_name].iloc[row][2],lw=2.5)
            ax.plot(df[column_name].iloc[row][3],lw=2.5)

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            sns.despine(left=True,bottom=True)

        except AttributeError as err:
            continue
        except ValueError as err:
            print('Not found: C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")))

        ax.set_title('C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no, df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")),fontsize=10,y=1.1)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
    print('Generating plot ...')
    plt.show()

def create_spike_plots_stimulus(df,number_plots,offset,draw_spikes):
    '''
    Needed in input dataframe:
        - analysis_window
        - sample_rate
        - counter_stimuli
        - stimulus_timepoints
        - spiketimes_cluster
        - sum_1ms
        - stim_freq
        - stim_length
        - "excited"
        - salt_p
        - salt_i
        - ex_latency_median, ex_latency_var, ex_latency_reliabil
        - "inhibited"
        - change_point_ms
    '''

    figure = plt.figure(figsize=(35,35),facecolor='w')
    sns.set(font_scale=2.5)
    sns.set_style('white')
    counter_fig = 1

    df = df.reset_index()

    for row in tqdm_notebook(np.arange(number_plots)):
        row = row + offset

        # continue with rest ...
        window = df.iloc[row].analysis_window
        num_bins = int(df.iloc[row].analysis_window*2*df.iloc[row].sample_rate/1000.)

        # initialize axes:
        ax1 = figure.add_subplot(5,5,row-offset+1)
        counter_fig +=1

        if draw_spikes:
            for no,ts in enumerate(df.iloc[row].stimulus_timepoints):
                spikes = df.iloc[row].spiketimes_cluster[(df.iloc[row].spiketimes_cluster>(ts-df.iloc[row].analysis_window)) &
                                            (df.iloc[row].spiketimes_cluster<(ts+df.iloc[row].analysis_window))]
                ax1.scatter(spikes-ts,np.zeros(len(spikes))+no,s=35,color='k',marker=',',alpha=.7)

        ax1.axvline(x=0,lw=2,color='k',linestyle="-")
        window_f = float(window)
        lower_x = -20.; lower_x_plus = (window_f+lower_x)/(2*window_f)
        upper_x = 40.; upper_x_plus = (window_f+upper_x)/(2*window_f)
        ax1.set_xlim(int(lower_x),int(upper_x))
        ax2 = ax1.twiny()
        if draw_spikes:
            bar = ax2.bar((np.linspace(-int(window),int(window),2*int(window))),
                          df.iloc[row].sum_1ms*(df.iloc[row].counter_stimuli/df.iloc[row].sum_1ms.max()),
                          1,alpha=.15,color='k',align='edge')
        else:
            # make bars a bit darker
            bar = ax2.bar((np.linspace(-int(window),int(window),2*int(window))),
                          df.iloc[row].sum_1ms*(df.iloc[row].counter_stimuli/df.iloc[row].sum_1ms.max()),
                          1,alpha=.75,color='k',align='edge')

        bar[1].set_linewidth(0)
        ax2.axis('off')
        ax2.set_xlim(int(lower_x+lower_x_plus),int(upper_x+upper_x_plus)) # +1 for the whole window !!! if you want to plot them into the same coordinates
        ax1.set_ylim(-5,df.iloc[row].counter_stimuli)
        sns.despine(top=True,bottom=False,left=True)

        if df.iloc[row].excited:
            ax1.axvline(x=df.ex_latency_median.iloc[row],lw=2,color='r',linestyle=":")
            info_bx_laser = ax1.text(0.5, 0.96, 'p: {:.3f} \nI: {:.3f} \nL: {:.2f} \nV: {:.2f}\nR: {:.2f}'.format(df.salt_p.iloc[row],
                                                                                                                   df.salt_i.iloc[row],
                                                                                                                   df.ex_latency_median.iloc[row],
                                                                                                                   df.ex_latency_var.iloc[row],
                                                                                                                   df.ex_latency_reliabil.iloc[row]),
                                        transform=ax2.transAxes, fontsize=25,verticalalignment='top')
            info_bx_laser.set_bbox(dict(color='white', alpha=0.75))
        if df.iloc[row].inhibited:
            ax1.axvline(x=df.change_point_ms.iloc[row],lw=2,color='b',linestyle=":")
            info_bx_laser = ax1.text(0.5, 0.96, 'p: {:.3f} ({})\nL: {:.2f}'.format(df.inhib_lowest_p.iloc[row],
                                                                                   df.inhib_lowest_p_interval.iloc[row],
                                                                                   df.change_point_ms.iloc[row]),
                                        transform=ax2.transAxes, fontsize=25,verticalalignment='top')
            info_bx_laser.set_bbox(dict(color='white', alpha=0.75))

        # display general stimulus information:
        info_bx_laser = ax1.text(0.01, 0.96, 'f: {:.1f} Hz\nl: {:.1f} ms'.format(df.stim_freq.iloc[row],
                                                                                df.stim_length.iloc[row]),
                                        transform=ax2.transAxes, fontsize=25,verticalalignment='top')
        info_bx_laser.set_bbox(dict(color='white', alpha=0.75))

        ax1.set_title('C{} T{} | {} {}'.format(df.iloc[row].cluster_no, df.iloc[row].tetrode_no,
                                               df.iloc[row].animal_id, df.iloc[row].session_ts.strftime("%b %d, %Y")),fontsize=25)


    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.35)
    print('Generating plot...')
    plt.show()




def get_session_indices(basenames,spiketimes_cluster,time_stamps,sample_rate):
    '''
    Copied from analysis_helpers!
    Exchanged the spiketime position dataframe with the direct time index of spikes
    for that cluster (spiketimes_cluster)
    '''
    indices_session = [0] # start with zero!
    for no_session, session in enumerate(basenames):
        no_session+=1
        idx = np.argmax(spiketimes_cluster > (time_stamps[no_session]/sample_rate)-1)
        indices_session.append(idx)
    indices_session[-1] = len(spiketimes_cluster)-1 # exchange last value with end index

    return indices_session

def get_waveforms_stats(spike_chs_wfs,indices_session,no_session):
    '''
    Exact copy from analysis_helpers!
    '''
    # get basic waveform statistics
    # feed in waveforms for this cluster and session indices
    maxima_wf = {}
    mean_wf = {}
    std_wf = {}
    for ch in range(4):
        mean_wf[ch] = np.mean(spike_chs_wfs[ch][indices_session[no_session]:indices_session[no_session+1]],axis=0)
        std_wf[ch] = np.std(spike_chs_wfs[ch][indices_session[no_session]:indices_session[no_session+1]],axis=0)
        maxima_wf[ch] = np.max(mean_wf[ch])
    return mean_wf, std_wf, maxima_wf

def corr_wf_base_laser(df, column_name_base, column_name_laser, plotting=False):
    '''
    Extract waveforms and plot basesession waveform vs. laser waveform;
    Calculate pearson'r between waveforms and return
    '''
    sns.set_style('white')
    all_pears_r = []
    all_pears_p = []

    for no in (range(len(df))):
        if plotting: figure = plt.figure(figsize=(10,10))

        tmp_pears_r = []
        tmp_pears_p = []

        for counter_ch in range(4):
            r,p = pearsonr(df[column_name_base].iloc[no][counter_ch], df[column_name_laser].iloc[no][counter_ch])
            tmp_pears_r.append(r);tmp_pears_p.append(p)

            if plotting:
                ax = figure.add_subplot(2,2,counter_ch+1)
                ax.plot(df[column_name_base].iloc[no][counter_ch],color='k',label='base session',lw=5)
                ax.plot(df[column_name_laser].iloc[no][counter_ch],color=[1,.2,.4],label='laser session',lw=4)
                #if counter_ch == 2: plt.legend()

                ylim = ax.get_ylim()
                ax.text(1,.7*ylim[0],'r: {:.3f}\np: {:.3f}'.format(r,p))
                #ax.text(1,.7*ylim[1],'Ch {}'.format(counter_ch+1),fontweight='bold')

                sns.despine(left=True,bottom=True)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)

        if plotting: plt.show()
        # get median correlation over all channels:
        all_pears_r.append(np.median(tmp_pears_r))
        all_pears_p.append(np.median(tmp_pears_p))
    return all_pears_r,all_pears_p

print('Loaded postgres_analysis_helpers -> general')
