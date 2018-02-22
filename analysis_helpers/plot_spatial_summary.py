# PLOT SPATIAL
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl

import seaborn as sns
sns.set()

def plot_spatial_summary(filename,tetrode,cluster,no_session, session, indices_session, mean_wf,std_wf, maxima_wf, time_stamps, sample_rate,spike_chs_wfs,boolean,spiketimes_tracking_df, spike_trig_LFP, sample_rate_eeg, bins_angle_center_phase, hist_angle_smooth_phase, phase_stats, calc, speed_cutoff, bins_angle_center, hist_angle_smooth, masked_ratemap,autocorr,grid_score,grid_valid, tracking_session,spiket_tracking_session, tc_stats, hist_ISI, bin_edges_ISI, ISI_stats,correlograms, theta_idxs,burst_idxs,waveforms, export_folder_basesession, export_folder_othersessions):

    sns.set(font_scale=1.9)
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(24,13))

    # plot waveforms means and standard dev
    for ch in xrange(4):
        ax1 = plt.subplot2grid((16,8), (0,ch),colspan=1,rowspan=4)

        if ch == 0:
            # show main figure title
              ax1.set_title('T: {} | C: {} | S: {} | {} spikes ({:.2f} Hz)'.format(tetrode,cluster, session, len(spiket_tracking_session),
                                                         float(len(spiket_tracking_session.index))/((time_stamps[no_session+1]-time_stamps[no_session])/sample_rate)),y=1.15,x=0,ha='left')
        x = np.linspace(0,50,50)
        # mean_wf, std_wf, maxima_wf
        ax1.fill_between(x,mean_wf[ch]-std_wf[ch],mean_wf[ch]+std_wf[ch], facecolor='#000000',antialiased=True, alpha=0.1)
        ax1.axhline(y=0,color='k',lw=2.5,alpha=0.3)
        ax1.plot(x,mean_wf[ch],color='k',lw=5.5,alpha=0.8)
        ax1.set_ylim([1.5*np.min([mean_wf[x] for x in mean_wf.keys()]), 1.3*np.max([maxima_wf[x] for x in maxima_wf.keys()])])
        ax1.axis('off')


    # plot spikes over time
    ax2 = plt.subplot2grid((20,14), (5,0),colspan=7,rowspan=4)
    ax2.scatter(spiketimes_tracking_df.index,(spike_chs_wfs[0][:,10]).T,color='#cccccc',s=4,alpha=.7)
    ax2.scatter(spiketimes_tracking_df.index,(spike_chs_wfs[1][:,10]).T,color='#333333',s=4,alpha=.7)
    ax2.scatter(spiketimes_tracking_df.index,(spike_chs_wfs[2][:,10]).T,color='#555555',s=4,alpha=.7)
    ax2.scatter(spiketimes_tracking_df.index,(spike_chs_wfs[3][:,10]).T,color='black',s=4,alpha=.7)
    [y1,y2] = ax2.get_ylim()
    yrange = y2-y1
    for no,stamp in enumerate(time_stamps[0:-1]):
        if no==0:
            ax2.text(60, y1-0.1*yrange,'{:.0f} ({:.0f}) s'.format((time_stamps[no_session+1]-time_stamps[no_session])/sample_rate, time_stamps[-1]/sample_rate),ha='left',fontsize=20)
        if no>0:
            ax2.axvline(x=stamp/sample_rate,color='k',lw=2.5,alpha=0.4,linestyle="-")

    ax2.set_ylim(y1-0.3*yrange,y2+0.1*yrange)
    ax2.set_xlim([0,time_stamps[-1]/sample_rate])
    ax2.axes.get_yaxis().set_visible(False)
    ax2.grid(color='k', linestyle=':', linewidth=1,alpha=.6)


    # ISI plot
    ax3 = plt.subplot2grid((20,10), (0,5),colspan=2,rowspan=4)
    # plot ISI:
    ISI_plot = ax3.bar(bin_edges_ISI[:-1],hist_ISI,width=1,color='k',alpha=.85);ISI_plot[0].set_color('r');ISI_plot[1].set_color('r')
    #ylim = ax3.get_ylim()
    #info_ISI = ax3.text(0.48, 0.15, 'Spikes < 2ms: {:3d}\n% in bursts: {:.2f}'.format(ISI_stats['ISI_contam'],ISI_stats['percent_bursts']),
    #                   transform=ax3.transAxes, fontsize=20)
    ax3.set_title('%Cont: {:.1f} | %Burst: {:.1f}'.format(ISI_stats['ISI_contam_perc'],ISI_stats['percent_bursts']),y=1.05, fontsize=20)
    #info_ISI.set_bbox(dict(color='white', alpha=0.75))
    ax3.axes.get_yaxis().set_visible(False)
    ax3.grid(color='k', linestyle=':', linewidth=1,alpha=.6)

    # Autocorr
    if len(correlograms)>0:
        ax4 = plt.subplot2grid((20,10), (5,5),colspan=2,rowspan=4)
        #ax4.yaxis.set_visible(False)
        ax4.bar(np.arange(501) - 250, correlograms[cluster,cluster], width=1.001, ec='none',color='black',alpha=.85);
        ax4.set_title('Theta: {:.2f} | Burst: {:.2f}'.format(theta_idxs[cluster],burst_idxs[cluster]),y=0.94, fontsize=20)
        ax4.set_xlim(-250,250)
        [y1,y2] = ax4.get_ylim()
        ax4.set_ylim(0,1.3*y2)
        ax4.axes.get_yaxis().set_visible(False)
        ax4.grid(color='k', linestyle=':', linewidth=1,alpha=.6)

    sns.despine(left=True, bottom=True, right=True,top=True)

    # spike triggered LFP average:
    # put it on axis 4 as well ... too tired to change that
    time_st_lfp = np.arange(spike_trig_LFP.shape[0])/float(sample_rate_eeg)
    time_spike_trig_lfp = time_st_lfp - time_st_lfp[-1]/2

    ax4 = plt.subplot2grid((20,35), (0,25),colspan=3,rowspan=4)
    ax4.plot(time_spike_trig_lfp,spike_trig_LFP.spike_trig_LFP_avg,color='k',alpha=.85, lw=3)
    ax4.plot(time_spike_trig_lfp,spike_trig_LFP.spike_trig_LFP_strong_avg,color='k',alpha=.4, lw=2.5)
    #ax4.plot(time_spike_trig_lfp,spike_trig_LFP.hilbert_angle_avg*20,color='r',alpha=.3, lw=2)
    ylim = ax4.get_ylim()
    ax4.set_title('-0.2     0.2'.format(),y=-0.17,fontsize=18)
    ax4.axvline(x=0,color='k',linestyle=':',lw=1.5)
    ax4.axhline(y=0,color='k',linestyle='-',lw=2.5,alpha=.75)
    ax4.set_xlim(-0.2,0.2)
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)
    sns.despine(left=True)

    # phase tuning plot
    if not np.isnan(phase_stats['MVL']):
        ax4 = plt.subplot2grid((20,35), (5,25),colspan=3,rowspan=4,projection='polar')
        ax4.set_theta_zero_location("N")
        ax4.plot(bins_angle_center_phase, hist_angle_smooth_phase,lw=2.5,color='k',alpha=.85)
        ax4.set_yticklabels([])
        ax4.set_xticklabels([])
        ax4.set_xticklabels([r'0', '', r'$\frac{1}{2}\pi$', '', r'$\pi$', '',r'$\frac{3}{2}\pi$', ''])
        ax4.spines['polar'].set_visible(False)
        ax4.axvline(x=phase_stats['mean'],color='k',lw=2,alpha=0.7,linestyle='--')
        ax4.set_title('MVL: {:.2f} | {:.1f}$\degree$'.format(phase_stats['MVL'],math.degrees(phase_stats['mean'])),y=-0.45,
                         fontsize=18)


    # peak plot
    ax5 = plt.subplot2grid((18,10), (0,8),colspan=2,rowspan=8)
    ax5.scatter((waveforms[:,10,0]+10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               (waveforms[:,10,1]+10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               s=7,alpha=0.4,color="black")
    ax5.scatter((waveforms[:,10,2]*-1-10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               (waveforms[:,10,1]+10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               s=7,alpha=0.4,color="black")
    ax5.scatter((waveforms[:,10,2]*-1-10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               (waveforms[:,10,3]*-1-10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               s=7,alpha=0.4,color="black")
    ax5.scatter((waveforms[:,10,0]+10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               (waveforms[:,10,3]*-1-10).T[boolean][indices_session[no_session]:indices_session[no_session+1]],
               s=7,alpha=0.4,color="black")
    ax5.axhline(y=0,color='k',lw=2,alpha=0.7,linestyle=':')
    ax5.axvline(x=0,color='k',lw=2,alpha=0.7,linestyle=':')
    # scale a bit:
    [y1,y2] = ax5.get_ylim()
    [x1,x2] = ax5.get_xlim()
    yrange=y2-y1
    xranges=x2-x1
    ax5.set_ylim(y1-0.05*yrange,y2+0.05*yrange)
    ax5.set_xlim(x1-0.05*xranges,x2+0.05*xranges)
    ax5.axis('off')

    ###################################################################################################
    # second row

    ax6 = plt.subplot2grid((2,14), (1,0),colspan=4)
    ax6.plot(tracking_session['correct_x_inter'][tracking_session['speed_filtered']>speed_cutoff],
         tracking_session['correct_y_inter'][tracking_session['speed_filtered']>speed_cutoff],color=[.6, .6, .6],linewidth=1.2,zorder=1)
    ax6.scatter(spiket_tracking_session['correct_x_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)],
            spiket_tracking_session['correct_y_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)],
                color='r',s=15,alpha=1, zorder=2)

    ax6.set_title('{} spikes > {} cm/s'.format(len(spiket_tracking_session.index[spiket_tracking_session['speed_filtered']>speed_cutoff]),
                                               speed_cutoff),y=-0.1, fontsize=20)
    ax6.axis('off');ax6.axis('equal')
    ax6.set_xlim(np.min(tracking_session['correct_x_inter']),np.max(tracking_session['correct_x_inter']))
    ax6.set_ylim(np.min(tracking_session['correct_y_inter']),np.max(tracking_session['correct_y_inter']))
    ax6.invert_yaxis()


    ax7 = plt.subplot2grid((2,14), (1,4),colspan=4)
    ax7.plot(tracking_session['correct_x_inter'][tracking_session['speed_filtered']>speed_cutoff],
         tracking_session['correct_y_inter'][tracking_session['speed_filtered']>speed_cutoff],color=[.8, .8, .8],linewidth=1.2,zorder=1)
    # color code head angle
    # scale head_angles in between 0 and 255
    head_angle_grey = (spiket_tracking_session.head_angle[(spiket_tracking_session['speed_filtered']>speed_cutoff)].values/(2*np.pi))*255

    angleplot = ax7.scatter(spiket_tracking_session['correct_x_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)],
            spiket_tracking_session['correct_y_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)],
                c=head_angle_grey,s=30,alpha=.8, zorder=4,cmap='jet',lw=0)

    ax7.axis('off');ax7.axis('equal')
    ax7.set_xlim(np.min(tracking_session['correct_x_inter']),np.max(tracking_session['correct_x_inter']))
    ax7.set_ylim(np.min(tracking_session['correct_y_inter']),np.max(tracking_session['correct_y_inter']))
    ax7.invert_yaxis()

    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    cbar = fig.colorbar(angleplot,cax=cax)
    cbar.set_clim(0, 255)
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['0','180','360']):
        cbar.ax.text(1.2, (2 * j) / 4.0, lab, ha='left', va='top',fontsize=20,rotation=270)

    if calc:
        ax8 = plt.subplot2grid((2,14), (1,8),colspan=4)
        ax8.imshow(masked_ratemap.T, cmap=cm.jet, interpolation='nearest')
        ax8.axis('off');ax8.axis('equal')
        ax8.set_xlim(0,masked_ratemap.shape[0])
        ax8.set_ylim(0,masked_ratemap.shape[1])
        ax8.invert_yaxis()
        ax8.set_title('GS: {:.2f} ({})'.format(grid_score,['valid' if grid_valid else 'invalid'][0]),fontsize=20,y=-0.1)

    # hd_tuning
    sns.set_style('darkgrid')
    ax9 = plt.subplot2grid((2,14), (1,12),colspan=3,projection='polar')
    ax9.set_theta_zero_location("S")

    ax9.plot(bins_angle_center, hist_angle_smooth,lw=3,color='k',alpha=.85)
    ax9.set_yticklabels([])
    ax9.spines['polar'].set_visible(False)
    sns.set(font_scale=1.3)

    if tc_stats['var'] < 2:
        ax9.axvline(x=tc_stats['mean'],color='k',lw=4,alpha=0.5,linestyle=':')
        ax9.set_title('MVL: {:.2f} | {:.1f}$\degree$'.format(tc_stats['MVL'],math.degrees(tc_stats['mean'])),y=-0.78,
                     fontsize=20)
    else:
        ax9.set_title('MVL: {:.2f}'.format(tc_stats['MVL']),y=-0.78, fontsize=20)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=None)

    if no_session == 0: # basessession
        fig.savefig(export_folder_basesession + "/" +"_".join(filename.split("/")[-4:-2]) + '_' + session + '_T' + str(tetrode) + '_' + str(cluster) +'.png',
                    bbox_inches='tight',pad_inches=0,dpi=150)
    else:
        fig.savefig(export_folder_othersessions + "/" +"_".join(filename.split("/")[-4:-2]) + '_' + session + '_T' + str(tetrode) + '_' + str(cluster) +'.png',
                   bbox_inches='tight', pad_inches=0,dpi=150)
    #plt.show()
    plt.close('all')

    return True

print('Loaded analysis helpers: Plot spatial summary')
