# all the lists ..

# sessions
LFP_dict = {'Session LFP':                  'lfp_session',
            'Theta peak frequency [Hz]':    'theta_freq'
            }

pos_dict = {'Session Tracking':             'tracking_session',
            'Px to cm':                     'px_to_cm',
            'Head offset':                  'head_offset',
            'Head offset var':              'head_offset_var'
            }

# CLUSTERS
spiketimes_dict = {'Spiketimes and Tracking':      'spiket_tracking_session',
            'Number of spikes':             'spike_no',
            'Average firing frequency':     'mean_freq'
            }

phase_tuning_dict = {'Theta phase hist center bins':  'bins_angle_center_phase',
            'Theta phase hist smoothed':              'hist_angle_smooth_phase',
            'Phase stats: MVL':                       'phase_stats_MVL',
            'Phase stats: Circ. Mean':                'phase_stats_mean',
            'Phase stats: Variability':               'phase_stats_var',
            'Phase stats: Rayleigh p':                'rayleigh_p',
            'Spike triggered LFP':                    'spike_trig_lfp',
            'Theta range [min]':                      'theta_range_min',
            'Theta range [max]':                      'theta_range_max'
             }

ISI_dict = {'Interspike interval (ISI) hist':      'hist_ISI',
            'Interspike interval (ISI) bins':      'bin_edges_ISI',
            'ISI no. of spikes < 2ms':             'ISI_stats_contam',
            'ISI percent spikes < 2ms':            'ISI_stats_contam_perc',
            'ISI percent in bursts':               'ISI_stats_percent_bursts'
            }

autocorr_dict = {'Spiketime autocorrelations':      'st_autocorr',
            'Theta index':                          'theta_idx',
            'Burst index 1':                        'burst_idx1',
            'Burst index 2':                        'burst_idx2'
            }

hd_tuning_dict = {'HD tuning hist center bins':     'bins_angle_center',
            'HD tuning hist':                       'hist_angle_smooth',
            'HD tuning stats: MVL':                 'tc_stats_MVL',
            'HD tuning stats: Circ. Mean':          'tc_stats_mean',
            'HD tuning stats: Variability':         'tc_stats_var'
            }

ratemaps_dict = {'No. of bins ratemap':             'nbins',
                 'edges x':                         'xedges',
                 'edges y':                         'yedges',
                 'Ratemap (KLUSTA)':               'masked_ratemap',
                 'Spatial autocorr calculated? (boolean)': 'calc',
                 'Bin size [cm]':                   'bin_size',
                 'Sigma rate (2D Gaussian filter)': 'sigma_rate',
                 'Sigma time (2D Gaussian filter)': 'sigma_time',
                 'Box size [cm]':                   'box_size_cm',
                 'Speed cutoff [cm/s]':             'speed_cutoff'
            }

autocorr_gs_dict = {'Spatial autocorrelation (KLUSTA)':     'autocorr',
                    'Autocorrelation overlap':              'autocorr_overlap',
                    'Grid valid? (boolean)':                'grid_valid',
                    'Grid score':                           'grid_score'
            }

waveforms_dict = {'Average waveforms':              'mean_wf',
                  'Standard dev. waveforms':        'std_wf',
                  'Maxima waveforms':               'maxima_wf'
            }

waveforms_stats_dict = {'Artefact (1 or 0)':              'artefact',
                  'Index Max':                      'idx_max_wf',
                  'Index Min':                      'idx_min_wf',
                  'Spike width (s)':                'swidth'

            }


stimulus_dict = {'Analysis window [ms]':              'analysis_window',
                 'SALT window [ms]':                  'SALT_window',
                 'Sample rate spikes':                'sample_rate',
                 'Sample rate input':                 'sample_rate_inp',
                 'Stimulus stats: Inter block interval [ms]': 'IBI',
                 'Stimulus stats: Frequency [Hz]': 'stim_freq',
                 'Stimulus stats: Length [ms]':    'stim_length',
                 'Counter stimuli':                'counter_stimuli',
                 'Excited? (boolean)':             'excited',
                 'SALT p value':                   'SALT_p',
                 'SALT I value':                   'SALT_I',
                 'Excitation latency mean [ms]':   'ex_latency_mean',
                 'Excitation latency median [ms]': 'ex_latency_median',
                 'Excitation latency variability [ms]': 'ex_latency_var',
                 'Excitation latency reliability': 'ex_latency_reliabil',
                 'p value inhibition 10 ms interval':       'stats_p_inhib_10',
                 'p value inhibition 20 ms interval':       'stats_p_inhib_20',
                 'p value inhibition 30 ms interval':       'stats_p_inhib_30',
                 'p value inhibition 40 ms interval':       'stats_p_inhib_40',
                 'Inhibited? (boolean)':           'inhibited',
                 'Inhibition lowest p':            'inhib_lowest_p',
                 'Inhibition lowest p interval':   'inhib_lowest_p_interval',
                 'Inhibition change point (latency)': 'change_point_ms'
            }

stimulus_mat_dict = {'Spiketimes cluster':                'spiketimes_cluster',
                     'Stimulus timepoints':               'stimulus_timepoints',
                     'Histogram 1 ms bins':               'sum_1ms',
                     'Histogram 1 ms bins - edges':       'bin_edges_1ms',
                     'Histogram 1 ms bins - bin number':  'binnumber_1ms'
            }


tracking_tb_BNT_dict = {'Tracking session BNT':                'tracking_session_bnt'
            }

spiket_tracking_tb_BNT_dict = {'Spiketimes Tracking BNT':     'spiketimes_tracking_session_bnt',
                               'Spike number (count) BNT':    'spike_no_bnt'
                                }

# Had this here before:
# stimulus_mat_dict = {'Sample rate spikes':                'sample_rate',
#                      'Sample rate input':                 'sample_rate_inp',
#                      'Analysis window [ms]':              'analysis_window',
#                      'Counter stimuli':                   'counter_stimuli',
#                      'Spiketimes cluster':                'spiketimes_cluster',
#                      'Stimulus timepoints':               'stimulus_timepoints',
#                      'Histogram 1 ms bins':               'sum_1ms',
#                      'Histogram 1 ms bins - edges':       'bin_edges_1ms',
#                      'Histogram 1 ms bins - bin number':  'binnumber_1ms'
#             }


bnt_scores_dict = {'BNT borderscore':                'borderScore',
                   'BNT coherence':                  'coherence',
                   'BNT field main':                 'fieldMain',
                   'BNT grid score':                 'gridScore',
                   'BNT grid stats ellipse theta':   'gridStats_EllipseTheta',
                   'BNT grid stats ellipse 1':       'gridStats_Ellipse_1',
                   'BNT grid stats ellipse 2':       'gridStats_Ellipse_2',
                   'BNT grid stats ellipse 3':       'gridStats_Ellipse_3',
                   'BNT grid stats ellipse 4':       'gridStats_Ellipse_4',
                   'BNT grid stats ellipse 5':       'gridStats_Ellipse_5',
                   'BNT grid stats orientation 1':       'gridStats_Orientation_1',
                   'BNT grid stats orientation 2':       'gridStats_Orientation_2',
                   'BNT grid stats orientation 3':       'gridStats_Orientation_3',
                   'BNT grid stats spacing 1':       'gridStats_Spacing_1',
                   'BNT grid stats spacing 2':       'gridStats_Spacing_2',
                   'BNT grid stats spacing 3':       'gridStats_Spacing_3',
                   'HD peak rate':                   'hdPeakRate',
                   'Information content':            'informationContent',
                   'Information rate':               'informationRate',
                   'Mean rate outside fields':       'meanRateOutsideFields',
                   'HD MVL':                         'mvl',
                   'HD mean direction':              'meanDirection',
                   'HD peak direction':              'peakDirection',
                   'Number of fields':               'numFields',
                   'Peak rate':                      'peakRate',
                   'Mean rate':                      'meanRate',
                   'Selectivity':                    'selectivity',
                   'Sparsity':                       'sparsity',
                   'Speed score':                    'speedscore',
                   'Stability half':                 'stabilityHalf',
                   'Theta idx (spikes)':             'thetaindex'
            }


bnt_dict = {'InformationContent_bnt':           'InformationContent_bnt',
            'tc_stats_hd_peakrate_bnt' :        'tc_stats_hd_peakrate_bnt',
            'gridstats_orientation_bnt':        'gridstats_orientation_bnt',
            'numFields_bnt':                    'numFields_bnt',
            'tc_stats_mean_direction_bnt':      'tc_stats_mean_direction_bnt',
            'meanrate_outside_fields_bnt':      'meanrate_outside_fields_bnt',
            'gridstats_ellipse_bnt':            'gridstats_ellipse_bnt',
            'peak_rate_bnt':                    'peak_rate_bnt',
            'gridstats_spacing_bnt':            'gridstats_spacing_bnt',
            'borderscore_bnt':                  'borderscore_bnt',
            'InformationRate_bnt':              'InformationRate_bnt',
            'speedscore_bnt':                   'speedscore_bnt',
            'tc_stats_peakdirection_bnt':       'tc_stats_peakdirection_bnt',
            'tc_stats_mvl_bnt':                 'tc_stats_mvl_bnt',
            'coherence_bnt':                    'coherence_bnt',
            'grid_score_bnt':                   'grid_score_bnt',
            'fieldmain_bnt':                    'fieldmain_bnt',
            'sparsity_bnt':                     'sparsity_bnt',
            'gridstats_ellipse_theta_bnt':      'gridstats_ellipse_theta_bnt',
            'calbindin_bnt':                    'calbindin_bnt',
            'theta_strength_bnt':               'theta_strength_bnt',
            'selectivity_bnt':                  'selectivity_bnt',
            'mean_rate_bnt':                    'mean_rate_bnt',
            'theta_mean_phase_bnt':             'theta_mean_phase_bnt',
            'stability_half_bnt':               'stability_half_bnt',
            'angular_stability_bnt':            'angular_stability_bnt',
            'masked_ratemap_bnt':               'masked_ratemap_bnt',
            'autocorr_bnt':                     'autocorr_bnt',
            'occupancy_map_bnt':                'occupancy_map_bnt',
            'hist_angle_smooth_bnt':            'hist_angle_smooth_bnt',
            'bins_angle_center_bnt':            'bins_angle_center_bnt',
            'spiketimes_cluster_bnt':           'spiketimes_cluster_bnt',
            'params_bnt':                       'params_bnt',
            'calbindin_dist_bnt':               'calbindin_dist_bnt'

            }
