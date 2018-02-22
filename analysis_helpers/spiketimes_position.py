import pandas as pd
import numpy as np
from general import find_nearest
# converge spiketimes and tracking information

def spiketimes_tracking(spiketimes,sample_rate,boolean,data_pos_df):
    # create spiketimes DF
    spiketimes_cluster = spiketimes[boolean]/sample_rate # in s
    spiketimes_tracking_df = pd.DataFrame(spiketimes_cluster,columns=['time'])
    spiketimes_tracking_df.set_index('time', drop=True, append=False, inplace=True, verify_integrity=False)
    # initialize rest of DF
    spiketimes_tracking_df['speed'] = np.zeros(len(spiketimes_tracking_df))
    spiketimes_tracking_df['speed_filtered'] = np.zeros(len(spiketimes_tracking_df))
    spiketimes_tracking_df['correct_x_inter'] = np.zeros(len(spiketimes_tracking_df))
    spiketimes_tracking_df['correct_y_inter'] = np.zeros(len(spiketimes_tracking_df))
    spiketimes_tracking_df['head_angle'] = np.zeros(len(spiketimes_tracking_df))

    # If filtering is required:
    #idx = find_nearest(spiketimes_cluster_df.index.values,((time_stamps-sample_rate)/sample_rate)[1])
    #spiketimes_cluster_df_shortened = spiketimes_cluster_df.iloc[0:idx+1,:]

    # get speed and position x / y / correct_x (interpolated values):
    for counter,spike_time in enumerate(spiketimes_tracking_df.index.values): # spike times for this cluster
        idx = find_nearest(data_pos_df.index.values,spike_time)

        spiketimes_tracking_df['speed'].values[counter] = data_pos_df['speed'].values[idx]
        spiketimes_tracking_df['speed_filtered'].values[counter] = data_pos_df['speed_filtered'].values[idx]
        spiketimes_tracking_df['correct_x_inter'].values[counter] = data_pos_df['correct_x_inter'].values[idx]
        spiketimes_tracking_df['correct_y_inter'].values[counter] = data_pos_df['correct_y_inter'].values[idx]
        spiketimes_tracking_df['head_angle'].values[counter] = data_pos_df['head_angle'].values[idx]

    return spiketimes_tracking_df

def get_session_indices(basenames,spiketimes_tracking_df,time_stamps,sample_rate):
    '''
    Gets session indices for shortening spiketimes_tracking_df to session length.
    Shortens every session for 1 second since Axona adds 1 second of nonsense to the
    end of (most?) recordings. To be on the save side and not to introduce tracking
    artefacts at session borders.
    
    '''

    indices_session = [0] # start with zero!
    for no_session, session in enumerate(basenames):
        no_session+=1
        idx = np.argmax(spiketimes_tracking_df.index.values > (time_stamps[no_session]/sample_rate)-1)
        indices_session.append(idx)
    indices_session[-1] = len(spiketimes_tracking_df.index)-1 # exchange last value with end index

    return indices_session

print('Loaded analysis helpers: Spiketimes tracking lookup')
