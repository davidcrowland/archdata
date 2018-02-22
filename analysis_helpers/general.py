from __future__ import print_function # Python 2.x

import os
import numpy as np
import pandas as pd
import h5py
import sys
import math
from fnmatch import fnmatch

# helper functions klusta analysis pipeline

def get_param_file(filename,params_folder):
    found = False
    params = []
    for path, subdirs, files in os.walk(params_folder):
        for name in files:
            for string in filename.split("/"):
                if string in name:
                    found = True
                    params = name
                    return found, params
    return found, params

def n_user(filename):
    #find n-drive username
    n_drive_user = 'non_identified'
    if 'MosersServer' in filename:
        # assuming osx/linux:
        idx = [i for i,x in enumerate(filename.split('/')) if x == 'MosersServer'][0]
        n_drive_user = filename.split('/')[idx+1]
    else:
        # assuming windows:
        idx = [i for i,x in enumerate(filename.split('/')) if x == 'N:'][0]
        n_drive_user = filename.split('/')[idx+1]

    return n_drive_user


def create_export_folders(filename):
    # create export folders
    export_folder = "/".join(filename.split("/")[:-1])+"/KLUSTA/"

    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
        print('Created export folder: {:s}'.format(export_folder))

    export_folder_basesession= "/".join(filename.split("/")[:-1])+"/KLUSTA/base_session"
    if not os.path.exists(export_folder_basesession):
        os.makedirs(export_folder_basesession)
        print('Created export_folder_basesession: {:s}'.format(export_folder_basesession))

    export_folder_othersessions= "/".join(filename.split("/")[:-1])+"/KLUSTA/other_session"
    if not os.path.exists(export_folder_othersessions):
        os.makedirs(export_folder_othersessions)
        print('Created export_folder_othersessions: {:s}'.format(export_folder_othersessions))

    export_folder_othersessions= "/".join(filename.split("/")[:-1])+"/KLUSTA/other_session"
    if not os.path.exists(export_folder_othersessions):
        os.makedirs(export_folder_othersessions)
        print('Created export_folder_othersessions: {:s}'.format(export_folder_othersessions))

    export_folder_lfp = "/".join(filename.split("/")[:-1])+"/KLUSTA/LFP"
    if not os.path.exists(export_folder_lfp):
        os.makedirs(export_folder_lfp)
        print('Created export_folder_lfp: {:s}'.format(export_folder_lfp))

    return export_folder, export_folder_basesession, export_folder_othersessions, export_folder_lfp

def get_clusters(filename,key=None):
    if not key:
        sys.stdout.write('No cluster group key given.')
        sys.exit()

    cluster_group_names = []
    key_clusters =[]
    with h5py.File(filename, mode="r") as f:
        cluster_groups = f['channel_groups/1/cluster_groups/main/'].keys()
        for clusterg in cluster_groups:
            name = f['channel_groups/1/cluster_groups/main/'+clusterg].attrs.values()
            cluster_group_names.append(name[0][0])

        for cluster in f['/channel_groups/1/clusters/main'].iteritems():
            name = f['channel_groups/1/clusters/main/'+cluster[0]].attrs.get('cluster_group')
            if cluster_group_names[int(name)] == key:
                key_clusters.append(int(cluster[0]))
    print('{} clusters: {}'.format(key,key_clusters))
    if not key_clusters:
        print('None found :(')
        sys.exit() # apparently this is a bad way to abort the execution ... but it does the job.
    return key_clusters

def get_clusters_dont_exit(filename,key=None):
    '''
    Had to add this function to debug stimulus artefacts ... it serves no other purpose...
    '''

    if not key:
        sys.stdout.write('No cluster group key given.')
        sys.exit()

    cluster_group_names = []
    key_clusters =[]
    try:
        with h5py.File(filename, mode="r") as f:
            cluster_groups = f['channel_groups/1/cluster_groups/main/'].keys()
            for clusterg in cluster_groups:
                name = f['channel_groups/1/cluster_groups/main/'+clusterg].attrs.values()
                cluster_group_names.append(name[0][0])

            for cluster in f['/channel_groups/1/clusters/main'].iteritems():
                name = f['channel_groups/1/clusters/main/'+cluster[0]].attrs.get('cluster_group')
                if cluster_group_names[int(name)] == key:
                    key_clusters.append(int(cluster[0]))
        print('{} clusters: {}'.format(key,key_clusters))
        if not key_clusters:
            print('None found :(')
    except KeyError as err:
        print('No valid cluster groups found.')

    return key_clusters

def get_basenames(filename):
    with h5py.File(filename, mode="r") as f:
        basenames = f['basenames'][:]
    return basenames

def extract_times(filename):
    with h5py.File(filename, mode="r") as f:
        spiketimes = np.array(f['/channel_groups/1/spikes/time_samples'][:],dtype=float)
        sample_rate = float(f['/application_data/spikedetekt'].attrs.get('sample_rate'))
        time_stamps = np.array(f['/event_types/sessions/events/time_samples'][:],dtype=float)

        # extract the time stamps yet another way ...
        time_stamps_sessions = np.cumsum(np.array(f['/time_stamps_sessions'][:], dtype=float))
        time_stamps_sessions_sample_rate = 96000. # cannot extract that from hdf5???
    print('Extracted spiketimes.')
    return spiketimes,sample_rate,time_stamps, time_stamps_sessions, time_stamps_sessions_sample_rate

def extract_input(filename):
    with h5py.File(filename, mode="r") as f:
        input_data = f['/input/input_data'][:]
        sample_rate_inp = f['/input'].attrs.get('sample_rate_inp')
        status=True
        try:
            sample_rate_inp = float(sample_rate_inp)
        except TypeError as error:
            status=False
        if not status:
            input_data=np.nan;time_inp=np.nan;time_stamps_sessions_input=np.nan;
            sample_rate_inp=np.nan;num_inp_samples=np.nan;duration_inp=np.nan
            return input_data,time_inp,time_stamps_sessions_input,sample_rate_inp,num_inp_samples,duration_inp

        # process:
        time_inp = np.array(input_data['time'],dtype=float)
        time_inp = time_inp/sample_rate_inp*1000. # in ms
        num_inp_samples = len(input_data)

        time_stamps_sessions_input = f['/input/time_stamps_sessions_input'][:]
        time_stamps_sessions_input = np.array(time_stamps_sessions_input,dtype=np.float64)
        time_stamps_sessions_input = np.cumsum(time_stamps_sessions_input)
        time_stamps_sessions_input = time_stamps_sessions_input/sample_rate_inp*1000. # in ms

        duration_inp = (time_stamps_sessions_input)[-1] # in ms

    print('Extracted input data.')
    return input_data,time_inp,time_stamps_sessions_input,sample_rate_inp,num_inp_samples,duration_inp

def extract_waveforms(filename):
    # extract waveforms:
    kwx_filename = filename[:-5] + '.kwx'
    file_kwx = h5py.File(kwx_filename, 'r')
    with h5py.File(kwx_filename, mode="r") as f:
        waveforms = f['/channel_groups/1/waveforms_raw'][:]
    print('Extracted waveforms.')
    return waveforms

def extract_positions(filename):
    with h5py.File(filename, mode="r") as f:
        data_pos = f['/positions/data_pos'][:]
        time_stamps_sessions_pos = np.cumsum(np.array(f['/positions/time_stamps_sessions_pos'][:]),dtype=float)
        timebase_pos = float(f['/positions/'].attrs.get('timebase_pos'))
    print('Extracted tracking data.')
    return data_pos, time_stamps_sessions_pos, timebase_pos
    #print('Found position samples at {} Hz ({} seconds over {} session(s))'.format(timebase_pos,time_stamps_sessions_pos[-1]/timebase_pos,len(time_stamps_sessions_pos)-1))

def extract_lfp(filename):
    with h5py.File(filename, mode="r") as f:
        eeg_raw = f['/eeg/data_eeg'][:]
        samples_sessions_eeg = f['/eeg/samples_sessions_eeg'][:]
        sample_rate_eeg = f['/eeg/'].attrs.get('sample_rate_eeg')
    samples_sessions_eeg = np.cumsum(samples_sessions_eeg)
    print('Extracted LFP data.')
    #print('Found EEG: {} samples at {} Hz ({} s). Separate LFPs recorded: {}'.format(eeg_raw.shape[1],sample_rate_eeg,len_eeg_s,len(eeg_raw)))
    return eeg_raw,samples_sessions_eeg,sample_rate_eeg

def eeg_make_df(eeg_raw,data_pos_df,sample_rate_eeg,timebase_pos):
    eeg_df = pd.DataFrame()
    for eeg_no in xrange(eeg_raw['eeg'].shape[0]):
        eeg_df['eeg{}'.format(eeg_no)] = eeg_raw['eeg'][eeg_no]
    eeg_df['eeg_mean'] = np.mean(eeg_raw['eeg'],axis=0)
    eeg_df['speed'] = np.repeat(data_pos_df['speed_filtered'].values,sample_rate_eeg/timebase_pos)

    eeg_df['time'] = eeg_df.index.values.astype(float)/float(sample_rate_eeg) # in sec
    eeg_df.set_index('time', drop=True, append=False, inplace=True, verify_integrity=False)
    return eeg_df

def sanity_check(spiketimes=None,sample_rate=None,time_stamps=None,time_stamps_sessions=None,time_stamps_sessions_sample_rate=None,
                 waveforms=None,time_stamps_sessions_pos=None,timebase_pos=None,data_pos=None,time_stamps_sessions_input=None,
                 samples_sessions_eeg=None,sample_rate_eeg=None):
    # performs a check of length of sessions etc.

    try:
        sys.stdout.write('\nComparing recorded session lengths...')
        if time_stamps is not None and sample_rate and time_stamps_sessions is not None and time_stamps_sessions_sample_rate:
            length_session1 = float(time_stamps[-1])/sample_rate
            length_session2 = float(time_stamps_sessions[-1])/time_stamps_sessions_sample_rate
            if length_session1 != length_session2:
                sys.stdout.write('\rInconsistency in calculated session lengths.')
                sys.exit()
            else:
                sys.stdout.write('Success.')
        else:
            sys.stdout.write('\rNo basic session information found.')
    except AttributeError:
        sys.stdout.write('\rNo basic session information found.')
    except TypeError:
        sys.stdout.write('\rNo basic session information found.')

    try:
        sys.stdout.write('\nComparing waveform and spike numbers...')
        if waveforms.shape[0] != len(spiketimes): # as many waveforms recorded as spikes?
            sys.stdout.write('\rNumber of recorded waveforms does not match length of recorded spikes.')
            sys.exit()
        else:
            sys.stdout.write('Success.')
    except AttributeError:
        sys.stdout.write('\rNo waveforms loaded.')
    except TypeError:
        sys.stdout.write('\rNo waveforms loaded.')

    try:
        sys.stdout.write('\nComparing recorded session lengths and position record...')
        if time_stamps_sessions_pos is not None and timebase_pos and time_stamps is not None and sample_rate and data_pos is not None:
            if not (time_stamps/sample_rate == time_stamps_sessions_pos/timebase_pos).all():
                sys.stdout.write('\rLength of sessions and position file do not match.')
                sys.exit()
            elif not len(data_pos)/timebase_pos == (time_stamps/sample_rate)[-1]:
                sys.stdout.write('\rLength of sessions and position file do not match.')
                sys.exit()
            else:
                sys.stdout.write('Success.')
        else:
            sys.stdout.write('\rNo position data loaded.')

    except AttributeError:
        sys.stdout.write('\rNo position data loaded.')
    except TypeError:
        sys.stdout.write('\rNo position data loaded.')

    # input data:
    try:
        sys.stdout.write('\nComparing recorded session lengths and input data...')
        if time_stamps is not None and sample_rate and time_stamps_sessions_input is not None:
            time_stamps_session = np.cumsum(time_stamps)/sample_rate
            if (time_stamps_session != np.cumsum(time_stamps_sessions_input)/1000).any():
                sys.stdout.write('\rInconsistency in calculated session lengths (session vs. input)')
                sys.exit()
            else:
                sys.stdout.write('Success.')
        else:
            sys.stdout.write('\rNo basic session or input information found!         ')

    except AttributeError:
        sys.stdout.write('\rNo basic session or input information found!             ')
        sys.exit() # this is fatal ...
    except TypeError:
        sys.stdout.write('\rNo basic session or input information found!             ')
        sys.exit() # this is fatal ...

    # LFP data:
    try:
        sys.stdout.write('\nComparing recorded session lengths and LFP data...')
        if time_stamps is not None and sample_rate and samples_sessions_eeg is not None and sample_rate_eeg:
            if (time_stamps/sample_rate != samples_sessions_eeg/sample_rate_eeg).all():
                sys.stdout.write('\rInconsistency in calculated session lengths (session vs. LFP)')
                sys.exit()
            else:
                sys.stdout.write('Success.')
        else:
            sys.stdout.write('\rNo basic session or LFP information found!')

    except AttributeError:
        sys.stdout.write('\rNo basic session or LFP information found!')
        sys.exit() # this is fatal ...
    except TypeError:
        sys.stdout.write('\rNo basic session or LFP information found!')
        sys.exit() # this is fatal ...


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


print('Loaded analysis helpers: General')
