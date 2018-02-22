# Maintanance functions for root_rable
import sys
import re
import h5py
import pickle
from datetime import datetime
import psycopg2
from analysis_helpers.general import *
from database_helpers.psql_start import *

def cast_pickle(data, cur):
    '''
    Function to retrieve binary data from Postgresql
    '''
    if data is None: return None
    return pickle.loads(psycopg2.BINARY(data, cur),encoding='latin1')

def execute_psql(command,fetch):
    '''
    Execute generic PSQL command.
    Define 'fetchone' or 'fetchall'!
    '''
    conn = None
    returned = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute(command)
        if fetch == 'fetchall':
            returned = cur.fetchall()
        elif fetch == 'fetchone':
            returned = cur.fetchone()
        else:
            print('Unrecognized argument "{}"'.format(fetch))
            sys.exit()
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        if "duplicate" in str(error).split(" "):
            print('Duplicate key!')
        else:
            print(error)
    finally:
        if conn is not None:
            conn.close()
    return returned

def distill_params_root_table(filename):
    '''
    Based on the filename get some parameters to sort session for database
    '''
    try:
        animal_id = re.findall(r"\D(\d{5})\D", filename)[0]
    except IndexError as error:
        try:
            # for animal ids that start with 'T'
            animal_id = re.findall(r"\D(T\d{4})\D", filename)[0]
        except IndexError as error:
            try:
                # for 4 digit animals:
                animal_id = re.findall(r"\D(\d{4})\D", filename)[0]
            except IndexError as error:
                try:
                    #desperation:
                    animal_id = re.findall(r"(\d{5})", filename)[0]
                except IndexError as error:
                    animal_id = 'UNIDENTIFIED'

    n_drive_user = n_user(filename) # n_user() is under analysis_helpers.general

    with h5py.File(filename, mode="r") as f:
        date_session = f['/'].attrs.get('date_session')
        time_session = f['/'].attrs.get('time_session')
        date_string = date_session+" "+time_session
    session_ts = datetime.strptime(date_string, '%A, %d %b %Y %H:%M:%S')
    analysis_ts = datetime.now()
    return animal_id,n_drive_user,session_ts,analysis_ts

def count_meta(n_drive_user,animal_id,session_ts):
    '''
    Check if and how many entries for this session timestamp / animal / ndrive user
    already exist in the database
    '''
    sql_command = "SELECT COUNT(*) FROM meta_tb WHERE n_drive_user = '{}' AND animal_id = '{}' AND session_ts = '{}';".format(n_drive_user, animal_id, session_ts)
    meta_rows = execute_psql(sql_command,'fetchone')
    return int(meta_rows[0])


def delete_tetrodes(tetrode_no,n_drive_user,animal_id,session_ts):
    '''
    Check if tetrode exists and delete it completely (all tetrode_nos -> CASCADing to all dependend tables).
    '''
    sql_command = "DELETE FROM tetrodes_tb WHERE tetrode_no = '{}' AND  n_drive_user = '{}' AND animal_id = '{}' AND session_ts = '{}' RETURNING tetrode_no;".format(tetrode_no,n_drive_user,animal_id,session_ts)
    deleted_t = execute_psql(sql_command,'fetchone')
    return deleted_t


def delete_sessions(session_name,n_drive_user,animal_id,session_ts):
    '''
    Check if session name exists and delete it completely.
    '''
    sql_command = "DELETE FROM sessions_tb WHERE session_name = '{}' AND  n_drive_user = '{}' AND animal_id = '{}' AND session_ts = '{}' RETURNING session_name;".format(session_name,n_drive_user,animal_id,session_ts)
    deleted_s = execute_psql(sql_command,'fetchone')
    return deleted_s

######################################################################################################################
######################################################################################################################
# WRITE TO TABLES ####################################################################################################
def to_test_tb(analysis_ts,n_drive_user,animal_id,session_ts):

    '''
    Write full entry to test_tb:

    analysis_ts TIMESTAMP NOT NULL,

    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    '''

    sql_command = """INSERT INTO test_tb(analysis_ts,n_drive_user,animal_id,session_ts)
             VALUES('{}','{}','{}','{}') RETURNING session_ts;""".format(analysis_ts,n_drive_user,animal_id,session_ts)
    id = execute_psql(sql_command,'fetchone')
    return id


def to_meta_tb(analysis_ts,n_drive_user,animal_id,session_ts):

    '''
    Write full entry to meta_tb:

    analysis_ts TIMESTAMP NOT NULL,

    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    '''

    sql_command = """INSERT INTO meta_tb(analysis_ts,n_drive_user,animal_id,session_ts)
             VALUES('{}','{}','{}','{}') RETURNING session_ts;""".format(analysis_ts,n_drive_user,animal_id,session_ts)
    id = execute_psql(sql_command,'fetchone')
    return id


def to_tetrodes_tb(tetrode_no,analysis_ts,filename,n_drive_user,animal_id,session_ts):

    '''
    Write full entry to tetrodes_tb:
    tetrode_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,
    '''

    sql_command = """INSERT INTO tetrodes_tb(tetrode_no,analysis_ts,filename,n_drive_user,animal_id,session_ts)
             VALUES({},'{}','{}','{}','{}','{}') RETURNING tetrode_no;""".format(tetrode_no,analysis_ts,filename,n_drive_user,animal_id,session_ts)
    id = execute_psql(sql_command,'fetchone')
    return id



def to_clusters_tb(cluster_no, analysis_ts, tetrode_no, filename, n_drive_user, animal_id, session_ts):

    '''
    Write full entry to units_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    '''

    sql_command = """INSERT INTO clusters_tb(cluster_no, analysis_ts, tetrode_no, filename, n_drive_user, animal_id, session_ts)
             VALUES({},'{}',{},'{}','{}','{}','{}') RETURNING cluster_no;""".format(cluster_no, analysis_ts, tetrode_no, filename, n_drive_user, animal_id, session_ts)
    id = execute_psql(sql_command,'fetchone')
    return id

def to_sessions_tb(session_name, analysis_ts, n_drive_user, animal_id, session_ts,session_start,session_stop):

    '''
    Write full entry to sessions_tb:
    session_name VARCHAR NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    session_start DOUBLE PRECISION,
    session_stop DOUBLE PRECISION,
    '''

    sql_command = """INSERT INTO sessions_tb(session_name, analysis_ts, n_drive_user, animal_id, session_ts,session_start,session_stop)
             VALUES('{}','{}','{}','{}','{}',{},{}) RETURNING session_name;""".format(session_name, analysis_ts, n_drive_user, animal_id, session_ts,session_start,session_stop)
    id = execute_psql(sql_command,'fetchone')
    return id





def to_spiketimes_tracking_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, spiket_tracking_session, spike_no, mean_freq):

    '''
    Write full entry to spiketimes_tracking_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    UNIQUE (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
    FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

    spiket_tracking_session BYTEA NOT NULL,
    spike_no BIGINT NOT NULL,
    mean_freq DOUBLE PRECISION NOT NULL
    '''

    # Transcribe spiket_tracking_session as binary blob object
    data_tmp = spiket_tracking_session.copy()
    data_tmp.reset_index(level=0, inplace=True)
    data_tmp = cPickle.dumps(data_tmp, -1)

    sql_command = """INSERT INTO spiketimes_tracking_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, spiket_tracking_session, spike_no, mean_freq) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, psycopg2.Binary(data_tmp), spike_no, mean_freq)

    id = execute_psql(sql_command,'fetchone')
    return id

def to_tracking_table(analysis_ts,session_name,n_drive_user,animal_id,session_ts, tracking_session, px_to_cm, head_offset, head_offset_var, speed_cutoff):

    '''
    Write full entry to tracking_tb:
    analysis_ts TIMESTAMP NOT NULL,

    session_name VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    tracking_session BYTEA NOT NULL,
    px_to_cm DOUBLE PRECISION NOT NULL,
    head_offset DOUBLE PRECISION NOT NULL,
    head_offset_var DOUBLE PRECISION NOT NULL,
    speed_cutoff DOUBLE PRECISION NOT NULL
    '''
    id = None
    # check if entry already exists:
    sql_command = """SELECT EXISTS (SELECT true FROM tracking_tb WHERE session_name='{}' AND n_drive_user='{}' AND animal_id = '{}' AND session_ts = '{}');""".format(session_name, n_drive_user, animal_id, session_ts)
    status = execute_psql(sql_command,'fetchone')[0]

    if not status:
        # Transcribe tracking_session as binary blob object
        data_tmp = tracking_session.copy()
        data_tmp.reset_index(level=0, inplace=True)
        data_tmp = cPickle.dumps(data_tmp, -1)

        sql_command = """INSERT INTO tracking_tb(analysis_ts,session_name,n_drive_user,animal_id,session_ts, tracking_session, px_to_cm, head_offset, head_offset_var, speed_cutoff) VALUES('{}','{}','{}','{}','{}',{},{},{},{},{}) RETURNING session_name;""".format(analysis_ts,session_name,n_drive_user,animal_id,session_ts, psycopg2.Binary(data_tmp), px_to_cm, head_offset, head_offset_var, speed_cutoff)
        id = execute_psql(sql_command,'fetchone')

    else:
        print('Tracking_tb entry for {} already exists.'.format(session_name))
    return id

def to_lfp_table(analysis_ts,session_name,n_drive_user,animal_id,session_ts, lfp_session, theta_freq):

    '''
    Write full entry to lfp_tb:
    analysis_ts TIMESTAMP NOT NULL,

    session_name VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    UNIQUE (session_name, n_drive_user, animal_id, session_ts),
    FOREIGN KEY (session_name, n_drive_user, animal_id, session_ts) REFERENCES sessions_tb (session_name, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

    lfp_session BYTEA NOT NULL,
    theta_freq DOUBLE PRECISION NOT NULL
    '''
    id = None
    # check if entry already exists:
    sql_command = """SELECT EXISTS (SELECT true FROM lfp_tb WHERE session_name='{}' AND n_drive_user='{}' AND animal_id = '{}' AND session_ts = '{}');""".format(session_name, n_drive_user, animal_id, session_ts)
    status = execute_psql(sql_command,'fetchone')[0]

    if not status:
        # Transcribe lfp_session as binary blob object
        data_tmp = lfp_session.copy()
        data_tmp.reset_index(level=0, inplace=True)
        data_tmp = cPickle.dumps(data_tmp, -1)

        sql_command = """INSERT INTO lfp_tb(analysis_ts,session_name,n_drive_user,animal_id,session_ts, lfp_session, theta_freq) VALUES('{}','{}','{}','{}','{}',{},{}) RETURNING session_name;""".format(analysis_ts,session_name,n_drive_user,animal_id,session_ts, psycopg2.Binary(data_tmp), theta_freq)
        id = execute_psql(sql_command,'fetchone')

    else:
        print('Lfp_tb entry for {} already exists.'.format(session_name))
    return id



def to_phase_tuning_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, bins_angle_center_phase, hist_angle_smooth_phase,  phase_stats, rayleigh_p, spike_trig_LFP,theta_range):

    '''
    Write full entry to phase_tuning_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    bins_angle_center_phase BYTEA NOT NULL,
    hist_angle_smooth_phase BYTEA NOT NULL,
    phase_stats_MVL DOUBLE PRECISION NOT NULL,
    phase_stats_mean DOUBLE PRECISION NOT NULL,
    phase_stats_var DOUBLE PRECISION NOT NULL,
    rayleigh_p DOUBLE PRECISION NOT NULL,
    spike_trig_LFP BYTEA NOT NULL,
    theta_range_min INTEGER NOT NULL,
    theta_range_max INTEGER NOT NULL


    '''
    # extract from phase_stats
    phase_stats_MVL = phase_stats['MVL']
    phase_stats_mean = phase_stats['mean']
    phase_stats_var = phase_stats['var']

    # Transcribe the following as binary blob object
    # bins_angle_center_phase, hist_angle_smooth_phase, spike_trig_LFP
    bins_angle_center_phase = cPickle.dumps(bins_angle_center_phase, -1)
    hist_angle_smooth_phase = cPickle.dumps(hist_angle_smooth_phase, -1)

    spike_trig_LFP_tmp = spike_trig_LFP.copy()
    spike_trig_LFP_tmp = cPickle.dumps(spike_trig_LFP_tmp, -1)

    # get ints from theta_range
    theta_range_min = int(theta_range[0])
    theta_range_max = int(theta_range[1])

    sql_command = """INSERT INTO phase_tuning_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, bins_angle_center_phase, hist_angle_smooth_phase,  phase_stats_MVL,phase_stats_mean,phase_stats_var, rayleigh_p, spike_trig_LFP,theta_range_min,theta_range_max) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{},{},{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, psycopg2.Binary(bins_angle_center_phase), psycopg2.Binary(hist_angle_smooth_phase), phase_stats_MVL, phase_stats_mean, phase_stats_var, rayleigh_p, psycopg2.Binary(spike_trig_LFP_tmp),theta_range_min,theta_range_max)

    id = execute_psql(sql_command,'fetchone')
    return id


def to_ISI_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, hist_ISI, bin_edges_ISI, ISI_stats):

    '''
    Write full entry to ISI_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    hist_ISI BYTEA NOT NULL,
    bin_edges_ISI BYTEA NOT NULL,
    ISI_stats_contam DOUBLE PRECISION NOT NULL,
    ISI_stats_contam_perc DOUBLE PRECISION NOT NULL,
    ISI_stats_percent_bursts DOUBLE PRECISION NOT NULL
    '''
    # extract from ISI_stats
    ISI_stats_contam = ISI_stats['ISI_contam']
    ISI_stats_contam_perc = ISI_stats['ISI_contam_perc']
    ISI_stats_percent_bursts = ISI_stats['percent_bursts']

    # Transcribe the following as binary blob object
    # hist_ISI, bin_edges_ISI
    hist_ISI = cPickle.dumps(hist_ISI, -1)
    bin_edges_ISI = cPickle.dumps(bin_edges_ISI, -1)

    sql_command = """INSERT INTO ISI_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, hist_ISI, bin_edges_ISI, ISI_stats_contam, ISI_stats_contam_perc, ISI_stats_percent_bursts) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, psycopg2.Binary(hist_ISI), psycopg2.Binary(bin_edges_ISI), ISI_stats_contam, ISI_stats_contam_perc, ISI_stats_percent_bursts)

    id = execute_psql(sql_command,'fetchone')
    return id


def to_st_autocorr_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, st_autocorr,theta_idx,burst_idx1,burst_idx2):

    '''
    Write full entry to st_autocorr_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    st_autocorr BYTEA NOT NULL,
    theta_idx DOUBLE PRECISION NOT NULL,
    burst_idx1 DOUBLE PRECISION NOT NULL,
    burst_idx2 DOUBLE PRECISION NOT NULL
    '''

    # Transcribe st_autocorr as binary blob object
    st_autocorr = cPickle.dumps(st_autocorr, -1)

    sql_command = """INSERT INTO st_autocorr_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, st_autocorr,theta_idx,burst_idx1,burst_idx2) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, psycopg2.Binary(st_autocorr),theta_idx,burst_idx1,burst_idx2)

    id = execute_psql(sql_command,'fetchone')
    return id


def to_hd_tuning_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, bins_angle_center,hist_angle_smooth,tc_stats):

    '''
    Write full entry to hd_tuning_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    bins_angle_center BYTEA NOT NULL,
    hist_angle_smooth BYTEA NOT NULL,

    tc_stats_MVL DOUBLE PRECISION NOT NULL,
    tc_stats_mean DOUBLE PRECISION NOT NULL,
    tc_stats_var DOUBLE PRECISION NOT NULL
    '''

    # extract from tc_stats
    tc_stats_MVL = tc_stats['MVL']
    tc_stats_mean = tc_stats['mean']
    tc_stats_var = tc_stats['var']

    # Transcribe the following as binary blob objects
    # bins_angle_center, hist_angle_smooth
    bins_angle_center = cPickle.dumps(bins_angle_center, -1)
    hist_angle_smooth = cPickle.dumps(hist_angle_smooth, -1)

    sql_command = """INSERT INTO hd_tuning_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, bins_angle_center,hist_angle_smooth,tc_stats_MVL,tc_stats_mean,tc_stats_var) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, psycopg2.Binary(bins_angle_center),psycopg2.Binary(hist_angle_smooth),tc_stats_MVL,tc_stats_mean,tc_stats_var)

    id = execute_psql(sql_command,'fetchone')
    return id



def to_ratemaps_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, nbins,xedges,yedges,masked_ratemap,calc, bin_size, sigma_rate,sigma_time,box_size_cm,speed_cutoff):

    '''
    Write full entry to ratemaps_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    nbins INTEGER NOT NULL,
    xedges BYTEA NOT NULL,
    yedges BYTEA NOT NULL,
    masked_ratemap BYTEA NOT NULL,
    calc BOOLEAN NOT NULL,
    bin_size DOUBLE PRECISION NOT NULL,
    sigma_rate DOUBLE PRECISION NOT NULL,
    sigma_time DOUBLE PRECISION NOT NULL,
    box_size_cm DOUBLE PRECISION NOT NULL,
    speed_cutoff DOUBLE PRECISION NOT NULL
    '''

    # Transcribe the following as binary blob objects
    # xedges, yedges, masked_ratemap
    xedges = cPickle.dumps(xedges, -1)
    yedges = cPickle.dumps(yedges, -1)
    masked_ratemap = cPickle.dumps(masked_ratemap, -1)

    sql_command = """INSERT INTO ratemaps_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, nbins,xedges,yedges,masked_ratemap,calc, bin_size, sigma_rate,sigma_time,box_size_cm,speed_cutoff) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{},{},{},{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, nbins,psycopg2.Binary(xedges),psycopg2.Binary(yedges),psycopg2.Binary(masked_ratemap),calc, bin_size, sigma_rate,sigma_time,box_size_cm,speed_cutoff)

    id = execute_psql(sql_command,'fetchone')
    return id


def to_autocorr_gs_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, autocorr, autocorr_overlap,  grid_valid, grid_score):

    '''
    Write full entry to autocorr_gs_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    autocorr BYTEA NOT NULL,
    autocorr_overlap DOUBLE PRECISION NOT NULL,
    grid_valid BOOLEAN NOT NULL,
    grid_score DOUBLE PRECISION NOT NULL,
    '''

    # Transcribe autocorr as binary blob objects
    autocorr = cPickle.dumps(autocorr, -1)

    sql_command = """INSERT INTO autocorr_gs_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, autocorr, autocorr_overlap, grid_valid, grid_score) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, psycopg2.Binary(autocorr), autocorr_overlap, grid_valid, grid_score)

    id = execute_psql(sql_command,'fetchone')
    return id


def to_waveforms_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,mean_wf, std_wf, maxima_wf):

    '''
    Write full entry to waveforms_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    mean_wf BYTEA NOT NULL,
    std_wf BYTEA NOT NULL,
    maxima_wf BYTEA NOT NULL
    '''

    # Transcribe the following as binary blob objects:
    # mean_wf,std_wf,maxima_wf
    mean_wf = cPickle.dumps(mean_wf, -1)
    std_wf = cPickle.dumps(std_wf, -1)
    maxima_wf = cPickle.dumps(maxima_wf, -1)

    sql_command = """INSERT INTO waveforms_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,mean_wf, std_wf, maxima_wf) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,psycopg2.Binary(mean_wf), psycopg2.Binary(std_wf), psycopg2.Binary(maxima_wf))

    id = execute_psql(sql_command,'fetchone')
    return id

def  to_waveforms_stats_table(cluster_no,tetrode_no,session_name,session_ts,n_drive_user,animal_id, artefact, idx_max_wf, idx_min_wf, swidth):

    '''
    Write full entry to waveforms_stats_tb:
    cluster_no SMALLINT NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    n_drive_user VARCHAR NOT NULL,
    animal_id animal_id VARCHAR NOT NULL,

    artefact INTEGER NOT NULL,
    idx_max_wf INTEGER NOT NULL,
    idx_min_wf INTEGER NOT NULL,
    swidth DOUBLE PRECISION NOT NULL
    '''

    # delete entry first:
    sql_command = "DELETE FROM waveforms_stats_tb WHERE cluster_no = '{}' AND tetrode_no = '{}' AND session_name = '{}' AND  n_drive_user = '{}' AND animal_id = '{}' AND session_ts = '{}' RETURNING cluster_no;".format(cluster_no,tetrode_no,session_name,n_drive_user,animal_id,session_ts)
    id_deleted = execute_psql(sql_command,'fetchone')

    sql_command = """INSERT INTO waveforms_stats_tb(cluster_no,tetrode_no,session_name,session_ts,n_drive_user,animal_id,artefact, idx_max_wf, idx_min_wf, swidth) VALUES({},{},'{}','{}','{}','{}',{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,tetrode_no,session_name,session_ts,n_drive_user,animal_id, artefact, idx_max_wf, idx_min_wf, swidth)
    id = execute_psql(sql_command,'fetchone')
    return id_deleted,id



def to_stimulus_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,analysis_window,SALT_window, sample_rate,sample_rate_inp, stim_params, counter_stimuli, excited, SALT_p, SALT_I, laser_stat, stats_inhib,inhibited,inhib_lowest_p,inhib_lowest_p_interval, change_point_ms):

    '''
    Write full entry to stimulus_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    analysis_window DOUBLE PRECISION NOT NULL,
    SALT_window DOUBLE PRECISION NOT NULL,
    sample_rate DOUBLE PRECISION NOT NULL,
    sample_rate_inp DOUBLE PRECISION NOT NULL,
    IBI DOUBLE PRECISION NOT NULL,
    stim_freq DOUBLE PRECISION NOT NULL,
    stim_length DOUBLE PRECISION NOT NULL,
    counter_stimuli INTEGER NOT NULL,
    excited BOOLEAN NOT NULL,
    SALT_p DOUBLE PRECISION NOT NULL,
    SALT_I DOUBLE PRECISION NOT NULL,
    ex_latency_mean DOUBLE PRECISION NOT NULL,
    ex_latency_median DOUBLE PRECISION NOT NULL,
    ex_latency_var DOUBLE PRECISION NOT NULL,
    ex_latency_reliabil DOUBLE PRECISION NOT NULL,
    stats_p_inhib_10 DOUBLE PRECISION NOT NULL,
    stats_p_inhib_20 DOUBLE PRECISION NOT NULL,
    stats_p_inhib_30 DOUBLE PRECISION NOT NULL,
    stats_p_inhib_40 DOUBLE PRECISION NOT NULL,
    inhibited BOOLEAN NOT NULL,
    inhib_lowest_p DOUBLE PRECISION NOT NULL,
    inhib_lowest_p_interval DOUBLE PRECISION NOT NULL,
    change_point_ms DOUBLE PRECISION NOT NULL
    '''
    # extract info from dictionaries:

    #IBI = DOUBLE PRECISION NOT NULL,
    #stim_freq = DOUBLE PRECISION NOT NULL,
    #stim_length = DOUBLE PRECISION NOT NULL,
    IBI = float(stim_params['IblockI'])
    stim_freq = float(stim_params['stim_freq'])
    stim_length = float(stim_params['stim_length'])

    #ex_latency_mean = DOUBLE PRECISION NOT NULL,
    #ex_latency_median = DOUBLE PRECISION NOT NULL,
    #ex_latency_var = DOUBLE PRECISION NOT NULL,
    #ex_latency_reliabil = DOUBLE PRECISION NOT NULL,
    ex_latency_mean = float(laser_stat['latency_mean'])
    ex_latency_median = float(laser_stat['latency_median'])
    ex_latency_var = float(laser_stat['latency_var'])
    ex_latency_reliabil = float(laser_stat['reliabil'])

    #stats_p_inhib_10 = DOUBLE PRECISION NOT NULL,
    #stats_p_inhib_20 = DOUBLE PRECISION NOT NULL,
    #stats_p_inhib_30 = DOUBLE PRECISION NOT NULL,
    #stats_p_inhib_40 = DOUBLE PRECISION NOT NULL,
    stats_p_inhib_10 = float(stats_inhib[10])
    stats_p_inhib_20 = float(stats_inhib[20])
    stats_p_inhib_30 = float(stats_inhib[30])
    stats_p_inhib_40 = float(stats_inhib[40])


    sql_command = """INSERT INTO stimulus_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, analysis_window,SALT_window,sample_rate,sample_rate_inp,IBI,stim_freq,stim_length,counter_stimuli,excited,SALT_p,SALT_I,ex_latency_mean,ex_latency_median,ex_latency_var,ex_latency_reliabil,stats_p_inhib_10,stats_p_inhib_20,stats_p_inhib_30,stats_p_inhib_40,inhibited,inhib_lowest_p,inhib_lowest_p_interval,change_point_ms) VALUES({},'{}',{},'{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}') RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,analysis_window,SALT_window,sample_rate,sample_rate_inp,IBI,stim_freq,stim_length,counter_stimuli,excited,SALT_p,SALT_I,ex_latency_mean,ex_latency_median,ex_latency_var,ex_latency_reliabil,stats_p_inhib_10,stats_p_inhib_20,stats_p_inhib_30,stats_p_inhib_40,inhibited,inhib_lowest_p,inhib_lowest_p_interval,change_point_ms)

    id = execute_psql(sql_command,'fetchone')
    return id


def to_stimulus_mat_table(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,sample_rate, sample_rate_inp, analysis_window, counter_stimuli, spiketimes_cluster, stimulus_timepoints, sum_1ms,  bin_edges_1ms, binnumber_1ms):

    '''
    Write full entry to stimulus_mat_tb:
    cluster_no SMALLINT NOT NULL,

    analysis_ts TIMESTAMP NOT NULL,

    tetrode_no SMALLINT NOT NULL,
    session_name VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    n_drive_user VARCHAR NOT NULL,
    animal_id INTEGER NOT NULL,
    session_ts TIMESTAMP NOT NULL,

    sample_rate DOUBLE PRECISION NOT NULL,
    sample_rate_inp DOUBLE PRECISION NOT NULL,
    analysis_window DOUBLE PRECISION NOT NULL,
    counter_stimuli INTEGER NOT NULL,
    spiketimes_cluster BYTEA NOT NULL,
    stimulus_timepoints BYTEA NOT NULL,
    sum_1ms BYTEA NOT NULL,
    bin_edges_1ms BYTEA NOT NULL,
    binnumber_1ms BYTEA NOT NULL
    '''

    # Transcribe the following as binary blob objects:
    # spiketimes_cluster, stimulus_timepoints, sum_1ms,bin_edges_1ms,binnumber_1ms
    spiketimes_cluster = cPickle.dumps(spiketimes_cluster, -1)
    stimulus_timepoints = cPickle.dumps(stimulus_timepoints, -1)

    sum_1ms = cPickle.dumps(sum_1ms, -1)
    bin_edges_1ms = cPickle.dumps(bin_edges_1ms, -1)
    binnumber_1ms = cPickle.dumps(binnumber_1ms, -1)

    sql_command = """INSERT INTO stimulus_mat_tb(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts,sample_rate, sample_rate_inp, analysis_window, counter_stimuli, spiketimes_cluster, stimulus_timepoints, sum_1ms, bin_edges_1ms,binnumber_1ms) VALUES({},'{}',{},'{}','{}','{}','{}','{}',{},{},{},{},{},{},{},{},{}) RETURNING cluster_no;""".format(cluster_no,analysis_ts,tetrode_no,session_name,filename,n_drive_user,animal_id,session_ts, sample_rate, sample_rate_inp, analysis_window, counter_stimuli, psycopg2.Binary(spiketimes_cluster), psycopg2.Binary(stimulus_timepoints), psycopg2.Binary(sum_1ms), psycopg2.Binary(bin_edges_1ms),psycopg2.Binary(binnumber_1ms))

    id = execute_psql(sql_command,'fetchone')
    return id
