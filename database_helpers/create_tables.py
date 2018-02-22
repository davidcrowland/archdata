# Initialize tables

from database_helpers.psql_start import *


def create_table_test():
    """ create test table """
    commands = [
        """
        CREATE TABLE test_tb (
            analysis_ts TIMESTAMP NOT NULL,

            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,
            PRIMARY KEY (n_drive_user, animal_id, session_ts)
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_meta():
    """ create meta table """
    commands = [
        """
        CREATE TABLE meta_tb (
            analysis_ts TIMESTAMP NOT NULL,

            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,
            PRIMARY KEY (n_drive_user, animal_id, session_ts)
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_tetrodes():
    """ create tetrodes table """
    commands = [
        """
        CREATE TABLE tetrodes_tb (
            tetrode_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,
            PRIMARY KEY (tetrode_no, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (n_drive_user, animal_id, session_ts) REFERENCES meta_tb (n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_clusters():
    """ create clusters table """
    commands = [
        """
        CREATE TABLE clusters_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts),

            FOREIGN KEY (tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES tetrodes_tb (tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_sessions():
    """ create sessions table """
    commands = [
        """
        CREATE TABLE sessions_tb (
            session_name VARCHAR NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            session_start DOUBLE PRECISION,
            session_stop DOUBLE PRECISION,

            PRIMARY KEY (session_name, n_drive_user, animal_id, session_ts),

            FOREIGN KEY (n_drive_user, animal_id, session_ts) REFERENCES meta_tb (n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


###############################################################################

def create_table_lfp():
    """ create LFP (session) table """
    commands = [
        """
        CREATE TABLE lfp_tb (
            analysis_ts TIMESTAMP NOT NULL,

            session_name VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (session_name, n_drive_user, animal_id, session_ts) REFERENCES sessions_tb (session_name, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            lfp_session BYTEA NOT NULL,
            theta_freq DOUBLE PRECISION NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def create_table_tracking():
    """ create tracking (session) table """
    commands = [
        """
        CREATE TABLE tracking_tb (
            analysis_ts TIMESTAMP NOT NULL,

            session_name VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (session_name, n_drive_user, animal_id, session_ts) REFERENCES sessions_tb (session_name, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            tracking_session BYTEA NOT NULL,
            px_to_cm DOUBLE PRECISION NOT NULL,
            head_offset DOUBLE PRECISION NOT NULL,
            head_offset_var DOUBLE PRECISION NOT NULL,
            speed_cutoff DOUBLE PRECISION NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def create_table_spiketimes_tracking():
    """ create spiketimes_tracking table """
    commands = [
        """
        CREATE TABLE spiketimes_tracking_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            spiket_tracking_session BYTEA NOT NULL,
            spike_no BIGINT NOT NULL,
            mean_freq DOUBLE PRECISION NOT NULL
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_phase_tuning():
    """ create sphase_tuning table """
    commands = [
        """
        CREATE TABLE phase_tuning_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            bins_angle_center_phase BYTEA NOT NULL,
            hist_angle_smooth_phase BYTEA NOT NULL,
            phase_stats_MVL DOUBLE PRECISION NOT NULL,
            phase_stats_mean DOUBLE PRECISION NOT NULL,
            phase_stats_var DOUBLE PRECISION NOT NULL,
            rayleigh_p DOUBLE PRECISION NOT NULL,
            spike_trig_LFP BYTEA NOT NULL,
            theta_range_min INTEGER NOT NULL,
            theta_range_max INTEGER NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()



def create_table_ISI():
    """ create ISI (inter spike interval) table """
    commands = [
        """
        CREATE TABLE ISI_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            hist_ISI BYTEA NOT NULL,
            bin_edges_ISI BYTEA NOT NULL,
            ISI_stats_contam DOUBLE PRECISION NOT NULL,
            ISI_stats_contam_perc DOUBLE PRECISION NOT NULL,
            ISI_stats_percent_bursts DOUBLE PRECISION NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_st_autocorr():
    """ create spiketimes autocorr / scores table """
    commands = [
        """
        CREATE TABLE st_autocorr_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            st_autocorr BYTEA NOT NULL,
            theta_idx DOUBLE PRECISION NOT NULL,
            burst_idx1 DOUBLE PRECISION NOT NULL,
            burst_idx2 DOUBLE PRECISION NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_hd_tuning():
    """ create hd_tuning / score """
    commands = [
        """
        CREATE TABLE hd_tuning_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            bins_angle_center BYTEA NOT NULL,
            hist_angle_smooth BYTEA NOT NULL,

            tc_stats_MVL DOUBLE PRECISION NOT NULL,
            tc_stats_mean DOUBLE PRECISION NOT NULL,
            tc_stats_var DOUBLE PRECISION NOT NULL
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_ratemaps():
    """ create ratemaps table"""
    commands = [
        """
        CREATE TABLE ratemaps_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

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
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_autocorr_gs():
    """ create spatial autocorr / gridscore / ... table"""
    commands = [
        """
        CREATE TABLE autocorr_gs_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,
            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            autocorr BYTEA NOT NULL,
            autocorr_overlap DOUBLE PRECISION NOT NULL,
            grid_valid BOOLEAN NOT NULL,
            grid_score DOUBLE PRECISION NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def create_table_waveforms():
    """ create waveforms table"""
    commands = [
        """
        CREATE TABLE waveforms_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            mean_wf BYTEA NOT NULL,
            std_wf BYTEA NOT NULL,
            maxima_wf BYTEA NOT NULL
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_table_waveforms_stats():
    """ create waveforms stats table"""
    commands = [
        """
        CREATE TABLE waveforms_stats_tb (
            cluster_no SMALLINT NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            artefact INTEGER NOT NULL,
            idx_max_wf INTEGER NOT NULL,
            idx_min_wf INTEGER NOT NULL,
            swidth DOUBLE PRECISION NOT NULL
        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()




def create_table_stimulus():
    """ create laser stimulus analysis table"""
    commands = [
        """
        CREATE TABLE stimulus_tb (
            cluster_no SMALLINT NOT NULL,
            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

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

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()



def create_table_stimulus_mat():
    """ create laser stimulus spike matrix table"""
    commands = [
        """
        CREATE TABLE stimulus_mat_tb (
            cluster_no SMALLINT NOT NULL,

            analysis_ts TIMESTAMP NOT NULL,

            tetrode_no SMALLINT NOT NULL,
            session_name VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            n_drive_user VARCHAR NOT NULL,
            animal_id VARCHAR NOT NULL,
            session_ts TIMESTAMP NOT NULL,

            PRIMARY KEY (cluster_no, tetrode_no, session_name, n_drive_user, animal_id, session_ts),
            FOREIGN KEY (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) REFERENCES clusters_tb (cluster_no, tetrode_no, n_drive_user, animal_id, session_ts) ON DELETE CASCADE ON UPDATE CASCADE,

            sample_rate DOUBLE PRECISION NOT NULL,
            sample_rate_inp DOUBLE PRECISION NOT NULL,
            analysis_window DOUBLE PRECISION NOT NULL,
            counter_stimuli INTEGER NOT NULL,
            spiketimes_cluster BYTEA NOT NULL,
            stimulus_timepoints BYTEA NOT NULL,
            sum_1ms BYTEA NOT NULL,
            bin_edges_1ms BYTEA NOT NULL,
            binnumber_1ms BYTEA NOT NULL

        )
        """]

    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for command in commands:
            print(command)
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
