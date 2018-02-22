# General stuff:
import sys
import os
import numpy as np
import math
import pandas as pd
from datetime import date
from tqdm import tqdm_notebook
from scipy.stats import pearsonr

from datetime import datetime
# Plotting:
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl

# Widget stuff
from ipywidgets import *
from IPython.display import display
import pandas as pd
import time
import numpy as np
from database_helpers.psql_start import *
from . import list_dicts
from .generate_query import generate_sql_query

print('Loaded postgres_analysis_helpers -> get_data')


class psql_neuroballs:
    def __init__(self, dataframe):
        self.df = dataframe # feed this in here for later ...

        self.selected_animal_ids = []
        self.selected_n_drive_users = []

        self.params = config()
        self.sql_empty_df = [] # for get_cursor function

        # retrieve clusters_tb ...
        sql = "SELECT animal_id,n_drive_user FROM clusters_tb FULL OUTER JOIN clusters_tb_bnt USING (cluster_no,tetrode_no,animal_id,n_drive_user,session_ts)"
        self.sql_db_pd = pd.read_sql_query(sql, psycopg2.connect(**self.params), index_col=None)
        self.animal_ids = self.sql_db_pd.animal_id.sort_values().unique().tolist()
        self.n_drive_users = self.sql_db_pd.n_drive_user.unique().tolist()
        sql = "SELECT animal_id FROM sessions_tb FULL OUTER JOIN sessions_tb_bnt USING (animal_id,n_drive_user,session_ts)"
        sessions_pd = pd.read_sql_query(sql, psycopg2.connect(**self.params), index_col=None)
        self.no_sessions = len(sessions_pd)

        # print statistics if input df is empty
        if len(self.df) == 0:
            print('{} animals across {} experimenters found.\nTotal # of sessions: {} (unique clusters: {})'.format(len(self.animal_ids),
            len(self.n_drive_users),self.no_sessions,len(self.sql_db_pd)))
        if len(self.df) > 0: print('Length of input dataframe: {}'.format(len(self.df)))

        self.toggle_button_lst = ['self.meta_toggle','self.sessions_toggle','self.tetrodes_toggle','self.clusters_toggle']
        self.LFP_dict = list_dicts.LFP_dict
        self.pos_dict = list_dicts.pos_dict

        self.spiketimes_dict = list_dicts.spiketimes_dict
        self.phase_tuning_dict = list_dicts.phase_tuning_dict
        self.ISI_dict = list_dicts.ISI_dict
        self.autocorr_dict = list_dicts.autocorr_dict
        self.hd_tuning_dict = list_dicts.hd_tuning_dict
        self.ratemaps_dict = list_dicts.ratemaps_dict
        self.autocorr_gs_dict = list_dicts.autocorr_gs_dict
        self.waveforms_dict = list_dicts.waveforms_dict
        self.stimulus_dict = list_dicts.stimulus_dict
        self.stimulus_mat_dict = list_dicts.stimulus_mat_dict
        #self.bnt_scores_dict = list_dicts.bnt_scores_dict
        self.bnt_dict = list_dicts.bnt_dict
        self.tracking_tb_BNT_dict = list_dicts.tracking_tb_BNT_dict
        self.spiket_tracking_tb_BNT_dict = list_dicts.spiket_tracking_tb_BNT_dict

        self.waveforms_stats_dict = list_dicts.waveforms_stats_dict

        self.selected_tb_dict = dict()
        self.list_tables = {'lfp_tb':'LFP','tracking_tb':'Tracking','spiketimes_tracking_tb':'Spiketimes + Tracking',
        'phase_tuning_tb':'Theta Phase Tuning','ISI_tb':'Interspike interval stats',
        'st_autocorr_tb':'Spiketime autocorr','hd_tuning_tb':'HD tuning stats','ratemaps_tb':'Ratemaps (KLUSTA)','autocorr_gs_tb':'Spatial autocorr','waveforms_tb':'Waveforms',
        'stimulus_tb':'Stimulus stats','stimulus_mat_tb':'Stimulus mats','BNT_tb_screen':'BNT all','tracking_tb_BNT':'BNT tracking','spiketimes_tracking_tb_bnt':'BNT spiketimes tracking', 'waveforms_stats_tb':'Waveform stats'}

        self.list_tables_lookup = {'lfp_tb':self.LFP_dict,'tracking_tb':self.pos_dict,'spiketimes_tracking_tb':self.spiketimes_dict,
        'phase_tuning_tb':self.phase_tuning_dict,'ISI_tb':self.ISI_dict,
        'st_autocorr_tb':self.autocorr_dict,'hd_tuning_tb':self.hd_tuning_dict,'ratemaps_tb':self.ratemaps_dict,'autocorr_gs_tb':self.autocorr_gs_dict,'waveforms_tb':self.waveforms_dict,
        'stimulus_tb':self.stimulus_dict,'stimulus_mat_tb':self.stimulus_mat_dict,
        'BNT_tb_screen':self.bnt_dict,'tracking_tb_BNT': self.tracking_tb_BNT_dict,'spiketimes_tracking_tb_bnt':self.spiket_tracking_tb_BNT_dict,
        'waveforms_stats_tb':self.waveforms_stats_dict}

        for tab in self.list_tables:
            self.selected_tb_dict[tab] = [] # set to false at the start (no selection)

    def retrieve_data(self,user_sql_tables=None,user_sql_animals=None,user_sql_filter=None):
        '''
        Main function for retrieval of data from the sql database
        Either call with

        user_sql_tables: defaults to None.
                         list of tables (exact postgres database table names!)
                         Will create full outer join if n>1
        user_sql_animals: defaults to None.
                          list of animal ids
        user_sql_filter: defaults to None.
                         User sql filter string (e.g. "AND session_name NOT LIKE %las%")

        or leave empty to create GUI.
        '''

        def handle_list_select_animal_ids(select):
            self.selected_animal_ids = select['new']

        def handle_list_select_n_drive_users(select):
            self.selected_n_drive_users = select['new']
            self.list_animals.options = np.sort(self.sql_db_pd.animal_id[self.sql_db_pd.n_drive_user.isin(self.selected_n_drive_users)].unique()).tolist()

        def handle_list_select_tables(select):

            if len(select['new']) > 1:
                print('Please only select one table at a time.')
            else:
                self.selected_table = select['new']
                #print(self.selected_table)
                #print(self.list_tables_lookup[str(self.selected_table).replace("(","").replace(")","").rstrip(",")[1:-1]])
                self.entry_selection_list.options = self.list_tables_lookup[str(self.selected_table).replace("(","").replace(")","").rstrip(",")[1:-1]]
                # ... and set previously selected items
                self.entry_selection_list.value = self.selected_tb_dict[str(self.selected_table).replace("(","").replace(")","").rstrip(",")[1:-1]]
                #print(self.entry_selection_list.value)

        def handle_table_entry_selection(select):
            if (len(select['new']) > 0) and ("Select a table" in select['new'][0]):
                print('First select a table...')
            elif len(select['new']) > 0:
                self.selected_entry = select['new']
                #print(self.selected_table)
                #print('Selected {} from {}'.format(self.selected_entry,self.selected_table))
                self.selected_tb_dict[str(self.selected_table).replace("(","").replace(")","").rstrip(",")[1:-1]] = self.selected_entry
                #print('New Selection:')
                #print(str(self.selected_table).replace("(","").replace(")","").rstrip(",")[1:-1])
                #print(self.selected_tb_dict) #!!!!

        def retrieve_concat(base_dataframe,selection,table,filter_,primary_keys):
            new_df = pd.DataFrame()
            count_nans = 0

            for i in tqdm_notebook(range(len(base_dataframe)),desc=table):
                c_entry = base_dataframe.iloc[i] # gets series obj
                c_entry_T = pd.DataFrame(c_entry).T
                c_entry_T.reset_index(inplace=True,drop=True)
                # sometimes session_name can jump to NaN (if it is part of the first retrieval session)
                if 'session_name' in c_entry_T.columns:
                    if not isinstance(c_entry_T.session_name.values[0], str):
                        #print(c_entry_T.session_name.values[0])
                        c_entry_T.drop('session_name',1,inplace=True)

                # generate sql query string
                sql = generate_sql_query(selection, table, c_entry_T, filter_,primary_keys)
                # query database ...
                c_sql = pd.read_sql_query(sql, psycopg2.connect(**self.params), index_col=None)

                if len(c_sql) == 0:
                    count_nans += 1
                    c_join = pd.concat([c_entry_T, c_sql],axis=1) # horizontally join the two entries
                    new_df = pd.concat([new_df,  c_join]) # vertically join it with the rest
                for c in range(len(c_sql)):
                    c_sql_reset = pd.DataFrame(c_sql.iloc[c]).T
                    c_join = pd.concat([c_entry_T, c_sql_reset.reset_index(drop=True)],axis=1) # horizontally join the two entries
                    new_df = pd.concat([new_df,  c_join]) # vertically join it with the rest

            new_df.reset_index(inplace=True,drop=True)
            return new_df,count_nans

        def handle_display_select(select):
            print(self.selected_tb_dict)

        def get_primary_keys(table):
            sql = "SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS data_type FROM pg_index i JOIN pg_attribute a ON \
            a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) WHERE i.indrelid = '{}'::regclass \
            AND    i.indisprimary;".format(table)
            c_sql = pd.read_sql_query(sql, psycopg2.connect(**self.params), index_col=None)
            return c_sql.attname.values.tolist()

        def order_tabs_sql(tables_to_mine):
            '''
            To retrieve data from database I have to order the
            elements (tables) I want to retrieve from in order of their
            primary keys such that the tables with lowest number of primary keys
            come last.
            '''
            lengths = []
            all_primary_keys = []
            all_entries = []

            for tab in tables_to_mine:
                primary_keys = get_primary_keys(tab)
                all_primary_keys.append(primary_keys)
                lengths.append(len(primary_keys))
                if len(tables_to_mine[tab]) == 0:
                    sql_colum_names_str = "SELECT * FROM {} WHERE false;".format(tab)
                    sql_column_names = pd.read_sql_query(sql_colum_names_str, psycopg2.connect(**self.params), index_col=None)
                    for column_name in sql_column_names.columns:
                        if column_name not in primary_keys:
                            all_entries.append([column_name])
                else:
                    all_entries.append(tables_to_mine[tab])
            unique_primaries = set([item for sublist in all_primary_keys for item in sublist])
            all_entries = [item for sublist in all_entries for item in sublist]
            # remove duplicates from query:
            # see https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
            seen = set()
            seen_add = seen.add
            all_entries =  [x for x in all_entries if not (x in seen or seen_add(x))]

            sorted_dict = dict()
            for no,idx in enumerate(np.argsort(np.array(lengths))[::-1]):
                sorted_dict[no] = [k for k,v in tables_to_mine.items()][idx]
            return sorted_dict,list(unique_primaries),all_entries

        def generate_table_sql_str(sorted_dict):
            table_str = " "
            if len(sorted_dict) > 1:
                for no in range(len(sorted_dict)-1):
                    #print(sorted_dict[no])
                    if no == 0:
                        table_str += " " + sorted_dict[no]
                    outer_join_using = str(list(set(get_primary_keys(sorted_dict[no])).intersection(get_primary_keys(sorted_dict[no+1])))).replace("[","").replace("]","").replace("'","").rstrip(',')
                    table_str += " FULL OUTER JOIN " + sorted_dict[no+1] + " USING ({})".format(outer_join_using)
            else:
                table_str +=  sorted_dict[0]
            return table_str

        def create_empty_df_sql():
            '''
            Function to call when GUI is created and user has fed in an empty dataframe.
            Performs full outer join on all tables (and selected entries) that user selected.

            Returns appropiate sql query string.
            '''
            tables_to_mine = {k:v for k,v in self.selected_tb_dict.items() if len(v)>0}
            sorted_dict,unique_primaries,all_entries = order_tabs_sql(tables_to_mine)
            table_str = generate_table_sql_str(sorted_dict)
            filter_  = self.session_name_box.value
            if len(filter_) > 0:
                filter_ = " AND session_name {}".format(filter_)
            else:
                filter_ = ""
            self.sql_empty_df = "SELECT {}, {} FROM {} WHERE animal_id = ANY('{}') {}".format(str(unique_primaries).replace("[","").replace("]","").replace("'","").rstrip(','),
            str(all_entries).replace("[","").replace("]","").replace("'","").rstrip(','),table_str,str([x.strip('"\'') for x in self.selected_animal_ids]).replace('[','{').replace(']','}').replace('\'',''),filter_)
            #print(self.sql_empty_df)
            return self.sql_empty_df

        def create_user_sql_nodf(user_sql_tables,user_sql_animals,user_sql_filter):
            '''
            If user puts in a "user_sql" dict.
            Create appropiate SQL query and retrieve data.

            Returns queried dataframe.

            '''
            sorted_dict,unique_primaries,all_entries = order_tabs_sql(user_sql_tables)
            table_str = generate_table_sql_str(sorted_dict)
            if user_sql_filter:
                filter_ = "{}".format(user_sql_filter)
            else:
                filter_ = ""

            self.sql_empty_df = "SELECT {}, {} FROM {} WHERE animal_id = ANY('{}') {}".format(str(unique_primaries).replace("[","").replace("]","").replace("'","").rstrip(','),
            str(all_entries).replace("[","").replace("]","").replace("'","").rstrip(','),table_str,str([x.strip('"\'') for x in user_sql_animals]).replace('[','{').replace(']','}').replace('\'',''),filter_)
            #print(self.sql_empty_df)
            # Retrieve dataframe
            self.df  = pd.read_sql_query(self.sql_empty_df, psycopg2.connect(**self.params), index_col=None,parse_dates=['session_ts'])
            print('{} entries retrieved.'.format(len(self.df)))

            return self.df

        def create_user_sql_df(user_sql_tables,user_sql_filter):
            '''
            If user puts in a "user_sql" AND input dataframe is not empty.
            Create appropiate SQL query and retrieve data.

            Returns queried dataframe.

            '''
            print('Dataframe not empty')
            tables_to_mine = user_sql_tables # omit the len(v) > 0 statement here
            if user_sql_filter:
                filter_ = "{}".format(user_sql_filter)
            else:
                filter_ = ""

            for tb in tqdm_notebook(tables_to_mine,desc='Mining tables'):
                primary_keys = get_primary_keys(tb)
                # check if that columns already exists or not:
                #print(self.selected_tb_dict[tb][0])
                #print(self.df.columns.values)
                if len(tables_to_mine[tb]) == 0:
                    print('Table dictionary is empty for this entry. Taking all columns!')
                    sql_colum_names_str = "SELECT * FROM {} WHERE false;".format(tb)
                    sql_column_names = pd.read_sql_query(sql_colum_names_str, psycopg2.connect(**self.params), index_col=None)
                    all_columns = []
                    for column_name in sql_column_names.columns:
                        if column_name not in primary_keys:
                            all_columns.append(column_name)
                    print('Included the following columns: {}'.format(tuple(all_columns)))
                    tables_to_mine[tb] = tuple(all_columns)

                # now check for every column name if it already exists in base dataframe
                all_columns = []
                if isinstance(tables_to_mine[tb], tuple) == True:
                    for column in tables_to_mine[tb]:
                        if column in self.df.columns.values:
                            print('Column {} already exists. Skipping.'.format(column))
                            continue
                        all_columns.append(column)
                else:
                    if tables_to_mine[tb] in self.df.columns.values:
                        print('Column {} already exists. Skipping.'.format(tables_to_mine[tb]))
                        continue
                    all_columns.append(tables_to_mine[tb])

                if len(all_columns) > 0:
                    tables_to_mine[tb] = tuple(all_columns)
                    self.df,count_nans = retrieve_concat(self.df,tables_to_mine[tb],tb,filter_,primary_keys)
                    print('Success. NaNs: {}'.format(count_nans))
                else:
                    print('Did not start retrieval because sql query is empty.')

            return self.df


        def prepare_cursor(select):
            if not self.sql_empty_df and len(self.df) == 0:
                print('No selection was made! Creating sql query string.')
                self.sql_empty_df = create_empty_df_sql()
                #print(self.sql_empty_df)
            if len(self.df) == 0:
                self.cur_conn,self.cur_cur,self.cur_status,self.cur_column_names = get_cursor(self.sql_empty_df,self.params)

            # yeah, yeah I know these variables names are confusing

        def get_cursor(sql,params):
            '''
            Instead of a pandas query, get a cursor object
            that one can iterate over.
            This enables the analysis of large datasets.

            '''
            conn = None
            status = True
            sql_columns = [name.strip() for name in sql.split('SELECT')[1].split('FROM')[0].strip().split(",")]
            #print(sql_columns)
            try:
                # read connection parameters
                params = config()
                # connect to the PostgreSQL server
                #print('Connecting to the PostgreSQL database...')
                conn = psycopg2.connect(**params)
                # create a cursor
                cur = conn.cursor('sql_1') # random name for now!
                # execute a statement
                cur.itersize = 1 # how much records to buffer on a client
                cur.execute(sql)
                return conn,cur,status,sql_columns

            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                status = False
                conn.close()
                cur.close()
                return np.nan,np.nan,status,np.nan

        def handle_retrieval(select):
            '''
            Actual dataset retrieval happens here...

            '''
            sys.stdout.write('Retrieving data ... ')
            self.start_time = datetime.now()

            #print(self.selected_animal_ids)
            #print(self.selected_n_drive_users)
            if len(self.df) == 0:
                print('(Empty dataframe) ...')

                self.sql_empty_df = create_empty_df_sql()
                # retrieve dataframe
                sql_db_pd = pd.read_sql_query(self.sql_empty_df, psycopg2.connect(**self.params), index_col=None,parse_dates=['session_ts'])
                self.df = sql_db_pd
                print('{} entries retrieved.'.format(len(self.df)))

            else: # pre-existing dataframe ...
                print('Dataframe not empty')
                tables_to_mine = {k:v for k,v in self.selected_tb_dict.items() if len(v)>0}
                filter_  = self.session_name_box.value
                #print('from {}'.format([k for k,v in self.selected_tb_dict.items() if len(v)>0]))

                for tb in tqdm_notebook(tables_to_mine,desc='Mining tables'):
                    primary_keys = get_primary_keys(tb)
                    # check if that columns already exists or not:
                    #print(self.selected_tb_dict[tb][0])
                    #print(self.df.columns.values)
                    print(self.selected_tb_dict[tb][0].lower())
                    if self.selected_tb_dict[tb][0].lower() in self.df.columns.values:
                        print('Column {} already exists. Skipping.'.format(self.selected_tb_dict[tb][0]))
                        continue
                    self.df,count_nans = retrieve_concat(self.df,self.selected_tb_dict[tb],tb,filter_,primary_keys)
                    print('Success. NaNs: {}'.format(count_nans))

        #####################################################################################################################
        #####################################################################################################################
        # Check some basic stuff:

        if len(self.df)>0 and (user_sql_animals != None):
            print('Input dataframe not empty. Either delete input dataframe or animal list.')
        elif len(self.df)==0 and (user_sql_tables != None) and (user_sql_animals != None):
            self.df = create_user_sql_nodf(user_sql_tables,user_sql_animals,user_sql_filter)
            return self.df
        elif len(self.df)>0 and (user_sql_tables != None):
            self.df = create_user_sql_df(user_sql_tables,user_sql_filter)
            return self.df

        else:
            # Generate complete layout:
            box_layout = Layout(display='flex',
                                flex_flow='columns',
                                align_items='stretch',
                                border='',
                                width='100%',height='200px')

            # Lists:
            list_layout = Layout(display='flex',flex_flow='row',align_items='stretch', width='100px',height='150px')
            list_layout_wider = Layout(display='flex',flex_flow='row',align_items='stretch', width='200px',height='150px')
            list_layout_evenwider = Layout(display='flex',flex_flow='row',align_items='stretch', width='240px',height='150px')

            self.list_animals = SelectMultiple(
                        options=self.animal_ids,
                        description='',disabled=False,layout=list_layout)
            self.list_n_drive_users = SelectMultiple(
                        options=self.n_drive_users,
                        description='',disabled=False,layout=list_layout)
            self.list_tables_list = SelectMultiple(
                        options={v:k for k,v in self.list_tables.items()},
                        description='',disabled=False,layout=list_layout_wider)
            self.entry_selection_list = SelectMultiple(options=["Select a table to the left"],
                        description='',disabled=False,layout=list_layout_evenwider)


            # Text fields:
            self.session_name_box = Textarea(value="NOT LIKE '%las%'",
            placeholder='Type filter for session_name', description='', disabled=False,layout=Layout(display='flex',flex_flow='column',align_items='stretch',height='150px'))

            # Buttons:
            display_select_button = Button(description='Print selection',disabled=False,
                button_style='',
                tooltip='Click here to show which entries you marked for retrieval',
                icon='',layout=Layout(display='flex',flex_flow='column',align_items='stretch', width='10%',height='15%')
            )
            get_cursor_button = Button(description='Get cursor',disabled=False,
                button_style='',
                tooltip='Click here to get a cursor instead of a dataset',
                icon='',layout=Layout(display='flex',flex_flow='column',align_items='stretch', width='10%',height='15%')
            )

            retrieval_button = Button(description='Get ze data!',disabled=False,
                button_style='info',
                tooltip='Click here to retrieve data from the database',
                icon='',layout=Layout(display='flex',flex_flow='column',align_items='stretch', width='10%',height='15%')
            )



            self.n_drive_users_animals_lists = [self.list_n_drive_users, self.list_animals,self.list_tables_list,
            self.entry_selection_list,self.session_name_box]
            self.without_n_drive_users_animals_lists = [self.list_tables_list,
            self.entry_selection_list,self.session_name_box]
            # Buttons:
            self.buttons_bottom = [display_select_button,get_cursor_button, retrieval_button]

            # GENERATE LAYOUT (GENERIC)
            if len(self.df) == 0:
                self.box = VBox([HBox(children=self.n_drive_users_animals_lists), HBox(children=self.buttons_bottom)],layout=box_layout)
                display(self.box)
            else:
                self.box = VBox([HBox(children=self.without_n_drive_users_animals_lists), HBox(children=self.buttons_bottom)],layout=box_layout)
                display(self.box)


            # functions (from "left" to "right"):
            self.list_animals.observe(handle_list_select_animal_ids, names='value')
            self.list_n_drive_users.observe(handle_list_select_n_drive_users, names='value')
            self.list_tables_list.observe(handle_list_select_tables,names='value')
            self.entry_selection_list.observe(handle_table_entry_selection,names='value')

            # If retrieve data button is pressed ...
            display_select_button.on_click(handle_display_select)
            get_cursor_button.on_click(prepare_cursor)
            retrieval_button.on_click(handle_retrieval)

    def cursor(self):
        if not self.cur_conn:
            print('DAMN!')
        else:
            return self.cur_conn,self.cur_cur,self.cur_status,self.cur_column_names

    def data(self):
        return self.df

    def time(self):
        return (self.end_time - self.start_time).total_seconds()
