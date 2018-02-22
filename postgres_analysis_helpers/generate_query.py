import pandas as pd
def generate_sql_query(selection, table, c_entry_T, filter_,primary_keys):
    '''
    Feed this a "selection" (SELECT)
    "table" (FROM), retrieval cue (derived from c_entry_T, WHERE)
    and additional filter (filter_, WHERE),
    then query database with that
    '''

    for column in c_entry_T.columns:
        if column == 'cluster_no':
            c_entry_T.cluster_no = pd.to_numeric(c_entry_T.cluster_no, downcast='integer')
        if column == 'tetrode_no':
            c_entry_T.tetrode_no = pd.to_numeric(c_entry_T.tetrode_no, downcast='integer')

    where_str = ""
    # generate search string on primary_keys:
    for no,key in enumerate(primary_keys):
        if key in c_entry_T.columns:
            temp_str = " {} = '{}' {}".format(key,c_entry_T[key].values[0],['AND' if no < len(primary_keys)-1 else ''][0])
            where_str += temp_str

    if len(filter_) > 0:
        filter_ = "AND session_name {}".format(filter_)
    else:
        filter_ = ""

    # supplement main primary keys if missing:
    missing_columns = []
    if 'session_name' not in c_entry_T.columns:
        missing_columns.append('session_name')
    if 'cluster_no' not in c_entry_T.columns:
        missing_columns.append('cluster_no')
    if 'tetrode_no' not in c_entry_T.columns:
        missing_columns.append('tetrode_no')
    if 'session_ts' not in c_entry_T.columns:
        missing_columns.append('session_ts')

    if len(missing_columns) > 0:
        missing_columns = ",".join(missing_columns)
        #print('missing_columns: {}'.format(missing_columns.replace("'","").rstrip(',')))
        sql = "SELECT {}, {} FROM {} WHERE {} {}".format(str(selection).replace("(","").replace(")","").replace("'","").rstrip(','),
        missing_columns.replace("'","").rstrip(','),table,where_str,filter_)
    else:
        sql = "SELECT {} FROM {} WHERE {} {}".format(str(selection).replace("(","").replace(")","").replace("'","").rstrip(','),
        table,where_str,filter_)

    #print(sql)
    return sql
