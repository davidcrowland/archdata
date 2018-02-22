from configparser import ConfigParser
import psycopg2
import pprint

def config(filename='database_helpers/db_params', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

def test_connect():
    """
    Connect to the PostgreSQL database server
    The password is taken from pgpass.conf under C:/Users/horsto/AppData/Roaming/postgresql

    """
    conn = None
    status = True
    try:
        # read connection parameters
        params = config()
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        # create a cursor
        cur = conn.cursor()

        # execute a statement
        #print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        #print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        #print(error)
        status = False
    finally:
        if conn is not None:
            conn.close()
            #print('Database connection closed.')

    return status
