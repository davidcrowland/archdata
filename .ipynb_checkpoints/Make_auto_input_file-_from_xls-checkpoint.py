# coding: utf-8

# # Auto input file generator - read in xls
from __future__ import print_function # Python 2.x
import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
from fnmatch import fnmatch
from villus_helpers.auto_input_general import *
from villus_helpers.run_pipeline import *
from villus_helpers.psql_start import *

# Get current path:
script_path = os.path.dirname(sys.argv[0])
script_path = "/".join(script_path.split("\\"))

print('The script path is: {}'.format(script_path))

# Take care of parameters file:
parser = argparse.ArgumentParser()
parser.add_argument("params_filename", nargs='?', default="empty")
args = parser.parse_args()


if args.params_filename == 'empty':
    print('No params_filename given.')
    print('Will load default (villus_helpers/params).')
    params_filename = script_path + "/villus_helpers/params"
    print(params_filename)
else:
    params_filename = args.params_filename

params_filename = "/".join(params_filename.split("\\"))

if not os.path.isfile(params_filename):
    print('params_filename not found!');sys.exit()
else:
    params = json.load(open(params_filename))


# ### Basics:
# load JSON params file and extract parameters
auto_input_file = str(params['auto_input_file'])
auto_input_skipped_file = str(params['auto_input_skipped_file'])

lock_file = str(params['lock_file'])

xls_file = str(params['xls_file'])
# keep KLUSTA or other (MClust, ...) .cut files?
keep_KLUSTA = bool(params['keep_KLUSTA'])
# cut-offs
cut_off_pos_file_frames = int(params['cut_off_pos_file_frames'])
cut_off_pos_file_error = float(str(params['cut_off_pos_file_error']))
# header and pattern (.cut of course):
header = str(params['header'])
pattern = str(params['pattern'])

table = str(params['table']) # postgres table (database is given in db_params!)

# ### Read in xlsx file
try:
    xls_data = pd.read_excel(xls_file, header=0, skiprows=1,sheetname="Current")
except IOError as err:
    print('Excel file not found - check path!')

# ### Take care of old input files (delete them):
if os.path.isfile(auto_input_file): os.remove(auto_input_file)
if os.path.isfile(auto_input_skipped_file): os.remove(auto_input_skipped_file)

# Write lock file (to block BNT / Matlab from executing):
with open(lock_file, "w") as lockf:
    lockf.write('LOCKED')

# # Actual pipeline:
for no in xrange(xls_data.shape[0]):
    search_dir = xls_data.iloc[no]['root folder']
    start_date = xls_data.iloc[no]['start date']
    end_date = xls_data.iloc[no]['end date']
    shape_parameter = xls_data.iloc[no]['shape parameter']
    LFP = xls_data.iloc[no]['LFP']
    overwrite = xls_data.iloc[no]['Overwrite']

    print('Current folder: {} \nStart date: {}\nEnd date: {}\nShape parameter: {}\nLFP: {}\nOverwrite: {}'.format(search_dir,
        start_date,end_date,shape_parameter,LFP,overwrite))
    print('\n\n')
    if not os.path.isdir(search_dir):
        print('Not a folder! Trying to convert to linux compatible format...\n')
        # change to linux compatible format ...
        if "N:" in search_dir:
            search_dir_tmp = "\\mnt\\N" + "".join(search_dir.split("N:"))
        elif "L:" in search_dir:
            search_dir_tmp = "\\mnt\\L" + "".join(search_dir.split("L:"))
        search_dir_tmp =  "/".join(search_dir_tmp.split('\\'))

        if not os.path.isdir(search_dir_tmp):
            print('Not a folder: {}\n'.format(search_dir_tmp))
            continue
        else:
            search_dir = search_dir_tmp

    run_pipeline(script_path,auto_input_file, auto_input_skipped_file, header, pattern, search_dir,start_date, end_date,
             cut_off_pos_file_frames, cut_off_pos_file_error, keep_KLUSTA, shape_parameter,LFP,overwrite,table)

# ... and unlock
# Write lock file (to block BNT / Matlab from executing):
with open(lock_file, "w") as lockf:
    lockf.write('UNLOCKED')
