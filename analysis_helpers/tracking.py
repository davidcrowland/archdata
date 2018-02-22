# functions that work with the tracking data ....
import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import circmean,circvar

import cv2
import sys

def pos_make_df(data_pos,box_size_cm,timebase_pos,time_stamps_sessions_pos, loop_n, divider_n):
    data_pos_df = pd.DataFrame(data_pos)
    data_pos_df['time']=np.array(data_pos_df['frame_counter'],dtype=float)/float(timebase_pos) # in sec
    data_pos_df.set_index('time', drop=True, append=False, inplace=True, verify_integrity=False)

    # find amount of invalid tracking
    x1_fail = np.sum(data_pos_df.x1.values == 1023)/float(len(data_pos_df))
    x2_fail = np.sum(data_pos_df.x2.values == 1023)/float(len(data_pos_df))
    y1_fail = np.sum(data_pos_df.y1.values == 1023)/float(len(data_pos_df))
    y2_fail = np.sum(data_pos_df.y2.values == 1023)/float(len(data_pos_df))

    # get rid of 1023 values ...
    data_pos_df['x1'].replace(to_replace=1023, inplace=True, method='ffill', axis=None) # ffill first
    data_pos_df['x1'].replace(to_replace=1023, inplace=True, method='bfill', axis=None) # then do bfill to get rid of 1023s at the end

    data_pos_df['x2'].replace(to_replace=1023, inplace=True, method='ffill', axis=None)
    data_pos_df['x2'].replace(to_replace=1023, inplace=True, method='bfill', axis=None)

    data_pos_df['y1'].replace(to_replace=1023, inplace=True, method='ffill', axis=None)
    data_pos_df['y1'].replace(to_replace=1023, inplace=True, method='bfill', axis=None)

    data_pos_df['y2'].replace(to_replace=1023, inplace=True, method='ffill', axis=None)
    data_pos_df['y2'].replace(to_replace=1023, inplace=True, method='bfill', axis=None)

    # get ratio (px to cm) ...
    # do the following calculations only on first session (base session)
    idx_start=int(time_stamps_sessions_pos[0]); idx_stop=int(time_stamps_sessions_pos[1]) # take first session (base session)
    if np.diff(data_pos_df['frame_counter'].values[idx_stop-int(timebase_pos):idx_stop]).sum() == 0:
        #sys.stdout.write('Shortening position data for {} frames (nonsense)'.format(timebase_pos))
        idx_stop -= int(timebase_pos)
    first_session = data_pos_df.iloc[idx_start:idx_stop,:]

    deltax1 = np.max(first_session['x1'])-np.min(first_session['x1'])
    deltay1 = np.max(first_session['y1'])-np.min(first_session['y1'])
    deltax2 = np.max(first_session['x2'])-np.min(first_session['x2'])
    deltay2 = np.max(first_session['y2'])-np.min(first_session['y2'])
    px_to_cm = box_size_cm/np.mean([deltax1,deltay1,deltax2,deltay2]) # assuming square arena
    #print('1 px = {} cm (assuming {} cm square box)'.format(px_to_cm,box_size_cm))

    # find correct LED ...
    x_art_all = np.zeros((loop_n,divider_n))
    y_art_all = np.zeros((loop_n,divider_n))
    # between the two LEDs try to find the center point as the point of minimum movement
    for i in xrange(loop_n): # first loop_n position samples
        counter_divider = 0
        for divider in np.linspace(-1.5,1.5,divider_n):
            art_point_x = divider*abs((first_session['x2'].values[i]-first_session['x1'].values[i]))
            art_point_y = divider*abs((first_session['y2'].values[i]-first_session['y1'].values[i]))

            if first_session['x1'].values[i] <= first_session['x2'].values[i]:
                x_art = first_session['x1'].values[i]+art_point_x
            if first_session['x1'].values[i] > first_session['x2'].values[i]:
                x_art = first_session['x1'].values[i]-art_point_x
            if first_session['y1'].values[i] <= first_session['y2'].values[i]:
                y_art = first_session['y1'].values[i]+art_point_y
            if first_session['y1'].values[i] > first_session['y2'].values[i]:
                y_art = first_session['y1'].values[i]-art_point_y
            x_art_all[i,counter_divider]  = x_art
            y_art_all[i,counter_divider]  = y_art

            counter_divider = counter_divider +1

    dist_art_all = np.zeros((loop_n-1,divider_n))
    for divider in xrange(divider_n):
        dist_art_all[:,divider] = np.sqrt(np.square(np.diff(x_art_all[:,divider]))+np.square(np.diff(y_art_all[:,divider])))

    total_dist_art = np.cumsum(dist_art_all,axis=0)[-1,:]
    fraction = np.linspace(-1.5,1.5,divider_n)[np.argmin(total_dist_art)]

    if (fraction > 0.5):
        if (x1_fail < 0.3) and (y1_fail < 0.3):
            data_pos_df['correct_x'] = data_pos_df['x1']
            data_pos_df['correct_y'] = data_pos_df['y1']
        else:
            data_pos_df['correct_x'] = data_pos_df['x2']
            data_pos_df['correct_y'] = data_pos_df['y2']
    else:
        if (x2_fail < 0.3) and (y2_fail < 0.3):
            data_pos_df['correct_x'] = data_pos_df['x2']
            data_pos_df['correct_y'] = data_pos_df['y2']
        else:
            data_pos_df['correct_x'] = data_pos_df['x1']
            data_pos_df['correct_y'] = data_pos_df['y1']

    # smooth positions ...
    cols = ['x1','x2','y1','y2','correct_x','correct_y']
    for col in cols:
        #data_pos_df[col+'_inter'] = savgol_filter(data_pos_df[col], 25, 4) # Savitzky golay
        data_pos_df[col+'_inter'] = gaussian_filter1d(data_pos_df[col], 2, mode='nearest') # smoothed position with sigma = 2

    # Get speed ...
    dist = np.sqrt(np.square(np.diff(data_pos_df['correct_x_inter']))+np.square(np.diff(data_pos_df['correct_y_inter'])))
    time_diff = np.diff(data_pos_df.index)
    time_diff[time_diff == 0] = np.inf
    speed = np.hstack((0,dist*px_to_cm/time_diff)) # cm/s
    speed_filtered = gaussian_filter1d(speed, 1) # smoothed speed with sigma = 1
    data_pos_df['speed'] = speed
    data_pos_df['speed_filtered'] = speed_filtered

    #######################################################################################################################
    # correction of arena and head direction offset

    # correct rotation of arena if it is not perfectly positioned at 90 degree to camera
    # renew first_session data (do calculations only on base sesssion)
    first_session = data_pos_df.iloc[idx_start:idx_stop,:]

    center_x = int((np.max(first_session['correct_x_inter']) - np.min(first_session['correct_x_inter'])))
    center_y = int((np.max(first_session['correct_y_inter']) - np.min(first_session['correct_y_inter'])))
    center = (center_x,center_y)

    first_session_coords = np.array(np.column_stack((first_session['correct_x_inter'],first_session['correct_y_inter'])),dtype=int)
    angle = cv2.minAreaRect(first_session_coords)[-1]
    if np.abs(angle) > 45:
        angle = 90 + angle
    sys.stdout.write('Detected a arena rotation angle of {:.2f} degree.\n'.format(angle))
    M = cv2.getRotationMatrix2D(center,angle,1)
    # rotation matrix is applied in the form:
    #M00x + M01y + M02
    #M10x + M11y + M12

    keys_to_correct = [['x1','y1'],['x2','y2'],['x1_inter','y1_inter'],['x2_inter','y2_inter'],
    ['correct_x','correct_y'],['correct_x_inter','correct_y_inter']]

    for pair in keys_to_correct:
        correct_xs, correct_ys = apply_rotation(data_pos_df,pair[0],pair[1],M)
        #sys.stdout.write('Corrected {} and {}.\n'.format(pair[0],pair[1]))
        # write corrected coordinates to dataframe
        data_pos_df[pair[0]] = correct_xs
        data_pos_df[pair[1]] = correct_ys

    # Correct head direction / LED offset:
    # Get LED direction ...
    diff_x_led = data_pos_df['x2_inter']-data_pos_df['x1_inter']
    diff_y_led = data_pos_df['y2_inter']-data_pos_df['y1_inter']
    led_angle = np.array([math.atan2(list(x)[0],list(x)[1]) for x in zip(diff_x_led,diff_y_led)])
    led_angle = (led_angle + 2*np.pi) % (2*np.pi)
    data_pos_df['led_angle'] = led_angle

    # Get moving direction ...
    diff_x_move = np.diff(data_pos_df['correct_x_inter'])
    diff_y_move = np.diff(data_pos_df['correct_y_inter'])
    mov_angle = np.array([math.atan2(list(x)[0],list(x)[1]) for x in zip(diff_x_move,diff_y_move)])
    mov_angle = np.hstack((mov_angle,0))
    mov_angle = (mov_angle + 2*np.pi) % (2*np.pi)
    data_pos_df['mov_angle'] = mov_angle

    # Calculate head direction / LED offset
    # ... renew first_session df:
    # to calculate only over first session
    first_session = data_pos_df.iloc[idx_start:idx_stop,:]
    mov_angle_first = first_session['mov_angle'][first_session['speed']>20].values # filter at 20 cm/s speed (that's quite random)
    led_angle_first = first_session['led_angle'][first_session['speed']>20].values
    diff_mov_led = mov_angle_first - led_angle_first
    diff_mov_led[diff_mov_led<0] = 2*np.pi+diff_mov_led[diff_mov_led<0]
    diff_mov_led[diff_mov_led>2*np.pi] = diff_mov_led[diff_mov_led>2*np.pi] - 2*np.pi

    head_offset = circmean(diff_mov_led)
    head_offset_var = circvar(diff_mov_led)
    sys.stdout.write('Head angle offset: {:.2f} degrees | Variance: {:.2f}\n'.format(math.degrees(head_offset),head_offset_var))
    if head_offset_var > 1:
        sys.stdout.write('Head angle offset variance > 1: This is not accurate.\n')

    # ... and correct LED angle:
    #led_angle_corr = [led_angle - head_offset if head_offset < 0 else led_angle + head_offset][0]
    led_angle_corr = led_angle + head_offset
    led_angle_corr[led_angle_corr<0] = 2*np.pi+led_angle_corr[led_angle_corr<0]
    led_angle_corr[led_angle_corr>2*np.pi] = led_angle_corr[led_angle_corr>2*np.pi] - 2*np.pi

    data_pos_df['head_angle'] = led_angle_corr

    # there is a problem here - pandas has problems reading this stuff because it has a
    # little endian compiler issue when adding the angle vector to the DataFrame.
    # Values can still be read though.
    return data_pos_df,px_to_cm,head_offset,head_offset_var

def apply_rotation(data_pos_df,xs,ys,M):
    coords = np.array(np.column_stack((data_pos_df[xs],data_pos_df[ys])),dtype=float)
    coords_rot = [[coord[0]*M[0,0]+coord[1]*M[0,1]+M[0,2],coord[0]*M[1,0]+coord[1]*M[1,1]+M[1,2]] for coord in coords]
    correct_xs = [element[0] for element in coords_rot]
    correct_ys = [element[1] for element in coords_rot]
    return correct_xs,correct_ys

print('Loaded analysis helpers: Tracking')
