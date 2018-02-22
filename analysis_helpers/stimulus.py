# Laser (input) stimulus analysis stuff

import numpy as np
from tqdm import tqdm
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.stats import entropy,mannwhitneyu

def stimulus_parameters(snippet,sample_rate_inp):
    X = np.array(zip(np.sort(np.diff(snippet)),np.zeros(len(np.sort(np.diff(snippet))))), dtype=np.int)

    stim_params = {}

    try:
        bandwidth = estimate_bandwidth(X, quantile=0.05)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        clusters = np.sort([cluster_centers[no][0] for no,label in enumerate(labels_unique) if len(labels[labels == label]) > 40])
        if clusters[0] < 0.1: clusters = clusters[1:] # filter out if stimulus length < 0.1 ms
    except ValueError as error:
        print('Stimulus parameters could not be evaluated. Skipping.')
        stim_params['stim_length'] = np.nan; stim_params['stim_freq'] = np.nan; stim_params['IblockI'] = np.nan
        return stim_params
    except IndexError as error:
        print('Stimulus parameters could not be evaluated. Skipping.')
        stim_params['stim_length'] = np.nan; stim_params['stim_freq'] = np.nan; stim_params['IblockI'] = np.nan
        return stim_params      

    if len(clusters) > 1:
        stim_length = clusters[0]
        stim_freq = 1000./clusters[:2].sum() # if there are more clusters detected than 3 this will go to bitches, but I'm too stupid
        IblockI = clusters[-1]
        print('Stim length: {:.2f} ms | Stim freq: {:.2f} Hz | IBlockI: {:.2f} ms'.format(stim_length,stim_freq,IblockI))
    else:
        print('No stimulus parameters identified.')
        stim_length=np.nan;stim_freq=np.nan;IblockI=np.nan
    stim_params['stim_length'] = stim_length; stim_params['stim_freq'] = stim_freq; stim_params['IblockI'] = IblockI
    return stim_params

def inhibition_stats(sum_1ms):
    inhibited = False
    lowest_p = 10 # start with 10 to be conservative
    lowest_p_inter = 0

    intervals = [10,20,30,40]
    stats_inhib = {}
    for inter in intervals:
        before = sum_1ms[len(sum_1ms)/2-inter:len(sum_1ms)/2]
        after = sum_1ms[len(sum_1ms)/2:len(sum_1ms)/2+inter]
        if (before==after).all() or before.sum()==0 or after.sum()==0:
            U,p=np.nan,np.nan
            stats_inhib[inter] = p
            lowest_p,lowest_p_inter = np.nan,np.nan
        else:
            U,p = mannwhitneyu(before,after,alternative='two-sided')
            if p < lowest_p:
                lowest_p = p
                lowest_p_inter = inter # save these values to display them in graph
            if p < 0.05:
                # check whether the ratio after/before is actually < 1 !
                if np.sum(after)/np.sum(before) < 1:
                    inhibited = True
            stats_inhib[inter] = p

    return stats_inhib,inhibited,lowest_p,lowest_p_inter

def change_point_inhib(spike_mat,sum_1ms,sample_rate,window,plotting=False):
    '''
    change point analysis:
    Change points are graphically represented by a change in the slope of a plot showing the cumulative
    sums of the firing frequency averaged over the 60 light trials (Extended Data Fig. 3e)
    http://dx.doi.org/10.1038/nature13258
    '''

    tc = np.zeros(spike_mat.shape[1])
    cum_act = 0
    cum_act_old = 0
    # state True = positive change
    state_old = True

    streak = []
    sum_streaks = []
    idxs_sum_streaks = []

    for tb in xrange(spike_mat.shape[1]):
        slice_ = np.sum(spike_mat[:,tb])
        # what does it need to keep it stable? this is only for visualization
        add = 1/(sum_1ms.sum()/len(sum_1ms)/(sample_rate/1000))

        cum_act = [cum_act + add if slice_ > 0 else cum_act-1][0]
        tc[tb] = cum_act

        if cum_act-cum_act_old >= 0:
            state = True
            streak.append(cum_act-cum_act_old)
        else:
            state = False
            streak.append(cum_act-cum_act_old)

        if state != state_old:
            sum_streaks.append(np.sum(streak))
            idxs_sum_streaks.append(tb)
            streak = []
            streak.append(cum_act-cum_act_old)
            # direction changed!

        state_old = state
        cum_act_old = cum_act

    sum_streaks = np.array(sum_streaks)
    idxs_sum_streaks = np.array(idxs_sum_streaks)

    std_neg = np.std(sum_streaks[sum_streaks<0])
    # get streaks that are < 3 SD of negative streaks
    idxs_sig = idxs_sum_streaks[(sum_streaks<-3*std_neg)]
    if len(idxs_sig) == 0:
        change_point_idx = np.nan
        change_point_ms = np.nan
    else:
        change_point_idx = (idxs_sig[np.argmax(idxs_sig>spike_mat.shape[1]/2)]) # filters for the first one after middle of recording

        # look for this value in all indices to go one back:
        change_point = idxs_sum_streaks[(np.argmax(idxs_sum_streaks==change_point_idx))-1]
        change_point_ms = (change_point / (sample_rate/1000)) - window
        print('Lat. inhibition: {} ms'.format(change_point_ms))

        if plotting:
            plt.plot(idxs_sum_streaks,sum_streaks,'k')
            plt.axvline(x=change_point_idx,color='r')
            plt.axvline(x=len(tc)/2)

            plt.xlim(len(tc)/2-(sample_rate/1000)*10,len(tc)/2+(sample_rate/1000)*10)
            plt.show()
    return change_point_ms

def get_cor_stimuli(input_data,snippet,stim_params):
    skipped = 0
    not_I = 0
    counter_stimuli = 0

    stimuli = []
    indxs = []

    ts_prev = snippet[0]

    for no,ts in enumerate(snippet):
        if input_data['IOK'][no] != 'I':
            not_I += 1
            continue

        if (ts-ts_prev > 2*stim_params['stim_length']): # 1.2* is a safety margin, only one sample / stim
            counter_stimuli += 1
            stimuli.append(ts)
            indxs.append(no)
        else:
            skipped += 1
        ts_prev = ts

    if np.mean(np.diff(indxs)) != 2:
        print('Some input timestamps were skipped - check that! (Mean: {:.2f})'.format(np.mean(np.diff(indxs))))
    if (not_I+counter_stimuli+skipped) != len(snippet): print('Length of sessions and snippet length don''t match!');sys.exit()

    return counter_stimuli,stimuli

###########################################################################################
# SALT stuff:

def SALT(spike_mat,sample_rate,win_SALT):
    '''
    SALT test function for excitation
    Adapted from matlab code:
    http://dx.doi.org/10.1038/nature12176
    '''

    # base params:
    sr_SALT = sample_rate/1000. # samples per ms
    win_lat_hist = int(win_SALT*sr_SALT)
    num_tr = spike_mat.shape[0]

    bins_baseline = np.linspace(0,spike_mat.shape[1]/2, int((spike_mat.shape[1]/2)/win_lat_hist)+1)
    bin_test = [spike_mat.shape[1]/2,spike_mat.shape[1]/2+win_lat_hist]

    all_lat = np.zeros((num_tr,len(bins_baseline)))   # preallocate latency matrix
    sort_lat = np.zeros((num_tr,len(bins_baseline)))   # preallocate sorted latency matrix
    sort_lat_hist = np.zeros((win_lat_hist-1,len(bins_baseline)))
    nsort_lat_hist = np.zeros((win_lat_hist-1,len(bins_baseline)))

    next_ = 0

    for bl in xrange(len(bins_baseline[:-1])):
        for tr in xrange(num_tr):
            idxs_bin = np.linspace(bins_baseline[bl],bins_baseline[bl+1],win_lat_hist+1).astype(int)
            snip = spike_mat[tr,idxs_bin]
            lat = np.argmax(snip) # zero if no spike was found, but also zero if spike at position 0
            all_lat[tr,next_] = lat

        sort_lat[:,next_] = np.sort(all_lat[:,next_])
        hst,edges = np.histogram(sort_lat[:,next_],np.arange(win_lat_hist+1))
        # hst contains last bin as well, which will be double counted then - so exclude it
        sort_lat_hist[:,next_] = hst[:-1]
        nsort_lat_hist[:,next_] = sort_lat_hist[:,next_] / sort_lat_hist[:,next_].sum()
        next_ +=1

    # continue with test interval

    for tr in xrange(num_tr):
        idxs_bin = np.linspace(bin_test[0],bin_test[1],win_lat_hist+1).astype(int)
        snip = spike_mat[tr,idxs_bin]
        lat = np.argmax(snip) # zero if no spike was found, but also zero if spike at position 0
        all_lat[tr,next_] = lat
    lat_laser = all_lat[:,next_]  # save and write out!
    sort_lat[:,next_] = np.sort(all_lat[:,next_])
    hst,edges = np.histogram(sort_lat[:,next_],np.arange(win_lat_hist+1))
    # hst contains last bin as well, which will be double counted then - so exclude it
    sort_lat_hist[:,next_] = hst[:-1]
    nsort_lat_hist[:,next_] = sort_lat_hist[:,next_] / sort_lat_hist[:,next_].sum()


    # JS-divergence
    kn = next_+1   # number of all windows (nm baseline win. + 1 test win.)
    jsd =  np.empty((kn, kn)) * np.nan
    for k1 in np.arange(kn):
        D1 = nsort_lat_hist[:,k1] # 1st latency histogram
        for k2 in np.arange(k1+1,kn):
            D2 = nsort_lat_hist[:,k2]  # 2nd latency histogram
            jsd[k1,k2] = np.sqrt(JSdiv(D1,D2)*2)  # pairwise modified JS-divergence (real metric!)

    # Calculate p-value and information difference
    p, I = makep(jsd,kn)
    print('p: {} | I: {}'.format(p,I))

    return p,I,lat_laser

def JSdiv(P,Q):
    '''
    JSDIV   Jensen-Shannon divergence. Adapted from matlab code:
    http://dx.doi.org/10.1038/nature12176

    D = JSDIV(P,Q) calculates the Jensen-Shannon divergence of the two
    input distributions.
    '''
    P = np.array(P,dtype=float)
    Q = np.array(Q,dtype=float)

    # Input argument check
    if(np.abs(np.sum(P)-1) > 0.00001) or (np.abs(np.sum(P)-1) > 0.00001):
        print('Input arguments must be probability distributions.')
        sys.exit()

    if P.shape != Q.shape:
        print('Input distributions must be of the same size.')
        sys.exit()

    # JS-divergence
    M = (P + Q) / 2.
    D1 = entropy(P,M)
    D2 = entropy(Q,M)
    D = (D1 + D2) / 2.
    return D

def makep(jsd,kn):
    '''
    Calculates p value from distance matrix.
    Adapted from matlab code:
    http://dx.doi.org/10.1038/nature12176
    '''

    pnhk = jsd[0:kn-1,0:kn-1]
    nullhypkld = pnhk[~np.isnan(pnhk)]  # nullhypothesis

    testkld = np.median(jsd[0:kn-1,kn-1]) # value to test
    sno = float(len(nullhypkld))   # sample size for nullhyp. distribution
    p_value = np.sum(nullhypkld>=testkld) / sno
    Idiff = testkld - np.median(nullhypkld)   # information difference between baseline and test latencies

    return p_value, Idiff


###########################################################################################
# old stuff:

def rolling_sum(a, n) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


def shuffle_timestamps(collected_spikes,iterations,ms_window,stim_length,cut_off_stimulus):
    max_rolling_all= []
    min_rolling_all= []
    pbar_shuffle = tqdm(total=int(iterations))

    for iteration in xrange(iterations):
        pbar_shuffle.update(1)
        pbar_shuffle.set_description('Shuffling')
        shuffled = np.zeros((collected_spikes.shape[0],2*ms_window))
        for row in xrange(collected_spikes.shape[0]-1):
            shuffled[row,:] = np.random.permutation(collected_spikes[row+1,collected_spikes.shape[1]/2-ms_window:collected_spikes.shape[1]/2+ms_window])
        sum_shuffled = np.sum(shuffled,axis=0)
        sum_roll_shuffled = rolling_sum(sum_shuffled,n=[3 if stim_length < cut_off_stimulus else 100][0]) # XX ms bins (adjacent)
        idx_max= np.argmax(sum_roll_shuffled[ms_window:])
        idx_min= np.argmin(sum_roll_shuffled[ms_window:])

        max_roll = sum_roll_shuffled[ms_window+idx_max]
        min_roll = sum_roll_shuffled[ms_window+idx_min]

        max_rolling_all.append(max_roll)
        min_rolling_all.append(min_roll)
    pbar_shuffle.close()
    ninetynine_max = np.percentile(max_rolling_all, 99.9)
    ninetyfive_max = np.percentile(max_rolling_all, 95)
    ninetynine_min = np.percentile(min_rolling_all, .1)
    ninetyfive_min = np.percentile(min_rolling_all, 5)
    return max_rolling_all,min_rolling_all,ninetyfive_max,ninetynine_max,ninetyfive_min,ninetynine_min


print('Loaded analysis helpers: Stimulus')
