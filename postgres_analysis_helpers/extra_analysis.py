# Space for extra analysis draw_waveforms
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def wavef_width_single(data,i,sample_rate,plotting):
    '''
    Function to run over data (cursor)
    '''
    if plotting:
        figure = plt.figure(figsize=(2,2))
        sns.set(font_scale=1.1)


    artefact = False
    #find highest amplitude waveform
    idx = np.argmax([data.iloc[i].mean_wf[x].ptp() for x in [0,1,2,3]])

    # cubic spline interpolation
    tck = interpolate.splrep(np.arange(0,len(data.iloc[i].mean_wf[idx])), data.iloc[i].mean_wf[idx], s=0) # calculate cubic spline, no smoothing
    xnew = np.arange(0, len(data.iloc[i].mean_wf[idx]), 0.001) # 1000x upsampling
    wf_interp = interpolate.splev(xnew, tck, der=0) # no derivative, just extrapolate
    len_wf_interp = len(xnew)

    max_wf = np.max(wf_interp[:int(len_wf_interp/2)]) # find max in first half
    idx_max_wf = np.argmax(wf_interp[:int(len_wf_interp/2)])
    min_wf = np.min(wf_interp[idx_max_wf:]) # find min after max
    idx_min_wf = np.argmin(wf_interp[idx_max_wf:])
    idx_min_wf = idx_min_wf+idx_max_wf

    if idx_max_wf == 0:
            return idx_max_wf,idx_min_wf,1,np.nan

    # check borderline criterion:
    idx_min_before = np.argmin(wf_interp[:idx_max_wf])
    min_wf_before = np.min(wf_interp[:idx_max_wf])


    if (np.median(wf_interp[-100:]) < min_wf) or (idx_min_wf > .8*len_wf_interp) or (np.abs(min_wf_before)/np.abs(min_wf)>10):
        artefact = True
        print('{}: possibly artefact! code: {}'.format(i,int(artefact)))

    swidth = (idx_min_wf-idx_max_wf)/sample_rate # in milliseconds, not seconds, because of the 1000x upsampling!

    if plotting:
        ax = figure.add_subplot(111)
        ax.plot(wf_interp,c='k',alpha= .7)
        ax.scatter(idx_max_wf,max_wf,s=60,c='b')
        ax.scatter(idx_min_wf,min_wf,s=60,c='b')
        if artefact:
            ax.scatter(idx_min_before,min_wf_before,s=60,c='r')
        if artefact:
            ax.set_title('{}: {:.2f} ms'.format(i,swidth),color='r')
        else:
            ax.set_title('{}: {:.2f} ms'.format(i,swidth))


    if plotting:
        plt.show()
    return idx_max_wf,idx_min_wf,int(artefact),swidth



def wavef_width_multiple(data,indices,sample_rate,plotting):
    if plotting:
        figure = plt.figure(figsize=(15,15))
        sns.set(font_scale=1.1)

    for no,i in enumerate(indices):
        artefact = False
        #find highest amplitude waveform
        idx = np.argmax([data.iloc[i].mean_wf[x].ptp() for x in [0,1,2,3]])

        # cubic spline interpolation
        tck = interpolate.splrep(np.arange(0,len(data.iloc[i].mean_wf[idx])), data.iloc[i].mean_wf[idx], s=0) # calculate cubic spline, no smoothing
        xnew = np.arange(0, len(data.iloc[i].mean_wf[idx]), 0.001) # 1000x upsampling
        wf_interp = interpolate.splev(xnew, tck, der=0) # no derivative, just extrapolate
        len_wf_interp = len(xnew)

        max_wf = np.max(wf_interp[:len_wf_interp/2]) # find max in first half
        idx_max_wf = np.argmax(wf_interp[:len_wf_interp/2])
        min_wf = np.min(wf_interp[idx_max_wf:]) # find min after max
        idx_min_wf = np.argmin(wf_interp[idx_max_wf:])
        idx_min_wf = idx_min_wf+idx_max_wf

        # check borderline criterion:
        idx_min_before = np.argmin(wf_interp[:idx_max_wf])
        min_wf_before = np.min(wf_interp[:idx_max_wf])


        if (np.median(wf_interp[-100:]) < min_wf) or (idx_min_wf > .8*len_wf_interp) or (np.abs(min_wf_before)/np.abs(min_wf)>10):
            artefact = True
            print('{}: possibly artefact! code: {}'.format(i,int(artefact)))

        swidth = (idx_min_wf-idx_max_wf)/sample_rate # in milliseconds, not seconds, because of the 1000x upsampling!

        if plotting:
            ax = figure.add_subplot(4,4,no+1)
            ax.plot(wf_interp,c='k',alpha= .7)
            ax.scatter(idx_max_wf,max_wf,s=60,c='b')
            ax.scatter(idx_min_wf,min_wf,s=60,c='b')
            if artefact:
                ax.scatter(idx_min_before,min_wf_before,s=60,c='r')
            if artefact:
                ax.set_title('{}: {:.2f} ms'.format(i,swidth),color='r')
            else:
                ax.set_title('{}: {:.2f} ms'.format(i,swidth))


    if plotting:
        plt.show()
    return idx_max_wf,idx_min_wf,int(artefact),swidth



def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


print('Loaded postgres_analysis_helpers -> extra analysis')
