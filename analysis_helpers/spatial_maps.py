# create spatial plots / maps

import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

from scipy.ndimage import filters
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d

from scipy.stats import circmean,circvar

from matplotlib import pyplot as plt

from skimage import measure
from skimage.transform import rotate
import cv2

import sys
import warnings

from .signal import normxcorr2,conv2

def hd_tuning_curve(tracking_session, spiket_tracking_session, speed_cutoff,timebase_pos):
    # hd plot
    head_angle_spikes = spiket_tracking_session['head_angle']
    head_angle_spikes = (head_angle_spikes)%(2*np.pi) # transform from -pi,pi to 0,2*pi
    head_angle_spikes_hist, bins_angle = np.histogram(head_angle_spikes,bins=180,range=(0,2*np.pi),density=False)

    head_angles = tracking_session['head_angle']
    head_angles = (head_angles)%(2*np.pi) # transform from -pi,pi to 0,2*pi
    head_angle_hist, bins_angle = np.histogram(head_angles,bins=180,range=(0,2*np.pi),density=False)
    masked_head_angle_hist = np.ma.masked_array(head_angle_hist,mask=head_angle_hist==0)

    bin_half_width = abs((bins_angle[1]-bins_angle[0])/2)
    bins_angle_center = (bins_angle + bin_half_width)[:-1]

    # normalize:
    sampleTime = 1./timebase_pos
    with np.errstate(invalid='ignore'): # IGNORE WARNING divide by zero -taken care of with masked array above!
        norm_hist = head_angle_spikes_hist.astype(float)/(masked_head_angle_hist.astype(float)*sampleTime)

    # smooth hd curve:
    hist_angle_for_smooth = np.tile(norm_hist, 3)
    hist_angle_smooth = gaussian_filter1d(hist_angle_for_smooth, 2, mode='nearest')
    hist_angle_smooth = hist_angle_smooth[len(head_angle_hist):2*len(head_angle_hist)]

    return bins_angle_center,hist_angle_smooth

# MVL  = resultant_vector_length(spiket_tracking_session.head_angle[spiket_tracking_session.speed > speed_cutoff])
def tuning_curve_stats(bins_angle_center,hist_angle_smooth):
    tc_stats={}
    # r,mean,variance,std
    tc_stats['MVL'],tc_stats['mean'], tc_stats['var'], tc_stats['std'] = resultant_vector_length(bins_angle_center,hist_angle_smooth)
    return tc_stats

def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    # source: https://github.com/circstat/pycircstat/blob/master/pycircstat/descriptive.py
    """
    Computes mean resultant vector length for circular data.
    This statistic is sometimes also called vector strength.
    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain resultant vector length
    r = np.abs(cmean)
    # obtain mean
    mean = np.angle(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    # obtain variance
    variance = 1 - r
    std = np.sqrt(-2 * np.log(r))
    return r,mean,variance,std

def rayleigh(alpha, w=None, d=None, axis=None):
    """
    Computes Rayleigh test for non-uniformity of circular data.
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!
    :param alpha: sample of angles in radian
    :param w:       number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    # if axis is None:
    # axis = 0
    #     alpha = alpha.ravel()

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r,mean,variance,std = resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute Rayleigh's z (equ. 27.2)
    z = R ** 2 / n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return pval, z

def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # REQUIRED for mean vector length calculation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"


    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            cmean = ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(w, axis=axis))
        except Warning as e:
            print('Could not compute complex mean for MVL calculation', e)
            cmean = np.nan
    return cmean

def ratemap(spiket_tracking_session,tracking_session,bin_size,sigma_rate,sigma_time,box_size_cm,speed_cutoff):

    nbins = int(box_size_cm/bin_size)
    range_2Dhist = [[np.min(tracking_session.correct_x_inter), np.max(tracking_session.correct_x_inter)], [np.min(tracking_session.correct_y_inter), np.max(tracking_session.correct_y_inter)]]

    # check if tracking record is long enough for calculations to make sense: (cutoff of 50 spikes)
    if (len(spiket_tracking_session['correct_x_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)]) > 50):

        H, xedges, yedges = np.histogram2d(spiket_tracking_session['correct_x_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)],
                                       spiket_tracking_session['correct_y_inter'][(spiket_tracking_session['speed_filtered']>speed_cutoff)],
                                       bins=nbins, range=range_2Dhist)
        H = np.array(H,dtype=float)
        H = filters.gaussian_filter(H, sigma=sigma_rate, mode='constant')

        H_time, xedges_t, yedges_t = np.histogram2d(tracking_session['correct_x_inter'][(tracking_session['speed_filtered']>speed_cutoff)],
                                       tracking_session['correct_y_inter'][(tracking_session['speed_filtered']>speed_cutoff)],
                                       bins=nbins, range=range_2Dhist)
        H_time = np.array(H_time,dtype=float)
        H_time_backup = H_time

        H_time = filters.gaussian_filter(H_time, sigma=sigma_time, mode='constant')
        masked_time = np.ma.masked_where(H_time_backup==0, H_time)


        calc = True
        # time is now masked ...
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                ratemap = H/(masked_time)
                masked_ratemap = np.ma.masked_where(H_time_backup==0, ratemap)
            except Warning as e:
                xedges,yedges,masked_ratemap = fake_ratemap(nbins,range_2Dhist)
                calc = False

        if not (xedges == xedges_t).all():
            sys.stdout.write('\nEdge vector lengths in ratemap do not match.')
            sys.exit()
        elif not (yedges == yedges_t).all():
            sys.stdout.write('\nEdge vector lengths in ratemap do not match.')
            sys.exit()

    else:
        # otherwise the position data is too short ...
        xedges,yedges,masked_ratemap = fake_ratemap(nbins,range_2Dhist)
        calc = False

    return nbins, xedges, yedges, masked_ratemap, calc

def fake_ratemap(nbins,range_2Dhist):
    xedges = np.linspace(range_2Dhist[0][0],range_2Dhist[0][1],nbins)
    yedges = np.linspace(range_2Dhist[1][0],range_2Dhist[1][1],nbins)
    fake_ratemap = np.ones((len(xedges), len(yedges)))
    masked_ratemap = np.ma.masked_where(fake_ratemap == 1, fake_ratemap)
    return xedges,yedges,masked_ratemap

def spatial_autocorr(masked_ratemap, autocorr_overlap):

    ratemap_zeroed = masked_ratemap.data
    ratemap_zeroed[masked_ratemap.mask == True] = 0
    # subtract mean and calculate autocorrelation:
    autocorr = normxcorr2(ratemap_zeroed-np.mean(ratemap_zeroed),ratemap_zeroed-np.mean(ratemap_zeroed))
    # exclude regions close to the wall from final autocorr: (this is adapted from BNT code)
    autocorr = autocorr[int(np.round((autocorr.shape[0])*autocorr_overlap)):int(np.round((autocorr.shape[0])*(1-autocorr_overlap))),
                      int(np.round((autocorr.shape[1])*autocorr_overlap)):int(np.round((autocorr.shape[1])*(1-autocorr_overlap)))]
    return autocorr


def get_slice_autocorr(autocorr,center_auto,radius):
    intensities = []
    for t in np.linspace(0,2*np.pi,360):
        x = radius*np.cos(t) + int(center_auto[0])
        y = radius*np.sin(t) + int(center_auto[1])
        intensities.append(autocorr[int(x),int(y)])
    return np.array(intensities)


def calculate_gridscore(autocorr,show_plot=False):

    valid = True # Let's be hopeful.

    gauss_autocorr = gaussian_filter(autocorr, 2)
    gauss_autocorr = gauss_autocorr/gauss_autocorr.max()
    contours = measure.find_contours(gauss_autocorr, 0.2)

    centroids = []
    radii = []
    for n, contour in enumerate(contours):
        x = [p[0] for p in contour]
        y = [p[1] for p in contour]
        centroid = (sum(x) / len(contour), sum(y) / len(contour)) # mean x and mean y
        centroids.append(centroid)
        radii.append([np.mean([np.sqrt(np.square(x-centroid[0])+np.square(y-centroid[1]))])])

    center_auto = [autocorr.shape[0]/2.,autocorr.shape[1]/2.]
    center_circle_no = np.argmin([np.sqrt(np.square(center_auto[0]-centroid[0])+np.square(center_auto[0]-centroid[1])) for centroid in centroids])
    center_circ_coord = [centroids[center_circle_no][0],centroids[center_circle_no][1]]
    center_circ_radius = radii[center_circle_no][0]

    if np.max(np.abs(1-np.array(center_auto)/np.array(center_circ_coord))) > 0.1:
        sys.stdout.write('Warning: Midpoints of autocorr and center contour diverge more than 10% - gridscore calculation invalid\n')
        valid = False
    #plot
    if show_plot:
        fig, ax = plt.subplots()
        ax.imshow(autocorr, interpolation='nearest', cmap=plt.cm.gray)
        circle1 = plt.Circle((center_circ_coord[1], center_circ_coord[0]), center_circ_radius, color='w',alpha=.2)
        circle2 = plt.Circle((center_circ_coord[1], center_circ_coord[0]), .4, color='g')
        ax.add_artist(circle1);ax.add_artist(circle2)
        ax.scatter(center_circ_coord[1],center_circ_coord[0],s=10,color="red")
        ax.text(center_circ_coord[1],center_circ_coord[0],'{:.1f}'.format(center_circ_radius),color="red")
        plt.show()

    inner_bound = np.ceil(center_circ_radius)
    outer_bound = np.floor(np.min(center_auto)) # center auto is half the autocorr width and height
    radii_GNS = np.linspace(inner_bound,outer_bound,int(outer_bound-inner_bound)+1).astype(int)

    angles = [30, 60, 90, 120, 150]
    scores = []

    for radius in radii_GNS:
        baseline_intens = get_slice_autocorr(autocorr,center_auto,radius)
        #loop over all radii
        corrs = []
        for angle in angles:
            # loop over all angles for this radius
            autocorr_rot = rotate(autocorr, angle, resize=False, center=None,
                                 order=1,clip=False, mode='constant', cval=0,preserve_range=True)
            rot_intens =  get_slice_autocorr(autocorr_rot,center_auto,radius)

            r,p = pearsonr(baseline_intens,rot_intens)
            corrs.append(r)
        scores.append(np.min([corrs[1],corrs[3]]) - np.max([corrs[0],corrs[2],corrs[4]]))

    grid_score = np.max(scores)
    if np.argmax(scores) < 2:
        sys.stdout.write('Warning: Possibly wrong grid score calculation - argmax < 2\n')
        valid = False
    return grid_score,valid


print('Loaded analysis helpers: Spatial maps')
