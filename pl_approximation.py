import numpy as np
import math
from frequency_oracle import *

def multi_piece_closed_solution(Y, break_points, return_residuals = False):
    '''
    We always compute the slopes from the whole domain.
    In this way, we can manage to alliviate the noise utilizing information from all collected noisy messages.
    Returns:
        A: parameters (a_0, a_1, ...., a_k), a_0 refers to the intercept, and others refer to the slopes.
    '''
    d = len(Y)
    S = np.array(break_points).copy()
    D = np.arange(d)
    D1 = np.repeat(np.expand_dims(np.arange(d), 1), len(S)-1, axis = 1)
    E1 = (D1 > S[:-1]) * (D1 <= S[1:])
    E1[:,0] += D == S[0]
    X = np.hstack([np.expand_dims(np.ones(d), 1), (D1 - S[:-1]) * E1])
    D2 = np.repeat(np.expand_dims(np.arange(d), 1), len(S)-2, axis = 1)
    E2 = D2 > S[1:-1]
    B = (S[1:-1] - S[0:-2]) * E2
    B = np.hstack([np.expand_dims(np.zeros(d), 1), B, np.expand_dims(np.zeros(d), 1)])
    X = X + B
    A = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y)
    if return_residuals == True:
        residuals = X.dot(A) - Y
        loss = residuals.T.dot(residuals)
        return A, loss
    else:
        return A


def search_the_best_break_points(Y, break_points, l, r, return_all_break_points = False, step = 1):
    '''
    Given the break points of (k-1) segements, find the best break point of k-th segments between range [l,r]
    Returns:
        break_point(s): if return_all_break_points is false, return a breakpoint, else return a breakpoint and all break_points
        best_slopes: of all segments
    '''
    index = np.where(break_points >= r)[0][0]
    new_break_points = np.insert(break_points, index, -1)
    best_slopes = None
    minimal_error = np.infty
    best_break_point = -1
    for i in range(l, r, step):
        new_break_points[index] = i+0.5
        slopes, error = multi_piece_closed_solution(Y, new_break_points, True)
        if error < minimal_error:
            minimal_error = error
            best_slopes = slopes
            best_break_point = i + 0.5
    if return_all_break_points:
        new_break_points[index] = best_break_point
        return best_break_point, new_break_points, best_slopes
    else:
        return best_break_point, best_slopes
    

def search_with_multi_level_steps(Y, break_points, l, r, return_all_break_points = False, minima_segment_length = 2, support_searched_domain_size = arguments.search_granularity):
    '''
    the formal pl fitting function
    '''
    # compute the granularity and how deep the overviews needed
    deep = 0
    domain_size = r - l + 1
    while True:
        g = math.ceil(domain_size / support_searched_domain_size)
        if g == 1:
            break
        elif g <= support_searched_domain_size:
            deep += 1
            break
        else: # g > support_at_most_domain_size:
            g = support_searched_domain_size
            deep += 1
            domain_size = math.ceil(domain_size / g)
    # find the best break point from the coarse domain repeatedly until deep = 0
    fined_l = l + minima_segment_length - 1
    fined_r = r - minima_segment_length + 1
    while deep >= 0:
        step = g ** deep 
        deep -= 1
        best_break_point, new_break_points, best_slopes= search_the_best_break_points(Y, break_points, fined_l, fined_r, True, step)
        fined_l = int(best_break_point + 0.5 - step)
        fined_l = fined_l if fined_l >= l + minima_segment_length - 1 else l + minima_segment_length - 1
        fined_r = int(best_break_point - 0.5 + step)
        fined_r = fined_r if fined_r <= r - minima_segment_length + 1 else r - minima_segment_length + 1
    if return_all_break_points:
        return best_break_point, new_break_points, best_slopes
    else:
        return best_break_point, best_slopes
    

# ---------------------------------------------------the following is the functions for pl error estimation-----------------------------


def piecewise_linear_approximate_function(break_points, slopes):
    '''
    We compute the approximate count according to the fitted parameters
    '''
    d = int(break_points[-1] - break_points[0] + 1)
    A = slopes.copy()
    S = np.array(break_points).copy()
    D = np.arange(d)
    D1 = np.repeat(np.expand_dims(np.arange(d), 1), len(S)-1, axis = 1)
    E1 = (D1 > S[:-1]) * (D1 <= S[1:])
    E1[:,0] += D == S[0]
    X = np.hstack([np.expand_dims(np.ones(d), 1), (D1 - S[:-1]) * E1])
    D2 = np.repeat(np.expand_dims(np.arange(d), 1), len(S)-2, axis = 1)
    E2 = D2 > S[1:-1]
    B = (S[1:-1] - S[0:-2]) * E2
    B = np.hstack([np.expand_dims(np.zeros(d), 1), B, np.expand_dims(np.zeros(d), 1)])
    X = X + B
    Y =  np.dot(X,A)   
    return Y


def residuals(Y, break_points, slopes):
    '''
    compute the residual of each points
    '''
    fitted_y = piecewise_linear_approximate_function(break_points, slopes)
    y = Y.copy()
    return y - fitted_y


def RSS(Y, break_points, slopes, total_not_segment = False):
    '''
    compute residual sum of squares (RSS) 
    
    Args:
        Y: the actual samples
        break_points: (k+1) breakpoints
        slopes: one intercept and k slopes, (k+1) parameters in total
    '''
    residual_sqr = np.square(residuals(Y, break_points, slopes))
    if total_not_segment:
        total_RSS = np.sum(residual_sqr)
        return total_RSS
    else:
        segments_RSS = [residual_sqr[math.floor(break_points[i])+1: math.floor(break_points[i+1])+1].sum() for i in range(len(break_points)-1)]
        segments_RSS[0] += residual_sqr[0]
        return segments_RSS