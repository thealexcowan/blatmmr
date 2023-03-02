import sys
from sage.all import *
import cProfile
import random
import numpy
import math
import live
import csv
import glicko
import rating

PREC = 400
K = RealField(PREC)
PI = K(pi)


def main():
    pass


###

def modified_BT_posterior_density(mean, variance, beta, y):
    def integrand(t):
        integrand_val = beta/(1 + 10.0**((y-t)/400.0))
        integrand_val *= exp(-(t-mean)**2/(2.0*variance)) / sqrt(2*PI*variance)
        return integrand_val
    int_val = numerical_integral(integrand, -Infinity, +Infinity)[0]
    denominator_val = int_val + (1 - beta)/2.0

    def density(x):
        L_val = (1 - beta)/2.0
        L_val += beta/(1 + 10.0**((y-x)/400.0))
        prior_val = exp(-(x-mean)**2/(2.0*variance)) / sqrt(2*PI*variance)
        numerator_val = L_val * prior_val
        density_val = numerator_val / denominator_val
        return density_val

    return density


def get_denominator_integrand(mean, variance, beta, y):
    def denominator_integrand(t):
        integrand_val = beta/(1 + 10**((y-t)/K(400)))
        integrand_val *= exp(-(t-mean)**2/(K(2)*variance)) / sqrt(2*PI*variance)
        return integrand_val
    return denominator_integrand


def get_numerator_integrand(mean, variance, beta, y):
    def numerator_integrand(t):
        integrand_val = beta/(1 + 10**((y-t)/K(400)))
        integrand_val *= exp(-(t-mean)**2/(K(2)*variance)) / sqrt(2*PI*variance)
        integrand_val *= t
        return integrand_val
    return numerator_integrand


def modified_BT_posterior_mean(mean, variance, beta, y):
    eps_abs = 10**(-PREC/10)
    eps_rel = 10**(-5)
    lower = -2000
    upper = 6000
    
    denominator_integrand = get_denominator_integrand(mean, variance, beta, y)
    denominator_int_val = numerical_integral(denominator_integrand, lower, upper, eps_abs=eps_abs, eps_rel=eps_rel)[0]
    denominator_val = denominator_int_val
    if beta != 1:
        denominator_val += (1 - beta)/K(2)
    
    numerator_integrand = get_numerator_integrand(mean, variance, beta, y)
    numerator_int_val = numerical_integral(numerator_integrand, lower, upper, eps_abs=eps_abs, eps_rel=eps_rel)[0]
    numerator_val = numerator_int_val
    if beta != 1:
        numerator_val += mean*(1 - beta)/K(2)
    
    new_mean = numerator_val / denominator_val
    return new_mean


def get_mu_diffs(var_list=None, beta_list=None, mu_list=None, y=2000, verbose=False):
    if var_list is None:
        var_list = [tmp**2 for tmp in [50,100,150,200,250,300]]
    if beta_list is None:
        beta_list = [K(4)/5, K(99)/100, 1]
    if mu_list is None:
        mu_list = list(range(4001))
    mu_diffs = {}
    for beta in beta_list:
        if verbose:
            print(beta)
        mu_diffs[beta] = {}
        for var in var_list:
            tmp_mu_diffs = {}
            for mu0 in mu_list:
                mu1 = modified_BT_posterior_mean(mu0, var, beta, y)
                tmp_mu_diffs[mu0] = mu1 - mu0
            mu_diffs[beta][int(sqrt(var))] = dict(tmp_mu_diffs)
    return mu_diffs


def make_mu_diff_plots(mu_diffs, y=2000, clean_pts=True, color_list=None, **kwargs):
    # mu_diffs == {sigma:{mu0:mu1-mu0}}
    if color_list is None:
        color_list = [hue(tmp*(1+sqrt(5.0))/2) for tmp in range(len(mu_diffs))]
    if 'dpi' not in kwargs:
        kwargs['dpi'] = 400
    if 'size' not in kwargs:
        kwargs['size'] = 2
    
    plt_list = []
    for i, (sigma, mudiffvals) in enumerate(sorted(mu_diffs.items())):
        color = color_list[i]
        pts = mudiffvals.items()
        if clean_pts:
            pts.sort()
            tmp_pts = [pts[0]]
            for j in range(2,len(pts)):
                ppval = pts[j-2]
                pval = pts[j-1]
                val = pts[j]
                if (0.8*ppval[1] <= pval[1]) and (pval[1] <= 1.2*ppval[1]) and (0.8*val[1] <= pval[1]) and (pval[1] <= 1.2*val[1]):
                    tmp_pts.append(pval)
                else:
                    print('Discarding pt:', sigma, j, pval)
            tmp_pts.append(pts[-1])
            pts = tmp_pts
        pts_plt = points(pts, color=color, **kwargs)
        
        maxtup = sorted(pts, key = lambda tup: -tup[1])[0]
        maxk, maxv = maxtup
        tmp_kwargs = dict(kwargs)
        tmp_kwargs['size'] *= 4
        maxpt_plt = points([maxtup], color=color, faceted=True, markeredgecolor='black', zorder=100, **tmp_kwargs)

        textstr = ''
        textstr += '$\sigma = %s$' % sigma
        textstr += '\n'
        textstr += '$(%s, %.1f)$' % (maxk, maxv)
        t1 = text(textstr, (maxk, maxv), fontsize=11, fontweight='bold', color='black')
        #t1 = text(textstr, (maxk-100, maxv+0.3), fontsize=11, fontweight='bold', color='black') # for mu_diff_beta_1_08_sigma_50.png
        #t1 = text(textstr, (maxk+180, maxv-1), fontsize=11, fontweight='bold', color='black') # for beta == 1
        t2 = text(textstr, (maxk, maxv), fontsize=11, color=color)

        sigma_plt = pts_plt
        sigma_plt += maxpt_plt
        sigma_plt += t1
        #sigma_plt += t2
        plt_list.append(sigma_plt)

    ymax = max([tmp.ymax() for tmp in plt_list])
    lplt = line([(y,0), (y,ymax)], color='grey', alpha=0.5, zorder=-1)
    plt_list.append(lplt)

    return plt_list
        

###


def get_rating_at(plr, time, strict=False):
    mu = 0.0
    sigma = 0.7
    for tup in plr.rating_history:
        if (tup[0] > time) or ((not strict) and (tup[0] == time)):
            break
        mu = tup[1]
        sigma = tup[2]
    return (mu, sigma)


def get_player_ml(playername):
    aliases = get_aliases()
    if playername in aliases:
        playernames = set(aliases[playername])
    else:
        playernames = set([playername])
    ml = []
    with open('data.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            match = live.Match.from_data(row, method='maser')
            if any([pn in [match.p1, match.p2] for pn in playernames]):
                ml.append(match)
    if playername in aliases:
        for match in ml:
            match.p1 = aliases.get(match.p1, [match.p1])[0]
            match.p2 = aliases.get(match.p2, [match.p2])[0]
    ml.sort(key = lambda m: m.start_time)
    return ml


def get_player_ml_dat(players, playername, ml, exclude=100):
    shift = len(players[playername].rating_history) - exclude
    pdat = []
    for i,tmp in enumerate(players[playername].rating_history[-shift:]):
        delta = tmp[1] - players[playername].rating_history[-shift+i-1][1]
        tmpm = ml[-shift+i]
        pdat.append((tmp[1], delta, tmpm))
    return pdat


def get_diffs_by_diff(players, playername, pdat, sigmathresh=None):
    aliases = get_aliases()
    diffs_by_diff = []
    for tup in pdat:
        pr = tup[0]
        d1 = tup[1]
        opp = [tmp for tmp in [tup[2].p1, tup[2].p2] if tmp != playername][0]
        oppr, oppsigma = get_rating_at(players[opp], tup[2].start_time)
        d2 = pr - oppr
        if (sigmathresh is None) or (oppsigma < sigmathresh):
            diffs_by_diff.append((d2, d1))
    return diffs_by_diff


def playername_to_diffs(players, playername, ml=None, exclude=100, sigmathresh=None):
    if ml is None:
        ml = get_player_ml(playername)
    pdat = get_player_ml_dat(players, playername, ml, exclude=exclude)
    diffs_by_diff = get_diffs_by_diff(players, playername, pdat, sigmathresh=sigmathresh)
    return diffs_by_diff


def get_aliases():
    aliases_list = []
    aliases_list.append(['example1', 'example2'])
    aliases = {}
    for al in aliases_list:
        for tmp in al:
            aliases[tmp] = al
    return aliases



###

def process_log_losses(data, glicko=False, omit_draws=True, sigma_thresh=0.4):
    # data == [(match, (p1.mu(), p1.sigma()), (p2.mu(), p2.sigma()), negative_log_likelihood)] or [(match, (p1_glicko, p2_glicko), negative_log_likelihood_glicko)]
    # from live.get_log_losses

    to_ret = []
    
    # make diffs
    diffs = []
    if not glicko:
        for (match, (p1_mu, p1_sigma), (p2_mu, p2_sigma), negative_log_likelihood) in data:
            result = match.result
            if (not omit_draws) or (abs(result - 0.5) > rating.EPS):
                if (sigma_thresh is None) or ((p1_sigma < sigma_thresh) and (p2_sigma < sigma_thresh)):
                    diffs.append((p2_mu - p1_mu, negative_log_likelihood))
    else:
        for (match, (p1, p2), negative_log_likelihood) in data:
            result = match.result
            if (not omit_draws) or (abs(result - 0.5) > rating.EPS):
                r1 = (p1.mu - 1500) * log(10.0)/400
                r2 = (p2.mu - 1500) * log(10.0)/400
                s1 = p1.phi * log(10.0)/400
                s2 = p2.phi * log(10.0)/400
                if (sigma_thresh is None) or ((s1 < sigma_thresh) and (s2 < sigma_thresh)):
                    diffs.append((r2-r1, negative_log_likelihood))
    to_ret += [diffs]
        
    # stats
    total_log_loss = sum([tmp[1] for tmp in diffs])
    avg_log_loss = total_log_loss / len(diffs)
    expected_log_loss = 0 # negative log loss
    expected_log_loss_variance = 0
    for (d, llv) in diffs:
        # this is probably wrong for draws?
        ellv = exp(-llv)
        expected_log_loss -= -llv * ellv + (1 - ellv)*log(1 - ellv)
        expected_log_loss_variance += ellv * (1 - ellv) * (log(1/ellv - 1))**2
    to_ret += [(total_log_loss, avg_log_loss, expected_log_loss, expected_log_loss_variance)]

    # make cdf indices
    lowerval = -3
    delta = 0.02
    numpts = int(-2*lowerval/delta) + 1 # 301
    eval_pts = [lowerval + delta*tmp for tmp in range(numpts)]
    lowerind = 0
    diffs.sort()
    cdfindices = {}
    for ind, tup in enumerate(diffs):
        if tup[0] > lowerval+delta:
            cdfindices[lowerind] = ind
            lowerval += delta
            lowerind += 1

    # make smoothedvals
    sigma = 0.03
    sqrt2pi = sqrt(2*float(pi))
    def kern(x,y,sgma=sigma):
        to_ret = exp(-(x-y)**2/(2*sgma**2)) / (sqrt2pi * sgma)
        return to_ret

    index_diff = int(5*sigma/delta) + 1
    smoothedvals = {}
    for tmpc, y in enumerate(eval_pts):
        if y >= 0:
            sumval = 0
            minindex_plus = cdfindices.get(tmpc-index_diff, len(diffs))
            maxindex_plus = cdfindices.get(tmpc+index_diff, len(diffs))
            minindex_minus = cdfindices.get(len(eval_pts)-tmpc-index_diff, 0)
            maxindex_minus = cdfindices.get(len(eval_pts)-tmpc+index_diff, 0)
            indices = set(list(range(minindex_plus, maxindex_plus)) + list(range(minindex_minus, maxindex_minus)))
            for index in indices:
                (df, llv) = diffs[index]
                sumval += llv * kern(abs(df), y, sgma=sigma)
            smoothedvals[y] = sumval

    # make loss density
    x_max = 2.5
    loss_density = {k : v/(len(diffs) * (1 + erf(k/(sigma*sqrt(2.0))))) for k,v in smoothedvals.items()} # not normalized yet
    avg_per_pt = sum([tmp[1] for tmp in diffs if abs(tmp[0]) < x_max]) / len([tmp[1] for tmp in diffs if abs(tmp[0]) < x_max])
    int_approx = x_max * sum(loss_density.values()) / len(loss_density.values())
    loss_density = {k : avg_per_pt/int_approx * v for k,v in loss_density.items()}
    to_ret += [avg_per_pt, loss_density]

    return to_ret


        


#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()')
    else:
        main()
