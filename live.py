import sys
from sage.all import *
import cProfile
import random
import datetime
import json
import csv
import rating
import glicko


#################
### Constants ###
#################


FILENAME = 'data.csv' # p1_username, p2_username, p1_won, 2022-12-16 21:52:32.114+00
REFERENCE_TIME = '2022-12-16 00:00:00.000+00'
DEFAULT_RATING_SCALE = 'glicko' # for printing


############
### Main ###
############


def main():
    pass


def example(t_max=None):
    # t_max in seconds
    players = process_matches(t_max=t_max, verbose=True)
    top10 = get_top_players(players, 10)
    player_plot = get_player_plots(top10, dpi=400, ticks=[[],None])
    show(player_plot, figsize=8, legend_loc=(0.025,0.7))
    return players



###############
### Classes ###
###############


class Match:

    def __init__(self, p1, p2, result, start_time, f1=None, f2=None):
        self.p1 = p1 # In Maser data, this is username on 2023/01/11
        self.p2 = p2 # In the future, I should handle this more robustly, accounting for aliases e.g.
        self.result = result # 'win' means p1 win
        self.format_result() # Standardizing so that 1 means p1 won, 0 means p2 won, and 0.5 means draw
        self.start_time = start_time
        self.format_start_time() # Standardizing so to be a float which is number of seconds since 2022-12-16 0:00:00+00
        self.f1 = f1
        self.f2 = f2
    

    def format_result(self):
        win_results = [1, 1.0, True, 'win', 'WIN', '1-0']
        loss_results = [0, 0.0, False, 'loss', 'LOSS', '0-1']
        draw_results = [0.5, 'draw', 'DRAW', '1/2-1/2', '0.5-0.5']
        
        win_val = 1
        loss_val = 0
        draw_val = 0.5
        
        if self.result in win_results:
            formatted_result = win_val
        elif self.result in loss_results:
            formatted_result = loss_val
        elif self.result in draw_results:
            formatted_result = draw_val
        else:
            error_message = 'Unhandled self.result: ' + str(self.result)
            raise NotImplementedError(error_message)
        
        self.result = formatted_result
        return


    def format_start_time(self):
        reference_time = REFERENCE_TIME
        start_time_formatted = get_seconds_since(self.start_time, reference_time)
        self.start_time = start_time_formatted
    

    @staticmethod
    def from_data(data, method='maser'):
        possible_methods = ['maser']
        method = method.lower()
        if method not in possible_methods:
            error_message = 'method '+str(method)+' not in possible_methods '+str(possible_methods)
            raise NotImplementedError(error_message)

        if method == 'maser':
            # csv file with lines like
            # stimulant,stimulant2,loss,2022-12-16 21:52:32.114+00
            # ['stimulant', 'stimulant2', 'loss', '2022-12-16 21:52:32.114+00']
            # data assumed to be one such line
            if len(data) == 4:
                p1, p2, result, timestr = data
                f1 = None
                f2 = None
            else:
                p1, p2, result, f1, f2, timestr = data
                f1 = int(f1)
                f2 = int(f2)
            match = Match(p1, p2, result, timestr, f1, f2)
        
        return match
    

    def __repr__(self):
        to_ret = ''
        to_ret += 'start_time: ' + str(self.start_time) + '\n'
        to_ret += 'p1: ' + self.p1 + '\n'
        to_ret += 'p2: ' + self.p2 + '\n'
        to_ret += 'result: ' + str(self.result) #+ '\n'
        return to_ret



###############
### Helpers ###
###############


def timestr_to_datetime(timestr):
    timestr = timestr.replace('T', ' ')
    timestr_split = timestr.split('+')
    timestr = timestr_split[0]
    if len(timestr_split) > 1:
        TZ = timestr_split[1]
        if TZ != '00':
            raise NotImplementedError
    if '.' not in timestr:
        timestr += '.000'
    time = datetime.datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S.%f')        
    return time


def get_seconds_since(timestr, starttime=None):
    if starttime is None:
        starttime = datetime.datetime.now()
    elif type(starttime) == type(''):
        starttime = timestr_to_datetime(starttime)
    time = timestr_to_datetime(timestr)
    time_delta = (time - starttime).total_seconds()
    return time_delta


def time_between(t, t_min=None, t_max=None):
    min_ok = (t_min is None) or (t_min < t)
    max_ok = (t_max is None) or (t < t_max)
    to_ret = min_ok and max_ok
    return to_ret



####################
### Data reading ###
####################


def maser_data_to_match_list(filename=FILENAME):
    # Takes about 30s
    match_list = []
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            match = Match.from_data(row, method='maser')
            match_list.append(match)
    match_list.sort(key = lambda m: m.start_time)
    return match_list



#######################
### Data processing ###
#######################


def process_matches(match_list=None, players=None, t_min=None, t_max=None, verbose=False, **kwargs):
    # Processes in the order of match_list
    # Matches in match_list expected to be Match objects
    # Takes very roughly 0.01s per match
    if match_list is None:
        match_list = maser_data_to_match_list()
    if players is None:
        players = {}
    aliases = get_aliases()
    
    counter = 0
    for match in match_list:
        if verbose:
            if (counter % (len(match_list)/1000)) == 0:
                sys.stdout.write('\r'+str(counter+1)+' / '+str(len(match_list)))
                sys.stdout.flush()
            counter += 1
        
        result = match.result # 1 if p1 won, 0 if p2 won, 0.5 if draw
        timestamp = match.start_time # number of seconds since REFERENCE_TIME
        p1_name = match.p1
        p2_name = match.p2
        if p1_name in aliases:
            p1_name = aliases[p1_name][0]
        if p2_name in aliases:
            p2_name = aliases[p2_name][0]
        
        if time_between(timestamp, t_min, t_max):
            # add new players to players dict
            for p_name in [p1_name, p2_name]:
                if p_name not in players:
                    p = rating.Player.from_default(**kwargs)
                    p.rating_history = [(timestamp-1, p.mu(), p.sigma())] # -1 as in 1 second before the match started
                    p.name = p_name
                    players[p_name] = p
            # process match
            p1 = players[p1_name]
            p2 = players[p2_name]
            try:
                rating.play_match(p1, p2, result)
                p1.rating_history.append((timestamp, p1.mu(), p1.sigma()))
                p2.rating_history.append((timestamp, p2.mu(), p2.sigma()))
            except ValueError as e:
                print(e)
                return players
    
    if verbose:
        print('')
    return players


def get_aliases():
    aliases_list = []
    aliases_list.append(['example1', 'example2'])
    aliases = {}
    for al in aliases_list:
        for tmp in al:
            aliases[tmp] = al
    return aliases



################
### Printing ###
################


def get_top_players(players, num, game_thresh=0):
    top_players = []
    sorted_players = sorted(players.values(), key = lambda tmp: -tmp.rating_history[-1][1]) # rating_history is list of (timestamp, mu, sigma)
    for p in sorted_players: 
        if len(p.rating_history) > game_thresh:
            top_players.append(p)
            if len(top_players) >= num:
                break
    return top_players


def print_player(p, starting_str='', ratingscale=DEFAULT_RATING_SCALE, return_print_str=False):
    mu = p.mu()
    sigma = p.sigma()
    if ratingscale in ['fide', 'glicko']:
        mu = mu*400/log(10.0) + 1500
        mu = int(mu)
        sigma = sigma*400/log(10.0)
        sigma = int(sigma)
        mu_str = '%s' % mu
        sigma_str = '+- %s' % sigma
    else:
        mu_str = '%.3f' % mu
        sigma_str = '+- %.3f' % sigma
    if mu_str[0] != '-':
        mu_str = ' ' + mu_str
    num_games_str = str(len(p.rating_history) - 1)
    name_str = str(p.name)
    
    to_print = starting_str
    if starting_str:
        to_print += '  '
    to_print += mu_str + ' ' + sigma_str
    to_print += '    '
    to_print += num_games_str
    to_print += '    '
    to_print += name_str
    
    print(to_print)
    if return_print_str:
        to_ret = to_print
    else:
        to_ret = None
    return to_ret


def print_player_list(player_list, with_indices=True, ratingscale=DEFAULT_RATING_SCALE):
    for index, p in enumerate(player_list):
        if not with_indices:
            index_str = ''
        else:
            max_index_str_len = len(str(len(player_list)+1))
            index_str = str(index + 1)
            index_str += ' '*(max_index_str_len - len(index_str))
        print_player(p, starting_str=index_str, ratingscale=ratingscale)
    return


def get_rating_history_plot(player, color='blue', t_min=None, t_max=None, with_lines=True, label=None, ratingscale=DEFAULT_RATING_SCALE, **kwargs):
    # This uses sagemath
    if 'size' not in kwargs:
        kwargs['size'] = 2
    data = []
    for time,mu,sigma in player.rating_history:
        if ratingscale in ['fide', 'glicko']:
            mu = mu*400/log(10.0) + 1500
        if time_between(time, t_min, t_max):
            data.append((time,mu))
    if label is None:
        plt = points(data, color=color, **kwargs)
    else:
        plt = points(data, color=color, legend_label=label, legend_color=color, **kwargs)
    if with_lines:
        plt += line(data, color=color, zorder=kwargs.get('zorder',None), thickness=0.5)
    return plt


def get_player_plots(player_list, t_min=None, t_max=None, with_label=True, with_lines=True, with_params=True, **kwargs):
    plot_list = []
    for i,player in enumerate(player_list):
        current_hue = float(i) * (1 - sqrt(5.0))/2
        if with_label:
            label = str(player.name)
        else:
            label = None
        tmp_kwargs = dict(kwargs)
        tmp_kwargs['zorder'] = 100 - i
        player_plot = get_rating_history_plot(player, color=hue(current_hue), t_min=t_min, t_max=t_max, label=label, with_lines=with_lines, **tmp_kwargs)
        plot_list.append(player_plot)

    if with_params:
        beta_text = 'beta: ' + '%.2f' % rating.DEFAULT.BETA
        sigma_text = 'sigma: ' + '%.3f' % rating.DEFAULT.SIGMA
        pts_text = 'num_pts: ' + str(rating.DEFAULT.NUM_PTS)
        bound_text = 'bound: ' + '%.3f' % rating.DEFAULT.X_MAX
        ker_text = 'kernel: '
        for k,v in sorted(rating.DEFAULT.KERNEL.items()):
            ker_text += '%.4f' % k + ' : ' + '%.4f' % v + ',  '
        ker_text = ker_text[:-3]
        params_text = beta_text + ', ' + sigma_text + ', ' + pts_text + ', ' + bound_text + '\n' + ker_text
        
        xmin = min([plt.xmin() for plt in plot_list])
        xmax = max([plt.xmax() for plt in plot_list])
        ymin = min([plt.ymin() for plt in plot_list])
        ymax = max([plt.ymax() for plt in plot_list])
        
        ypos = max(ymin + 0.1, 0.1)
        xpos = xmin
        
        tplt = text(params_text, (xpos, ypos), color='black', fontsize='large', horizontal_alignment='left')
        plot_list.append(tplt)
    
    to_ret = sum(plot_list)
    return to_ret




################
### Analysis ###
################


def get_log_losses(match_list=None, t_min=None, t_max=None, glicko_only=False, verbose=False, **kwargs):
    # Processes in the order of match_list
    # Matches in match_list expected to be Match objects
    # Takes very roughly 0.01s per match
    if match_list is None:
        match_list = maser_data_to_match_list()
    players = {}
    players_glicko = {}
    aliases = get_aliases()
    match_likelihoods = []
    Q = log(10.0)/400
    PI = float(pi)
    g_const = 3*Q**2/PI**2
    
    counter = 0
    for match in match_list:
        if verbose:
            if (counter % (len(match_list)/1000)) == 0:
                sys.stdout.write('\r'+str(counter+1)+' / '+str(len(match_list)))
                sys.stdout.flush()
            counter += 1
        
        result = match.result # 1 if p1 won, 0 if p2 won, 0.5 if draw
        timestamp = match.start_time # number of seconds since REFERENCE_TIME
        p1_name = match.p1
        p2_name = match.p2
        if p1_name in aliases:
            p1_name = aliases[p1_name][0]
        if p2_name in aliases:
            p2_name = aliases[p2_name][0]
        
        if time_between(timestamp, t_min, t_max):
            # add new players to players dicts
            for p_name in [p1_name, p2_name]:
                if not glicko_only:
                    if p_name not in players:
                        p = rating.Player.from_default(**kwargs)
                        p.rating_history = [(timestamp-1, p.mu(), p.sigma())] # -1 as in 1 second before the match started
                        p.name = p_name
                        players[p_name] = p
                if p_name not in players_glicko:
                    p = glicko.Rating()
                    players_glicko[p_name] = [(timestamp-1, p)] # -1 as in 1 second before the match started
            
            if not glicko_only:
                # rating process
                p1 = players[p1_name]
                p2 = players[p2_name]

                L1 = p1.match_likelihood(p2, result)
                expected_prob = 0
                for i,prb in enumerate(p1.probs):
                    expected_prob += prb * L1[i]
                negative_log_likelihood = -log(expected_prob)
                ll_data = (match, (p1.mu(), p1.sigma()), (p2.mu(), p2.sigma()), negative_log_likelihood)

                try:
                    rating.play_match(p1, p2, result)
                    p1.rating_history.append((timestamp, p1.mu(), p1.sigma()))
                    p2.rating_history.append((timestamp, p2.mu(), p2.sigma()))
                except ValueError as e:
                    print(e)
                    return (players, players_glicko, match_likelihoods)

            # glicko process
            p1_glicko = players_glicko[p1_name][-1][1]
            p2_glicko = players_glicko[p2_name][-1][1]
            
            # Glicko (16)
            g_val = 1.0/sqrt(1 + g_const * (p1_glicko.phi**2 + p2_glicko.phi**2))
            expo_val = -g_val * (p1_glicko.mu - p2_glicko.mu)/400.0
            p_val = 1.0/(1 + 10**expo_val) # prob of p1 winning
            if abs(result - 0) < rating.EPS:
                p_val = 1 - p_val
            elif abs(result - 0.5) < rating.EPS:
                p_val = sqrt(p_val * (1 - p_val))
            negative_log_likelihood_glicko = -log(p_val)
            ll_data_glicko = (match, (p1_glicko, p2_glicko), negative_log_likelihood_glicko)
            
            if abs(result - 1) < rating.EPS:
                new_p1, new_p2 = glicko.Glicko2.rate_1vs1(p1_glicko, p2_glicko, is_draw=False)
            elif abs(result - 0) < rating.EPS:
                new_p2, new_p1 = glicko.Glicko2.rate_1vs1(p2_glicko, p1_glicko, is_draw=False)
            elif abs(result - 0.5) < rating.EPS:
                new_p1, new_p2 = glicko.Glicko2.rate_1vs1(p1_glicko, p2_glicko, is_draw=True)
            else:
                raise ValueError('Got result = '+str(result))
            players_glicko[p1_name].append((timestamp, new_p1))
            players_glicko[p2_name].append((timestamp, new_p2))

            if not glicko_only:
                match_likelihoods.append((ll_data, ll_data_glicko))
            else:
                match_likelihoods.append(ll_data_glicko)
            
    if verbose:
        print('')
    if not glicko_only:
        to_ret = (players, players_glicko, match_likelihoods)
    else:
        to_ret = (players_glicko, match_likelihoods)
    return to_ret



#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()')
    else:
        main()
