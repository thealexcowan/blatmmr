import sys
from sage.all import *
import cProfile
import random
import numpy
import math
from scipy.signal import convolve


EPS = 10.0**(-9)
PI = float(pi)


def main():
    run_tests()



################
### Defaults ###
################


# at the end of this section there is DEFAULT = Defaults()
class Defaults:
    def __init__(self):
        self.BETA = 0.8
        self.SIGMA = 0.7
        self.X_MAX = 7
        self.NUM_PTS = 1001
        self.LIKELIHOOD_TYPE = 'fft'
        self.KERNEL_TYPE = 'fft'
        self.KERNEL = {0.03:1.0} #{0.025:0.9, 0.06:0.1} # sigma : weight
        self.KERNEL_LAPLACE = {0.008:0.97, 0.1:0.03}
        self.SUPPORT = None # use get_default_support; the first time that's called, it'll run make_default_support()
        self.LIKELIHOOD_DATA_FFT = None #  make_default_likelihood_data_fft()
        self.KERNEL_DATA_FFT_TRIM = None # make_default_kernel_data_fft(trim=True)
        self.KERNEL_DATA_FFT_NOTRIM = None # make_default_kernel_data_fft(trim=False)



def get_default_player(**kwargs):
    mu = kwargs.get('mu', 0.0)
    sigma = kwargs.get('sigma', DEFAULT.SIGMA)
    fn = lambda x: math.exp(-(x-mu)**2/(2.0*sigma**2))/sqrt(2*PI*sigma**2)
    x_vals = get_default_support()
    default_player = Player.from_function(fn, x_vals, **kwargs)
    return default_player


def get_default_support():
    if DEFAULT.SUPPORT is None:
        DEFAULT.SUPPORT = make_default_support()
    return DEFAULT.SUPPORT


def make_default_support():
    x_max = DEFAULT.X_MAX
    x_min = -x_max
    num_pts = DEFAULT.NUM_PTS
    delta = (x_max - x_min)/float(num_pts - 1)
    default_support = [x_min + tmp*delta for tmp in range(num_pts)]
    return default_support


def get_default_beta():
    default_beta = DEFAULT.BETA
    return default_beta


def get_default_likelihood_type():
    return DEFAULT.LIKELIHOOD_TYPE


def get_default_likelihood_data(likelihood_type=None):
    if likelihood_type is None:
        likelihood_type = get_default_likelihood_type()
    if likelihood_type == 'naive':
        default_likelihood_data = get_default_likelihood_data_naive()
    elif likelihood_type == 'laplace':
        default_likelihood_data = get_default_likelihood_data_laplace()
    elif likelihood_type == 'fft':
        default_likelihood_data = get_default_likelihood_data_fft()
    else:
        raise NotImplementedError
    return default_likelihood_data


def get_default_likelihood_data_naive():
    def fn(x,y):
        return 1.0/(1 + math.exp(y-x))
    return fn


def get_default_likelihood_data_laplace():
    to_ret = {1.0:1.0}
    return to_ret


def get_default_likelihood_data_fft():
    if DEFAULT.LIKELIHOOD_DATA_FFT is None:
        DEFAULT.LIKELIHOOD_DATA_FFT = make_default_likelihood_data_fft()
    return DEFAULT.LIKELIHOOD_DATA_FFT


def make_default_likelihood_data_fft():
    fn = lambda x: 1.0/(1 + math.exp(-x)) - (1 if x > 0 else (0 if x < 0 else 0.5))
    x_vals = get_default_support()
    default_likelihood_data = [fn(x) for x in x_vals]
    return default_likelihood_data


def get_default_kernel_type():
    return DEFAULT.KERNEL_TYPE


def get_default_kernel_data(kernel_type=None, trim=None):
    if kernel_type is None:
        kernel_type = get_default_kernel_type()
    if kernel_type == 'naive':
        default_kernel_data = get_default_kernel_data_naive()
    elif kernel_type == 'laplace':
        default_kernel_data = get_default_kernel_data_laplace()
    elif kernel_type == 'fft':
        default_kernel_data = get_default_kernel_data_fft(trim=trim)
    else:
        raise NotImplementedError
    return default_kernel_data


def get_default_kernel_data_naive():
    sigma_dict = DEFAULT.KERNEL
    def fn(x,y):
        sumval = 0
        for sigma, weight in sigma_dict.items():
            sumval += weight * math.exp(-(x-y)**2/(2.0*sigma**2)) / sqrt(2*PI*sigma**2)
        return sumval
    return fn


def get_default_kernel_data_laplace():
    default_kernel_data = DEFAULT.KERNEL_LAPLACE
    return default_kernel_data


def get_default_kernel_data_fft(trim=None):
    if trim is None:
        trim = True # using None to mean default, beware!
    if trim:
        if DEFAULT.KERNEL_DATA_FFT_TRIM is None:
            DEFAULT.KERNEL_DATA_FFT_TRIM = make_default_kernel_data_fft(trim=True)
        default_kernel_data = DEFAULT.KERNEL_DATA_FFT_TRIM
    else:
        if DEFAULT.KERNEL_DATA_FFT_NOTRIM is None:
            DEFAULT.KERNEL_DATA_FFT_NOTRIM = make_default_kernel_data_fft(trim=False)
        default_kernel_data = DEFAULT.KERNEL_DATA_FFT_NOTRIM
    return default_kernel_data


def make_default_kernel_data_fft(trim=None):
    if trim is None:
        trim = True # using None to mean default, beware!
    sigma_dict = DEFAULT.KERNEL
    fn = lambda x: diysum([math.exp(-x**2/(2.0*sigma**2))/sqrt(2*PI*sigma**2) * weight for sigma, weight in sigma_dict.items()])
    x_vals = get_default_support()
    default_kernel_data = [fn(x) for x in x_vals]
    if trim:
        default_kernel_data = [val for val in default_kernel_data if val > EPS]
    return default_kernel_data


DEFAULT = Defaults()


##############
### Player ###
##############


class Player:
    '''
    Constructors
    Info/helper
    Match Processing
    Likelihood
    Kernel
    '''
    
    ####################
    ### Constructors ###
    ####################
    # __init__(self, support, probs, **kwargs)
    # extract_kwargs(self, old_kwargs)
    # extract_kernel_info(self, old_kwargs)
    # extract_likelihood_info(self, old_kwargs)
    # extract_beta(self, old_kwargs)
    # staticmethod from_default()
    # staticmethod from_dict(probs_dict, **kwargs)
    # staticmethod from_function(fn, x_vals, **kwargs)
    # staticmethod from_gaussian(mu, sigma, num_samples, x_bounds=None, kernel=None, **kwargs)
    # staticmethod from_samples(samples, **kwargs)
    # staticmethod from_uniform(num_samples, x_bounds=None, **kwargs)
    # staticmethod from_delta(loc, num_samples, x_bounds, **kwargs)
    # copy(self)

    
    def __init__(self, probs, normalize=True, **kwargs):
        self.probs = probs
        self.support = kwargs.get('support', get_default_support())
        self.beta = kwargs.get('beta', get_default_beta())
        self.likelihood_type = kwargs.get('likelihood_type', get_default_likelihood_type())
        self.likelihood_data = kwargs.get('likelihood_data', get_default_likelihood_data(self.likelihood_type))
        self.kernel_type = kwargs.get('kernel_type', get_default_kernel_type())
        self.kernel_data = kwargs.get('kernel_data', get_default_kernel_data(self.kernel_type, trim=kwargs.get('trim',None)))
        self.kwargs = kwargs
        self._mu = None
        self._sigma = None
        self._heaviside_cdf = None
        self._support_set = None
        self._support_value_to_index = None
        if normalize:
            self.normalize()
    

    def support_set(self):
        if self._support_set is None:
            self._support_set = set(self.support)
        return self._support_set


    def support_value_to_index(self, v):
        if self._support_value_to_index is None:
            self._support_value_to_index = {v:k for k,v in enumerate(self.support)}
        return self._support_value_to_index[v]
    

    @staticmethod
    def from_default(**kwargs):
        return get_default_player(**kwargs)
    

    @staticmethod
    def from_dict(probs_dict, **kwargs):
        support = []
        probs = []
        for k,v in sorted(probs_dict.items()):
            support.append(k)
            probs.append(v)
        player = Player(probs, support=support, **kwargs)
        return player

    
    @staticmethod
    def from_function(fn, x_vals, **kwargs):
        player = {}
        for x in x_vals:
            player[x] = fn(x)
        player = Player.from_dict(player, **kwargs)
        return player
    
    
    @staticmethod
    def from_gaussian(mu, sigma, num_samples, x_bounds=None, kernel=None, **kwargs):
        if x_bounds is not None:
            x_min, x_max = x_bounds
            x_min = float(x_min)
            x_max = float(x_max)
        else:
            x_min = mu - 4.0*sigma
            x_max = mu + 4.0*sigma
        fn = lambda x: math.exp(-(x-mu)**2/(2*sigma**2))/sqrt(2*PI*sigma**2)
        x_vals = [x_min + i*(x_max - x_min)/num_samples for i in range(num_samples + 1)]
        player = Player.from_function(fn, x_vals, **kwargs)
        return player

    
    @staticmethod
    def from_samples(samples, **kwargs):
        player = Player.from_dict({float(x):1.0 for x in samples}, **kwargs)
        return player    
    
    
    @staticmethod
    def from_uniform(num_samples, x_bounds=None, **kwargs):
        if x_bounds is not None:
            x_min, x_max = x_bounds
            x_min = float(x_min)
            x_max = float(x_max)
        else:
            x_min = min(get_default_support())
            x_max = max(get_default_support())
        fn = lambda x: 1.0
        x_vals = [x_min + i*(x_max - x_min)/float(num_samples) for i in range(num_samples+1)]
        player = Player.from_function(fn, x_vals, **kwargs)
        return player
    

    @staticmethod
    def from_delta(loc, num_samples, x_bounds, **kwargs):
        x_min, x_max = x_bounds
        delta = (x_max - x_min)/(float(num_samples)-1)
        player_dict = {x_min + delta*i : 0.0 for i in range(num_samples)}
        player_dict[loc] = 1.0
        player = Player.from_dict(player_dict, **kwargs)
        return player
    

    def copy(self):
        # assumes support won't change
        new_player = Player(list(self.probs), normalize=False, **self.kwargs)
        new_player._mu = self._mu
        new_player._sigma = self._sigma
        new_player._support_set = self._support_set
        new_player._support_value_to_index = self._support_value_to_index
        return new_player
    
    
    
    ############
    ### Info ###
    ############
    # __repr__(self)
    # __str__(self)
    # set_probs(self, new_probs, normalize=True)
    # support(self)
    # mean(self)
    # std(self)
    # mode(self)
    # mu(self)
    # sigma(self)
    # clear_stats(self)
    # sample(self, num_samples)
    # plot(self, **kwargs)
    # get_plot(self, **kwargs)
    # prob_tups(self)
    # total_mass(self)
    # normalize(self, in_place=True)
    
    
    def __repr__(self):
        k_str_list = []
        v_str_list = []
        for i,k in enumerate(self.support):
            v = self.probs[i]
            k_str = '%.2f' % round(k, 4)
            v_str = '%.4f' % round(v, 6)
            k_str_list.append(k_str)
            v_str_list.append(v_str)
        max_k_str_len = max([len(ks) for ks in k_str_list])
        max_v_str_len = max([len(vs) for vs in v_str_list])
        to_ret = ''
        for i in range(len(k_str_list)):
            k_str = k_str_list[i]
            v_str = v_str_list[i]
            k_str = ' '*(max_k_str_len - len(k_str)) + k_str
            v_str = ' '*(max_v_str_len - len(v_str)) + v_str
            to_ret += k_str+': '+v_str+'\n'
        to_ret += 'kernel_type: ' + self.kernel_type + '\n'
        to_ret += 'kernel_data: ' + str(self.kernel_data) + '\n'
        to_ret += 'likelihood_type: ' + self.likelihood_type + '\n'
        to_ret += 'likelihood_data: ' + str(self.likelihood_data) + '\n'
        to_ret += 'beta: ' + str(self.beta) # + '\n'
        return to_ret


    def __str__(self):
        return self.__repr__()
    

    def set_probs(self, new_probs, normalize=True):
        self.clear_stats()
        self.probs = new_probs
        if normalize:
            self.normalize()
    
    
    def mean(self):
        mu_val = 0.0
        for i,k in enumerate(self.support):
            mu_val += self.probs[i] * k
        return mu_val
    
        
    def std(self):
        mu = self.mu()
        std_val = 0.0
        for i,k in enumerate(self.support):
            std_val += self.probs[i] * (k - mu)**2
        try:
            std_val = sqrt(std_val)
        except ValueError:
            print(self)
            raise
        return std_val
    
    
    def mode(self):
        kv_list = self.prob_tups()
        kv_list.sort(key = lambda tup: tup[1])
        to_ret = kv_list[-1][0]
        return to_ret

    
    def mu(self):
        if self._mu is None:
            self._mu = self.mean()
        return self._mu


    def sigma(self):
        if self._sigma is None:
            self._sigma = self.std()
        return self._sigma

    
    def clear_stats(self):
        self._mu = None
        self._sigma = None
        self._heaviside_cdf = None
    
    
    def sample(self, num_samples):
        samples = numpy.random.choice(self.support, num_samples, p=self.probs)
        return samples
    

    def plot(self, **kwargs):
        # sage specific
        P = self.get_plot(**kwargs)
        show(P)
    
    
    def get_plot(self, **kwargs):
        # sage specific
        P = points(self.prob_tups(), **kwargs)
        return P


    def prob_tups(self):
        return [(self.support[i], self.probs[i]) for i in range(len(self.support))]
        

    def total_mass(self):
        total_mass = diysum(self.probs)
        return total_mass
    
    
    def normalize(self, in_place=True, round_to_zero=False):
        if not in_place:
            new_player = self.copy()
            new_player.normalize(in_place=True)
            return new_player
        total_mass = self.total_mass()
        if total_mass < EPS:
            print(str(self))
            raise ValueError
        if abs(total_mass - 1) > EPS:
            tmp = 1 # for profiling
            for i in range(len(self.support)):
                self.probs[i] /= total_mass
            if round_to_zero: # can be needed for numerical stability?
                thresh = EPS/len(self.support)
                for i in range(len(self.support)):
                    if abs(self.probs[i]) < thresh:
                        self.probs[i] = 0


    
    ########################
    ### Match processing ###
    ########################
    # process_match(self, other, won, apply_kernel=True, in_place=True)
    # match_likelihood(self, other, won)
    # posterior(self, likelihood_values, in_place=True)
    
    
    def process_match(self, other, won, apply_kernel=True, in_place=True):
        if not in_place:
            new_player = self.copy()
            new_player.process_match(other, won, apply_kernel=apply_kernel, in_place=True)
            return new_player
        L = self.match_likelihood(other, won)
        self.posterior(L, in_place=True)
        if apply_kernel:
            self.apply_kernel(in_place=True)

    
    def match_likelihood(self, other, won):
        if (won in RR) and (won not in ZZ):
            L_won = self.likelihood_fn(other, True)
            L_lost = self.likelihood_fn(other, False)
            L = exponential_combination([(L_won, won), (L_lost, 1-won)])
        else:
            L = self.likelihood_fn(other, won)
        return L
    
    
    def posterior(self, likelihood_values, in_place=True):
        if not in_place:
            new_player = self.copy()
            new_player.posterior(likelihood_values, in_place=True)
            return new_player
        self.clear_stats() # This is DIY set_probs
        for i in range(len(self.support)):
            self.probs[i] *= likelihood_values[i]
        self.normalize()
    
    
    
    ##################
    ### Likelihood ###
    ##################
    # likelihood_fn(self, other, won)
    # win_likelihood_noconstant(self, other)
    ###  Naive
    # likelihood_naive(self, other)
    ### Laplace
    # likelihood_laplace(self, other)
    # T(self, b, extra_eval_pts=None)
    ### FFT
    # likelihood_fft(self, other)
    # heaviside_cdf(self, weight_at_zero=0.5)
    # get_heaviside_cdf(self, weight_at_zero=0.5)
    

    def likelihood_fn(self, other, won):
        L = self.win_likelihood_noconstant(other)
        if won:
            L = affine_rescaling(L, self.beta, (1 - self.beta)/2.0)
        else:
            L = affine_rescaling(L, -self.beta, (1 + self.beta)/2.0)
        return L
    
    
    def win_likelihood_noconstant(self, other):
        # Probability of winning if Lambda is rescaled to go from 0 to 1.
        if self.likelihood_type == 'naive':
            L = self.likelihood_naive(other)
        elif self.likelihood_type == 'laplace':
            L = self.likelihood_laplace(other)
        elif self.likelihood_type == 'fft':
            L = self.likelihood_fft(other)
        else:
            message = 'Unhandled self.likelihood_type: ' + self.likelihood_type
            raise NotImplementedError(message)
        return L


    ### Likelihood: Naive ###
    # Expecting that self.likelihood_data is a function of two variables

    def likelihood_naive(self, other):
        L = []
        for i in range(len(self.support)):
            winprob = 0.0
            for j in range(len(other.support)):
                winprob += other.probs[j] * self.likelihood_data(self.support[i], other.support[j])
            L.append(winprob)
        return L
        
    
    ### Likelihood Laplace ###
    # Expecting that self.likelihood_data is a dict {b: weight}

    def likelihood_laplace(self, other):
        L_dict = {z:0.0 for z in self.support}
        for b, weight in self.likelihood_data.items():
            T_vals = other.T(b, extra_eval_pts=self.support_set())
            for z in self.support:
                L_dict[z] += T_vals[z] * weight
        L = []
        for z in self.support:
            L.append(L_dict[z])
        return L
    
    
    def T(self, b, extra_eval_pts=None):
        T_vals = {}
        if extra_eval_pts:
            support = set(self.support_set())
            support.update(extra_eval_pts)
            support = list(support)
            support.sort()
        else:
            support = self.support
        z_prev = support[0]
        M = 0.0
        L = 0.0
        for z in support:
            delta = z - z_prev
            if z in self.support_set():
                self_z = self.probs[self.support_value_to_index(z)]
            else:
                self_z = 0
            M += self_z
            L *= math.exp(-delta/b)
            L += 0.5 * self_z
            T_vals[z] = M - L
            z_prev = z
        R = 0.0
        support.reverse()
        for z in support:
            delta = z_prev - z
            R *= math.exp(-delta/b)
            T_vals[z] += R
            if z in self.support_set():
                R += 0.5 * self.probs[self.support_value_to_index(z)]
            z_prev = z
        return T_vals
    
    
    ### Likelihood FFT ###
    # Expecting that self.likelihood_data is a list representing difference between actual likelihood and Heaviside function

    def likelihood_fft(self, other):
        heaviside_vals = other.heaviside_cdf()
        adjustment_vals = convolve(other.probs, self.likelihood_data, mode='same')
        L = sum_lists(heaviside_vals, adjustment_vals)
        return L
    

    def heaviside_cdf(self, weight_at_zero=0.5):
        # sum_{x<y} self[x] + weight_at_zero * self[y]
        if self._heaviside_cdf is None:
            self._heaviside_cdf = self.get_heaviside_cdf(weight_at_zero=weight_at_zero)
        return self._heaviside_cdf
    
        
    def get_heaviside_cdf(self, weight_at_zero=0.5):
        heaviside_vals = []
        cdf_val = 0.0
        if abs(weight_at_zero) < EPS:
            for y in self.probs:
                heaviside_vals.append(cdf_val)
                cdf_val += y
        elif abs(weight_at_zero - 1) < EPS:
            for y in self.probs:
                cdf_val += y
                heaviside_vals.append(cdf_val)
        else:
            one_minus_weight_at_zero = 1 - weight_at_zero
            for y in self.probs:
                cdf_val += weight_at_zero * y
                heaviside_vals.append(cdf_val)
                cdf_val += one_minus_weight_at_zero * y
        return heaviside_vals
    

    
    ##############
    ### Kernel ###
    ##############
    # apply_kernel(self, in_place=True)
    ### Naive
    # apply_kernel_naive(self)
    # kernel_val(self, x,y)
    ### Laplace
    # kernel_is_valid(self)
    # kernel_val_laplace(self, x, y)
    # apply_kernel_laplace(self)
    # S(self, h, extra_eval_pts=None)
    ### FFT
    # apply_kernel_fft(self)

    
    def apply_kernel(self, in_place=True):
        if not in_place:
            new_player = self.copy()
            new_player.apply_kernel(in_place=True)
            return new_player
        if self.kernel_type == 'naive':
            self.apply_kernel_naive()
        elif self.kernel_type == 'laplace':
            self.apply_kernel_laplace()
        elif self.kernel_type == 'fft':
            self.apply_kernel_fft()
        else:
            message = 'Unhandled self.kernel_type: ' + self.kernel_type
            raise NotImplementedError(message)
    
    
    ### Kernel: Naive ###
    # Expecting that self.kernel_data is a function of two variables
    
    def apply_kernel_naive(self):
        new_probs = []
        for i in range(len(self.support)):
            new_val = 0.0
            for j in range(len(self.support)):
                new_val += self.probs[i] * self.kernel_data(self.support[i], self.support[j])
            new_probs.append(new_val)
        self.set_probs(new_probs)
    
    
    ### Kernel: Laplace ###
    # Expecting that self.kernel_data is a dict {b : weight}
    
    def laplace_kernel_is_valid(self):
        scales_nonnegative = all([b >= 0 for b in self.kernel_data])
        weights_nonnegative = all([weight >= 0 for weight in self.kernel_data.values()])
        weights_sum_to_one = abs(diysum(self.kernel_data.values()) - 1.0) < EPS
        to_ret = scales_nonnegative and weights_nonnegative and weights_sum_to_one
        return to_ret

    
    def kernel_val_laplace(self, x, y):
        val = 0.0
        for b, weight in self.kernel_data.items():
            if abs(b) < EPS:
                raise NotImplementedError
            else:
                val += weight/(2.0*b) * math.exp(-abs(x-y)/b)
        return val


    def apply_kernel_laplace(self):
        new_probs = [0.0 for i in range(len(self.support))]
        for b, weight in self.kernel_data.items():
            if abs(b) < EPS:
                S_vals = self.probs
                S_vals = scale_list(S_vals, weight)
                new_probs = sum_lists(new_probs, S_vals)
            else:
                S_vals = self.S(b)
                # S_vals will be scaled such that when interpolated and interpreted as a density, the integral is 1.
                # This means that, as a discrete probability distribution, if the points in the support are equally spaced, the values will sum to approximately len(support)/(x_max - x_min).
                S_vals = normalize_dict(S_vals)
                for i,k in enumerate(self.support):
                    new_probs[i] += weight * S_vals[k]
        self.set_probs(new_probs, normalize=False) # passing normalize=False means this assumes weights in kernel sum to 1.
    
    
    def S(self, b, extra_eval_pts=None):
        S_vals = {}
        if extra_eval_pts:
            support = set(self.support_set())
            support.update(extra_eval_pts)
            support = list(support)
            support.sort()
        else:
            support = self.support
        z_prev = support[0]
        L = 0.0
        for z in support:
            delta = z - z_prev
            L *= math.exp(-delta/b)
            if z in self.support_set():
                L += 0.5/b * self.probs[self.support_value_to_index(z)]
            S_vals[z] = L
            z_prev = z
        R = 0.0
        support.reverse()
        for z in support:
            delta = z_prev - z
            R *= math.exp(-delta/b)
            S_vals[z] += R
            if z in self.support_set():
                R += 0.5/b * self.probs[self.support_value_to_index(z)]
            z_prev = z
        return S_vals
    
    
    ### Kernel: FFT ###
    # Expecting that self.kernel_data is a list
    
    def apply_kernel_fft(self):
        new_probs = convolve(self.probs, self.kernel_data, mode='same')
        self.set_probs(new_probs)
    
    


##############
### Helper ###
##############

                
def normalize_dict(prob_dict):
    total_mass = diysum(prob_dict.values())
    new_dict = {k:v/total_mass for k,v in prob_dict.items()}
    return new_dict


def sum_lists(list1, list2):
    min_len = min(len(list1), len(list2))
    new_list = []
    for i in range(min_len):
        new_list.append(list1[i] + list2[i])
    return new_list


def scale_list(list1, weight):
    new_list = [weight * val for val in list1]
    return new_list


def diysum(myiterable):
    # sum(myiterable) is sage specific
    sumval = 0
    for tmp in myiterable:
        sumval += tmp
    return sumval

                
def exponential_combination(term_list):
    '''
    term_list is list of tuples (term, term_weight)
    Each term is a list or dict representing a function
    Returns prod term**term_weight
    If all terms are dicts, result keys will be intersection of term keys
    If all terms are lists, result will ignore all elements with index >= min([len(term) for term in term_list])
    '''
    if all([isinstance(term, dict) for term, term_weight in term_list]):
        support = set(term_list[0][0])
        for term, term_weight in term_list[1:]:
            support.intersection_update(term)
        comb = {}
        for x in support:
            val = 1.0
            for term, term_weight in term_list:
                val *= term[x]**term_weight
            comb[x] = val
    elif all([isinstance(term, list) for term, term_weight in term_list]):
        comb = []
        for i in range(min([len(term) for term, term_weight in term_list])):
            val = 1.0
            for term, term_weight in term_list:
                val *= term[i]**term_weight
            comb.append(val)
    else:
        raise NotImplementedError
    return comb


def affine_rescaling(fn, scale, constant):
    '''
    fn is a list or dict representing a function
    returns fn*scale + constant
    '''
    if isinstance(fn, dict):
        new_fn = {}
        for k in fn:
            new_fn[k] = fn[k]*scale + constant
    elif isinstance(fn, list):
        new_fn = []
        for v in fn:
            new_fn.append(v*scale + constant)
    else:
        raise NotImplementedError
    return new_fn


def play_match(A, B, A_won, in_place=True):
    if not in_place:
        Ac = A.copy()
        Bc = B.copy()
        play_match(Ac, Bc, A_won, in_place=True)
        return (Ac, Bc)
    L_A = A.match_likelihood(B, A_won) # DIY process_match so order doesn't matter
    B.process_match(A, 1 - A_won) # 1-True == 0, 1-False == 1. This is to handle A_won == 0.5 for draws
    A.posterior(L_A)
    A.apply_kernel()



#############
### Tests ###
#############


def run_tests():
    play_match_test()
    types_test()
    constructor_test()
    trim_test()


def play_match_test(AB_tup=None):
    # Assuming A,B are equal and symmetric for this test
    if AB_tup is None:
        A = Player.from_default()
        B = Player.from_default()
    else:
        A, B = AB_tup
    shift = 1 #len(A.probs)%2
    assert(all([abs(A.probs[i] - A.probs[-i-shift]) < EPS for i in range(len(A.probs))]))
    (Ac, Bc) = play_match(A, B, True, in_place=False)
    # Winning increases as much as losing decreases if priors for A and B are the same.
    assert(abs(Ac.mu() + Bc.mu()) < EPS)
    play_match(A, B, 0.5)
    # Tie between identical players should leave them identical
    assert(all([abs(A.probs[i] - B.probs[i]) < EPS for i in range(len(A.probs))]))
    play_match(A, B, 1)
    assert(abs(A.mu() + B.mu()) < EPS)
    assert(all([abs(A.probs[i] - B.probs[-i-shift]) < EPS for i in range(len(A.probs))]))


def types_test():
    # Takes about 1 minute
    types = ['naive', 'laplace', 'fft']
    for likelihood_type in types:
        for kernel_type in types:
            A = Player.from_default(likelihood_type=likelihood_type, kernel_type=kernel_type)
            B = A.copy()
            play_match_test((A,B))


def constructor_test(verbose=False):
    # staticmethod from_gaussian(mu, sigma, num_samples, x_bounds=None, kernel=None, **kwargs)
    # staticmethod from_samples(samples, **kwargs)
    # staticmethod from_uniform(num_samples, x_bounds=None, **kwargs)
    # staticmethod from_delta(loc, num_samples, x_bounds, **kwargs)
    AB_list = []
    
    A = Player.from_gaussian(0.0, 2.0, 500)
    AB_list.append((A, A.copy()))
    
    samples = [random.random() for _ in range(300)]
    samples += [-tmp for tmp in samples]
    A = Player.from_samples(samples)
    AB_list.append((A, A.copy()))
    
    A = Player.from_uniform(966)
    AB_list.append((A, A.copy()))
    
    A = Player.from_delta(0.0, 300, (-5,5))
    AB_list.append((A, A.copy()))

    for i, AB_tup in enumerate(AB_list):
        if verbose:
            print(i)
        play_match_test(AB_tup)


def trim_test():
    A_trim = Player.from_default(trim=True)
    B_trim = A_trim.copy()
    A_notrim = Player.from_default(trim=False)
    B_notrim = A_notrim.copy()
    play_match(A_trim, B_trim, False)
    play_match(A_notrim, B_notrim, False)
    assert(all([abs(A_trim.probs[i] - A_notrim.probs[i]) < EPS for i in range(len(A_trim.probs))]))
    assert(all([abs(B_trim.probs[i] - B_notrim.probs[i]) < EPS for i in range(len(B_trim.probs))]))




#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()')
    else:
        main()
