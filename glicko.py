"""
    Player ratings object backed by glicko2.
    Originally branched from: https://github.com/sublee/glicko2
"""
import math


class Glicko2Constants:
    # The actual score for win
    WIN = 1.
    # The actual score for draw
    DRAW = 0.5
    # The actual score for loss
    LOSS = 0.

    MU = 1500
    PHI = 200 #350
    SIGMA = 0.06
    TAU = 0.5 #1.2
    EPSILON = 0.000001
    # A constant which is used to standardize the logistic function to `1/(1+exp(-x))` from `1/(1+10^(-r/400))`
    Q = math.log(10) / 400


class Rating(object):

    def __init__(self, mu=Glicko2Constants.MU, phi=Glicko2Constants.PHI, sigma=Glicko2Constants.SIGMA):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def __repr__(self):
        c = type(self)
        args = (c.__module__, c.__name__, self.mu, self.phi, self.sigma)
        return '%s.%s(mu=%.3f, phi=%.3f, sigma=%.3f)' % args
    
    def copy(self):
        new_rating = Rating(self.mu, self.phi, self.sigma)
        return new_rating



class Glicko2:

    @staticmethod
    def create_rating(mu=None, phi=None, sigma=None):
        if mu is None:
            mu = Glicko2Constants.MU
        if phi is None:
            phi = Glicko2Constants.PHI
        if sigma is None:
            sigma = Glicko2Constants.SIGMA
        return Rating(mu, phi, sigma)

    @staticmethod
    def _scale_down(rating):
        mu = (rating.mu - Glicko2Constants.MU) * Glicko2Constants.Q
        phi = rating.phi * Glicko2Constants.Q
        return Glicko2.create_rating(mu, phi, rating.sigma)

    @staticmethod
    def _scale_up(rating):
        mu = rating.mu / Glicko2Constants.Q + Glicko2Constants.MU
        phi = rating.phi / Glicko2Constants.Q
        return Glicko2.create_rating(mu, phi, rating.sigma)

    @staticmethod
    def reduce_impact(rating):
        """The original form is `g(RD)`. This function reduces the impact of
        games as a function of an opponent's RD.
        """
        return 1 / math.sqrt(1 + (3 * rating.phi ** 2) / (math.pi ** 2))

    @staticmethod
    def expect_score(rating, other_rating, impact):
        expected_result = 1. / (1 + math.exp(-impact * (rating.mu - other_rating.mu)))
        return expected_result

    @staticmethod
    def determine_sigma(rating, difference, variance):
        """Determines new sigma."""
        phi = rating.phi
        difference_squared = difference ** 2
        # 1. Let a = ln(s^2), and define f(x)
        alpha = math.log(rating.sigma ** 2)
        def f(x):
            """This function is twice the conditional log-posterior density of
            phi, and is the optimality criterion.
            """
            tmp = phi ** 2 + variance + math.exp(x)
            t1 = math.exp(x) * (difference_squared - tmp) / (2 * tmp ** 2)
            t2 = (x - alpha) / (Glicko2Constants.TAU ** 2)
            return t1 - t2
        # 2. Set the initial values of the iterative algorithm.
        a = alpha
        if difference_squared > phi ** 2 + variance:
            b = math.log(difference_squared - phi ** 2 - variance)
        else:
            k = 1
            while f(alpha - k * math.sqrt(Glicko2Constants.TAU ** 2)) < 0:
                k += 1
            b = alpha - k * math.sqrt(Glicko2Constants.TAU ** 2)
        # 3. Let fA = f(A) and f(B) = f(B)
        f_a, f_b = f(a), f(b)
        # 4. While |B-A| > e, carry out the following steps.
        # (a) Let C = A + (A - B)fA / (fB-fA), and let fC = f(C).
        # (b) If fCfB < 0, then set A <- B and fA <- fB; otherwise, just set
        #     fA <- fA/2.
        # (c) Set B <- C and fB <- fC.
        # (d) Stop if |B-A| <= e. Repeat the above three steps otherwise.
        while abs(b - a) > Glicko2Constants.EPSILON:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b <= 0: # Needs to allow for equality; this was a minor oversight in Glicko before 2022.
                a, f_a = b, f_b
            else:
                f_a /= 2
            b, f_b = c, f_c
        # 5. Once |B-A| <= e, set s' <- e^(A/2)
        return math.exp(1) ** (a / 2)

    @staticmethod
    def rate(rating, series):
        # Step 2. For each player, convert the rating and RD's onto the
        #         Glicko-2 scale.
        rating = Glicko2._scale_down(rating)
        # Step 3. Compute the quantity v. This is the estimated variance of the
        #         team's/player's rating based only on game outcomes.
        # Step 4. Compute the quantity difference, the estimated improvement in
        #         rating by comparing the pre-period rating to the performance
        #         rating based only on game outcomes.
        variance_inv = 0
        difference = 0
        if not series:
            # If the team didn't play in the series, do only Step 6
            phi_star = math.sqrt(rating.phi ** 2 + rating.sigma ** 2)
            return Glicko2._scale_up(Glicko2.create_rating(rating.mu, phi_star, rating.sigma))
        for actual_score, other_rating in series:
            other_rating = Glicko2._scale_down(other_rating)
            impact = Glicko2.reduce_impact(other_rating)
            expected_score = Glicko2.expect_score(rating, other_rating, impact)
            variance_inv += impact ** 2 * expected_score * (1 - expected_score)
            difference += impact * (actual_score - expected_score)
        difference /= variance_inv
        variance = 1. / variance_inv
        # Step 5. Determine the new value, Sigma', ot the sigma. This
        #         computation requires iteration.
        #print 'rating, difference, variance:', rating, difference, variance
        sigma = Glicko2.determine_sigma(rating, difference, variance)
        # Step 6. Update the rating deviation to the new pre-rating period
        #         value, Phi*.
        phi_star = math.sqrt(rating.phi ** 2 + sigma ** 2)
        # Step 7. Update the rating and RD to the new values, Mu' and Phi'.
        phi = 1 / math.sqrt(1 / phi_star ** 2 + 1 / variance)
        mu = rating.mu + phi ** 2 * (difference / variance)
        # Step 8. Convert ratings and RD's back to original scale.
        return Glicko2._scale_up(Glicko2.create_rating(mu, phi, sigma))

    @staticmethod
    def rate_1vs1(rating1, rating2, is_draw=False):
        return (Glicko2.rate(rating1, [(Glicko2Constants.DRAW if is_draw else Glicko2Constants.WIN, rating2)]),
                Glicko2.rate(rating2, [(Glicko2Constants.DRAW if is_draw else Glicko2Constants.LOSS, rating1)]))

    @staticmethod
    def quality_1vs1(rating1, rating2):
        expected_score1 = Glicko2.expect_score(rating1, rating2, Glicko2.reduce_impact(rating1))
        expected_score2 = Glicko2.expect_score(rating2, rating1, Glicko2.reduce_impact(rating2))
        expected_score = (expected_score1 + expected_score2) / 2
        return 2 * (0.5 - abs(0.5 - expected_score))
