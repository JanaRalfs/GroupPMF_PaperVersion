# %% Compute PMF for Case B, y>=0
import numpy as np
import scipy.special
from scipy.stats import poisson
from scipy.stats import binom
import math
import time

# --- Joint PMF for Case B

# --- we distinguish between two subcases: B0 <= g2, g2 < B0 <= g2 + g3 and B0 > g2 + g3

# --- Joint PMF of g1, g2, g3
class g1g2g3_caseB:
    def __init__(self, TN, L0, laX, laY, max_size):
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.pmf_array = np.zeros((max_size+1,max_size+1,max_size+1))
        self.maxg1 = max_size
        self.maxg2 = max_size
        self.maxg3 = max_size
        epsi = 0.000000001
        up_g3 = np.zeros((max_size + 1, max_size + 1))
        up_g2 = np.zeros((max_size + 1))
        up_g1 = max_size

        for g1 in range(0, max_size + 1):
            for g2 in range(0, max_size + 1):
                up_g2[g1] = g2
                for g3 in range(0, max_size + 1):  # range(g1, max_size + 1):
                    up_g3[g1, g2] = g3
                    vecsum = np.zeros((max_size + 1))
                    for k in range(0, g2 + 1):
                        vecsum[k] = (math.comb(g2, k) * self.L ** (g2 - k) * ((-1) ** k) * math.factorial(g1 + g3 + k) /
                                     (self.la ** (g1 + g3 + k + 1)) #first part
                                     * (1 - scipy.stats.poisson.cdf(g1 + g3 + k,self.la * min(self.T,self.L)))) #second part
                    # in some situtaions the prob get negative (numerical issues if first part of vecsum gets to large
                    # (1*10^40) and second part (1*10^-40) gets to small, therefore max(0,...)
                    # different solution??
                    self.pmf_array[g1, g2, g3] = max(0, (np.exp(-self.la * self.L) * self.la ** (g1 + g2 + g3) /
                                                         (math.factorial(g1) * math.factorial(g2) * math.factorial(g3))
                                                         * 1 / (min(self.T, self.L)) * sum(vecsum)))
                    #set bound for g3 based on g2 and g1
                    #if lambda is large, it could happen that we start we a small value and then a minus comes(0) and we stop
                    if (self.pmf_array[g1, g2, max(0, g3 - 1)] > self.pmf_array[g1, g2, g3]
                            and self.pmf_array[g1, g2, g3] < epsi):
                        break
                # set bound for g2 based on g1
                if (np.sum(self.pmf_array[g1, max(0, g2 - 1), :]) > np.sum(self.pmf_array[g1, g2, :])
                        and np.sum(self.pmf_array[g1, g2, :]) < epsi):
                    break
            # set bound for g1
            if (np.sum(self.pmf_array[max(0, g1 - 1), :, :]) > np.sum(self.pmf_array[g1, :, :])
                    and np.sum(self.pmf_array[g1, :, :]) < epsi):
                up_g1 = g1
                break

        self.up_g1 = up_g1
        self.up_g2 = up_g2
        self.up_g3 = up_g3

    def __call__(self, i, j, k):
        return self.pmf(i, j, k)

    def pmf(self, i, j, k):
        if i > self.maxg1 | j > self.maxg2 | k > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j, k]


#%% Conditional probability a1, given g1, g2 and g3
class a_g_caseB:
    def __init__(self, TN, L0, laX, laY, max_size):
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxg1 = max_size
        self.maxa1 = max_size
        self.pmf_array = np.zeros((max_size + 1, max_size + 1))

        for g in range(0, max_size+1):
            for a in range(0,g+1):
                self.pmf_array[a,g] = binom.pmf(a,g,self.p)

    def __call__(self, i, j):
        return self.pmf(i, j)

    def pmf(self, i, j):
        if i > self.maxa1 | j > self.maxg1:
            return 0
        else:
            return self.pmf_array[i, j]


# --- Conditional probability a2b, given g1, g2 and g3, B0<=g2
class b0N_a2b2a3b3_1_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.maxz = 0
        self.p = laX / (laX + laY)
        self.maxg2 = max_size
        self.maxg3 = max_size
        self.maxb0N = max_size
        self.pmf_array = np.zeros((self.Q,2*max_size + 1, max_size + 1, max_size + 1, max_size + 1, max_size + 1))
        self.pmf_array1 = np.zeros((self.Q, 2 * max_size + 1, max_size + 1, max_size + 1, max_size + 1, max_size + 1))

        for a2 in range(0, max_size+1):
            for b2 in range(0, max_size+1):
                for a3 in range(0, max_size + 1):
                    for b3 in range(0, max_size + 1):
                        for ip in range(self.R+1,self.R+self.Q+1):
                            b0 = max(0, -(ip - a2 - b2 - a3 - b3))
                            if b0 <= a2 + b2:
                                for b0N in range(0,b0+1):
                                    # first is faster and hypergeom.pmf(0,0,0,0) gives nan, even if it is 1.0
                                    self.pmf_array[ip - self.R - 1, b0N, a2, b2, a3, b3] = math.comb(a2, b0N) * math.comb(a2 + b2 - a2, b0 - b0N) / math.comb(a2 + b2, b0)
                                    self.pmf_array1[ip - self.R - 1, b0N, a2, b2, a3, b3] = scipy.stats.hypergeom.pmf(b0N,a2+b2,a2,b0)
        hi=2

    def __call__(self, i, j, k, l, m, n):
        return self.pmf(i, j, k, l, m, n)

    def pmf(self, i, j, k, l, m, n):
        if i > self.Q-1 | j > self.maxb0N | k > self.max_size | l > self.max_size | m > self.max_size | n > self.max_size:
            return 0
        else:
            return self.pmf_array[i, j, k, l, m, n]


# --- Conditional probability a2, given g2, g3 and b0N, B0<=g2
'''class a2_g2g3_b0N_1_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxa2 = max_size
        self.maxg2 = max_size
        self.maxg3 = max_size
        self.maxb0N = max_size
        self.pmf_array = np.zeros((self.Q, max_size + 1, max_size + 1, max_size + 1, max_size + 1))

        for g2 in range(0, max_size+1):
            for g3 in range(0, max_size+1):
                for ip in range(self.R+1, self.R+self.Q+1):
                    b0 = max(0, -(ip - g2 - g3))
                    if b0 <= g2:
                        for b0N in range(b0 + 1):
                            for a2 in range(b0N, g2 + 1):
                                self.pmf_array[ip-self.R-1,a2,g2,g3,b0N] = binom.pmf(a2-b0N,g2-b0,self.p)

    def __call__(self, i, j, k, l, m):
        return self.pmf(i, j, k, l, m)

    def pmf(self, i, j, k, l, m):
        if i > self.Q-1 | j > self.maxa2 | k > self.maxg2 | l > self.maxg3 | m > self.maxb0N:
            return 0
        else:
            return self.pmf_array[i, j, k, l, m]'''

# --- Conditional probability a3, given g3, B0<=g2
'''class a3_g3_1_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.maxz = 0
        self.p = laX / (laX + laY)
        self.maxa3 = max_size
        self.maxg3 = max_size
        self.pmf_array = np.zeros((max_size + 1, max_size + 1))

        for g3 in range(0, max_size+1):
            for a3 in range(0,g3+1):
                self.pmf_array[a3,g3] = scipy.stats.binom.pmf(a3,g3,self.p)

    def __call__(self, i, j):
        return self.pmf(i, j)

    def pmf(self, i, j):
        if i > self.maxa3 | j > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j]'''

# --- Conditional probability a2, given g2, g2<B0<=g2+g3
'''class a2_g2_2_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxg2 = max_size
        self.maxa2 = max_size
        self.pmf_array = np.zeros((max_size + 1, max_size + 1))

        for g2 in range(0, max_size+1):
            #for g3 in range(0, max_size+1):
                #for ip in range(self.R+1, self.R+self.Q+1):
                    #B0 = max(0, -(ip - g2 - g3))
                    #if g2 < B0 <= g2 + g3:
            for a2 in range(0,g2+1):
                #self.pmf_array[ip-self.R-1,a2,g2,g3] = binom.pmf(a2,g2,self.p)
                self.pmf_array[a2,g2] = binom.pmf(a2,g2,self.p)

    def __call__(self, i, j):
        return self.pmf(i, j)

    def pmf(self, i, j):
        if i > self.maxa2 | j > self.maxg2:
            return 0
        else:
            return self.pmf_array[i, j]'''


#%% Conditional probability b0N, given a2, g2 and g3, g2<B0<=g2+g3
class b0N_a2b2a3b3_2_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxa2 = max_size
        self.maxg2 = max_size
        self.maxg3 = max_size
        self.maxb0N = 2*max_size
        self.pmf_array = np.zeros((self.Q, 3*max_size + 1, max_size + 1, max_size + 1, max_size + 1, max_size + 1))

        for a2 in range(0, max_size+1):
            for b2 in range(0, max_size + 1):
                for a3 in range(0, max_size+1):
                    for b3 in range(0, max_size + 1):
                        for ip in range(self.R+1,self.R+self.Q+1):
                            b0 = max(0, -(ip - a2 - b2 - a3 - b3))
                            if a2 + b2 < b0 <= a2 + b2 + a3 + b3:
                                for b0N in range(0,b0 - a2 - b2 + 1):
                                    # first is faster and hypergeom.pmf(0,0,0,0) gives nan, even if it is 1.0
                                    self.pmf_array[ip - self.R - 1, b0N + a2, a2, b2, a3, b3] = math.comb(a3, b0N) * math.comb(a3 + b3 - a3, b0 -a2 - b2 - b0N) / math.comb(a3 + b3, b0 - a2 - b2)
                                    # self.pmf_array[ip - self.R - 1, b0N + a2, a2, b2, a3, b3] = scipy.stats.hypergeom.pmf(b0N,a3+b3,a3,b0-a2-b2)

    def __call__(self, i, j, k, l, m, n):
        return self.pmf(i, j, k, l, m, n)

    def pmf(self, i, j, k, l, m, n):
        if i > self.Q - 1 | j > self.maxb0N | j > self.max_size | l > self.max_size | m > self.max_size | n > self.max_size:
            return 0
        else:
            return self.pmf_array[i, j, k, l, m, n]


# --- Conditional probability a2b, given g1, g2 and g3, B0>g2+g3
'''class a2_g2_3_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxg2 = max_size
        self.maxa2 = max_size
        self.pmf_array = np.zeros((max_size + 1, max_size + 1))

        for g2 in range(0, max_size+1):
            for a2 in range(0,g2 + 1):
                self.pmf_array[a2,g2] = binom.pmf(a2,g2,self.p)

    def __call__(self, i, j):
        return self.pmf(i, j)

    def pmf(self, i, j):
        if i > self.maxa2 | j > self.maxg2:
            return 0
        else:
            return self.pmf_array[i, j]'''

#%% Conditional probability a3b, given g1, g2 and g3, B0>g2+g3
'''class a3_g3_3_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxg3 = max_size
        self.maxa3 = max_size
        self.pmf_array = np.zeros((max_size + 1, max_size + 1))

        for g3 in range(0, max_size+1):
            for a3 in range(0,g3 + 1):
                self.pmf_array[a3,g3] = binom.pmf(a3, g3, self.p)

    def __call__(self, i, j):
        return self.pmf(i, j)

    def pmf(self, i, j):
        if i > self.maxa3 | j > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j]'''


#%% Conditional probability a3b, given g1, g2 and g3, B0>g2+g3
class b0N_a2b2a3b3_3_caseB:
    def __init__(self, R, Q, TN, L0, laX, laY, max_size):
        S0 = R + Q
        self.R = R
        self.Q = Q
        self.laX = laX
        self.laY = laY
        self.la = laX + laY
        self.L = L0
        self.T = TN
        self.p = laX / (laX + laY)
        self.maxb0N = 2*max_size + self.Q
        self.maxa2 = max_size
        self.maxa3 = max_size
        self.pmf_array = np.zeros((self.Q, 2*max_size + self.Q + 1, max_size + 1, max_size + 1))

        for ip in range(self.R+1,self.R+self.Q+1):
            if ip < 0:
                g4 = -ip
                for a2 in range(0, max_size + 1):
                    for a3 in range(max_size + 1):
                        for a4 in range(0,g4 + 1):
                            self.pmf_array[ip-self.R-1,a2 + a3 + a4, a2, a3] = binom.pmf(a4, g4, self.p)

    def __call__(self, i, j, k , l):
        return self.pmf(i, j, k , l)

    def pmf(self, i, j, k, l):
        if i > self.Q - 1 | j > self.maxb0N | k > self.maxa2 | l > self.maxa3:
            return 0
        else:
            return self.pmf_array[i, j, k , l]


class joint_pmf_caseB:
    # Provides the joint pmf/conditional pmf of g1, g2, g3, a1, a2b, a2q, a3b
    def __init__(self, R0, Q0, TN, L0, laX, laY, max_size):
        # save the time
        t_start = time.time()
        # general distributions
        self.g1g2g3 = g1g2g3_caseB(TN, L0, laX, laY, max_size)
        self.a1 = a_g_caseB(TN, L0, laX, laY, max_size)
        self.a2 = a_g_caseB(TN, L0, laX, laY, max_size)
        self.a3 = a_g_caseB(TN, L0, laX, laY, max_size)
        #%%sub-case 1: B0 <= g2
        self.b0N_1 = b0N_a2b2a3b3_1_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        #self.a2_1 = a2_g2g3_b0N_1_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        #%%sub-case 2: g2 < B0 <= g2 + g3
        #self.a2_2 = a2_g2_2_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        self.b0N_2 = b0N_a2b2a3b3_2_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        #self.a3_2 = a3_g2g3_b0N_2_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        # %%sub-case 3: B0 > g2 + g3
        #self.a2_3 = a2_g2_3_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        #self.a3_3 = a3_g3_3_caseB(R0, Q0, TN, L0, laX, laY, max_size)
        self.b0N_3 = b0N_a2b2a3b3_3_caseB(R0, Q0, TN, L0, laX, laY, max_size)

        self.t_joint_pmf = time.time() - t_start


# %% PMF for Poisson arrivals with intensity lambda in period L
class poisson_pmf:
    def __init__(self, la, L, max_size=500, epsilon=0.000001):
        self.par = la * L
        self.maxx = max_size
        self.pmf_array = np.array([poisson.pmf(0, self.par)])

        for x in range(1, max_size + 1):
            self.pmf_array = np.append(self.pmf_array, np.array([poisson.pmf(x, self.par)]))
            if sum(self.pmf_array) > 1 - epsilon:
                self.maxx = x
                break
        else:
            print(' max_size = ' + str(max_size) + ' not long enough')

    def __call__(self, i):
        return self.pmf(i)

    def pmf(self, i):
        if i > self.maxx:
            return 0
        else:
            return self.pmf_array[i]