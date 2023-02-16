# %% Compute PMF for Case A, y>=0
import math
import time

import numpy as np
import scipy.special
from scipy.stats import poisson
from scipy.stats import binom

# %% Joint pmf for a2 and b2 in case A

class a2b2_caseA:
    def __init__(self, TN, L0, laX, la, max_size, epsilon=0.000001):
        self.b = TN - L0
        self.laX = laX
        self.la = la
        self.maxg2 = max_size
        self.p = laX / la
        self.pmf_array = np.zeros([max_size + 1, max_size + 1])
        self.pmf_array[0, 0] = (1 / (self.la * self.b)) * (1 - poisson.cdf(0, self.b * self.la))
        g2_p_sum = 0
        for g2 in range(1, max_size + 1):
            g2_p = (1 / (self.la * self.b)) * (1 - poisson.cdf(g2, self.b * self.la))
            g2_p_sum += g2_p
            for a2 in range(0, g2 + 1):
                self.pmf_array[a2, g2 - a2] = binom.pmf(a2, g2, self.p) * g2_p
            if g2_p_sum > 1 - epsilon:
                self.maxg2 = g2 - 1
                break

    def __call__(self, i, j):
        return self.pmf(i, j)

    def pmf(self, i, j):
        if i + j > self.maxg2:
            return 0
        else:
            return self.pmf_array[i, j]


class a2_caseA:
    def __init__(self, TN, L0, laX, la, max_size):
        self.b = TN - L0
        self.laX = laX
        self.la = la
        self.maxg = max_size
        self.p = laX / la
        self.pmf_array = np.zeros((max_size +1))
        self.a2b2 = a2b2_caseA(TN, L0, laX, la, max_size)

        for b2 in range(0, max_size+1):
            self.pmf_array[:] += self.a2b2.pmf_array[:,b2]

    def __call__(self, i):
        return self.pmf(i)

    def pmf(self, i):
        if i > self.maxg:
            return 0
        else:
            return self.pmf_array[i]

class b0N_a3b3_1_caseA: #case B0 <= g3
    def __init__(self, R, Q, TN, L0, laX, la, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.la = la
        self.p = laX / la
        self.maxg3 = max_size
        self.maxb0N = max_size
        self.pmf_array = np.zeros((self.Q, 2*max_size + 1, max_size + 1, max_size + 1))

        tstart = time.time()
        for a3 in range(0, max_size+1):
            for b3 in range(0, max_size + 1):
                for ip in range(self.R+1,self.R+self.Q+1):
                    b0 = max(0, -(ip - a3 - b3))
                    if b0 <= a3 + b3:
                        for b0N in range(0,b0 + 1):
                            #first is faster and hypergeom.pmf(0,0,0,0) gives nan, even if it is 1.0
                            self.pmf_array[ip-self.R-1,b0N,a3,b3] = math.comb(a3, b0N) * math.comb(a3 + b3 - a3, b0 - b0N) / math.comb(a3 + b3 , b0)
                            #self.pmf_array[ip-self.R-1,b0N,a3,b3] = scipy.stats.hypergeom.pmf(b0N,a3+b3,a3,b0)

    def __call__(self, i, j, k, l):
        return self.pmf(i, j, k, l)

    def pmf(self, i, j, k, l):
        if i > self.Q - 1 | j > self.maxb0N | k > self.maxa3 | l > self.maxb3:
            return 0
        else:
            return self.pmf_array[i, j, k, l]


class b0N_g3_1_caseA: #case B0 <= g3
    def __init__(self, R, Q, TN, L0, laX, la, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.la = la
        self.p = laX / la
        self.maxg3 = max_size
        self.maxb0N = max_size
        self.pmf_array = np.zeros((self.Q, max_size + 1, max_size + 1))

        for g3 in range(0, max_size+1):
            for ip in range(self.R+1,self.R+self.Q+1):
                B0 = max(0, -(ip - g3))
                if B0 <= g3:
                    for b0N in range(0,B0 + 1):
                        self.pmf_array[ip-self.R-1,b0N,g3] = binom.pmf(b0N, B0, self.p)

    def __call__(self, i, j, k):
        return self.pmf(i, j, k)

    def pmf(self, i, j, k):
        if i > self.Q - 1 | j > self.maxb0N | j > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j, k]

class a3_g3_b0N_1_caseA: #case B0 <= g3
    def __init__(self, R, Q, TN, L0, laX, la, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.la = la
        self.p = laX / la
        self.maxg3 = max_size
        self.maxb0N = max_size
        self.maxa3 = max_size
        self.pmf_array = np.zeros((self.Q, max_size + 1, max_size + 1, max_size + 1))
        #self.pmf_array2 = np.zeros((self.Q, max_size + 1, max_size + 1))

        for g3 in range(0, max_size+1):
            for ip in range(self.R+1,self.R+self.Q+1):
                b0 = max(0, -(ip - g3))
                if b0 <= g3:
                    for b0N in range(b0 + 1):
                        for a3 in range(b0N, g3 + 1):
                            self.pmf_array[ip-self.R-1,a3,g3,b0N] = binom.pmf(a3-b0N, g3-b0, self.p)
        '''for g3 in range(0, max_size+1):
            for ip in range(self.R+1,self.R+self.Q+1):
                b0 = max(0, -(ip - g3))
                if b0 <= g3:
                    for a3q in range(g3 - b0 + 1):
                        self.pmf_array2[ip-self.R-1,a3q,g3] = binom.pmf(a3q, g3-b0, self.p)
                        hu = binom.pmf(a3q, g3-b0, self.p)
                        hi = 3'''


    def __call__(self, i, j, k, l):
        return self.pmf(i, j, k, l)

    def pmf(self, i, j, k, l):
        if i > self.Q - 1 | j > self.maxa3 | k > self.maxg3 | l > self.maxb0N:
            return 0
        else:
            return self.pmf_array[i, j, k, l]

class a3_g3_2_caseA: #case B0 > g3
    def __init__(self, R, Q, TN, L0, laX, la, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.la = la
        self.p = laX / la
        self.maxg3 = max_size
        self.maxa3b = max_size
        self.pmf_array = np.zeros((self.Q, max_size + 1, max_size + 1))

        for g3 in range(0, max_size+1):
            for ip in range(self.R+1,self.R+self.Q+1):
                B0 = max(0, -(ip - g3))
                if B0 > g3:
                    for a3 in range(0,g3 + 1):
                        self.pmf_array[ip-self.R-1,a3,g3] = binom.pmf(a3,g3, self.p)

    def __call__(self, i, j, k):
        return self.pmf(i, j, k)

    def pmf(self, i, j, k):
        if i > self.Q - 1 | j > self.maxa3b | j > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j, k]


class b0N_a3_2_caseA: #case B0 > g3
    def __init__(self, R, Q, TN, L0, laX, la, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.la = la
        self.p = laX / la
        self.maxg3 = max_size
        self.maxa4b = max_size
        self.pmf_array = np.zeros((self.Q, 2*max_size + self.Q + 1, max_size + 1))

        for ip in range(self.R+1,self.R+self.Q+1):
            if ip < 0:
                for a3 in range(0, max_size + 1):
                    for a4 in range(0,-ip + 1):
                        self.pmf_array[ip-self.R-1, a3 + a4, a3] = binom.pmf(a4,-ip, self.p)

    def __call__(self, i, j, k):
        return self.pmf(i, j, k)

    def pmf(self, i, j, k):
        if i > self.Q - 1 | j > self.maxa4b | j > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j, k]

class a4_2_caseA: #case B0 > g3
    def __init__(self, R, Q, TN, L0, laX, la, max_size):
        self.R = R
        self.Q = Q
        self.laX = laX
        self.la = la
        self.p = laX / la
        self.maxg3 = max_size
        self.maxa4b = max_size
        self.pmf_array = np.zeros((self.Q, self.Q + 1))

        for ip in range(self.R+1,self.R+self.Q+1):
            if ip < 0:
                for a4 in range(0,-ip + 1): #
                    self.pmf_array[ip-self.R-1, a4] = binom.pmf(a4,-ip, self.p)

    def __call__(self, i, j, k):
        return self.pmf(i, j, k)

    def pmf(self, i, j, k):
        if i > self.Q - 1 | j > self.maxa4b | j > self.maxg3:
            return 0
        else:
            return self.pmf_array[i, j, k]


# %% PMF for Poisson arrivals with intensity lambda in period L
class poisson_pmf:
    def __init__(self, la, L, max_size = 500, epsilon = 0.0000001):
        self.par = la * L
        self.maxa = max_size
        self.pmf_array = np.array([poisson.pmf(0, self.par)])

        for a in range(1, max_size + 1):
            self.pmf_array = np.append(self.pmf_array, np.array([poisson.pmf(a, self.par)]))
            if sum(self.pmf_array) > 1 - epsilon:
                self.maxa = a
                break
        else:
            print(' max_size = ' + str(max_size) + ' not long enough')

    def __call__(self, i):
        return self.pmf(i)

    def pmf(self, i):
        if i > self.maxa:
            return 0
        else:
            return self.pmf_array[i]

# %%
class joint_pmf_caseA:
    # Provides the joint pmf of B0,a1,a2 and b1
    def __init__(self, R0, Q0, TN, L0, laX, la, max_size_A):
        #save the time
        t_start = time.time()
        #general distributions
        self.a1 = poisson_pmf(laX, L0, max_size_A)
        self.b1 = poisson_pmf(la-laX, L0, max_size_A)
        self.a2b2 = a2b2_caseA(TN, L0, laX, la, max_size_A)
        self.a2 = a2_caseA(TN, L0, laX, la, max_size_A)
        #self.g3 = poisson_pmf(la, L0, max_size_A)
        self.a3 = poisson_pmf(laX, L0, max_size_A)
        self.b3 = poisson_pmf(la - laX, L0, max_size_A)
        # case B0 <= g3
        #self.b0N_1 = b0N_g3_1_caseA(R0, Q0, TN, L0, laX, la, max_size_A)
        #self.a3_1 = a3_g3_b0N_1_caseA(R0, Q0, TN, L0, laX, la, max_size_A)
        self.b0N_1 = b0N_a3b3_1_caseA(R0, Q0, TN, L0, laX, la, max_size_A)
        # case B0 > g3
        #self.a3_2 = a3_g3_2_caseA(R0, Q0, TN, L0, laX, la, max_size_A)
        self.b0N_2 = b0N_a3_2_caseA(R0, Q0, TN, L0, laX, la, max_size_A)
        #self.a4_2 = a4_2_caseA(R0, Q0, TN, L0, laX, la, max_size_A)

        self.maxa2 = self.a2b2.maxg2
        self.maxb2 = self.a2b2.maxg2
        self.maxa1 = self.a1.maxa
        self.maxb1 = self.b1.maxa
        self.maxa3 = self.a3.maxa
        self.maxb3 = self.b3.maxa
        #self.maxg3 = self.g3.maxa
        #self.maxa3 = self.a3_1.maxa3

        self.t_joint_pmf = time.time() - t_start



