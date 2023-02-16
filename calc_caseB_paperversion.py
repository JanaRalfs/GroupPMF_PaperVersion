#%% Calculate the pmf for n outstanding orders at the retailer, Case B
import math

import numpy as np
import scipy.stats
import scipy.special
import pmf_caseB_paperversion as pmfB
import time
import csv

# --- COMPUTATIONS OUTSTANDING ORDERS CASE B ---

def calc(R0, Q0, TN, L0, laX, La, QNd, max_n, epsilon):
    laY = La - laX
    # compute max_size dependent on longest separate interval, could be more restrictive,
    # but we can be sure that we are exact enough
    z = pmfB.poisson_pmf(La, max(L0-TN, TN))
    max_size_B = np.size(z.pmf_array)
    print(max_n)

    Pr_B = pmfB.joint_pmf_caseB(R0, Q0, TN, L0, laX, laY, max_size_B)

    t_start = time.time()
    ip_min = R0 + 1
    ip_max = R0 + Q0

    # initiate matrices to save probability of name1 / name2
    Pr_B_name1 = np.array([np.arange(0,max_n , 1, int), np.zeros(max_n)])
    Pr_B_name2 = np.array([np.arange(0,max_n , 1, int), np.zeros(max_n)])
    Pr_name1 = np.zeros((ip_max - ip_min + 1, 2, max_n ))
    Pr_name2 = np.zeros((ip_max - ip_min + 1, 2, max_n ))
    for i in range(ip_max-ip_min+1):
        Pr_name1[i, 0, :] = range(0,max_n)
        Pr_name2[i, 0, :] = range(0,max_n)
    n_min = 0

    # matrix beta-binomial
    # flexible bound?
    beta_ub = 75
    x_ub = beta_ub
    alpha_ub = 75
    k_ub = alpha_ub

    Pr_BB = np.zeros((x_ub + 1, k_ub + 1, alpha_ub + 1, beta_ub + 1))
    for beta in range(beta_ub + 1):
        for x in range(beta + 1):
            for alpha in range(1, alpha_ub + 1):
                for k in range(alpha + 1):
                    Pr_BB[x, k, alpha, beta] = (scipy.special.comb(beta, x)
                                                * scipy.special.beta(x + k, beta - x + alpha - k + 1)
                                                / scipy.special.beta(k, alpha - k + 1))

    if R0 >= -1:
        for ip in range(ip_min, ip_max + 1):
            for n in range(1, max_n + 1):
                print(n)
                for g1 in range(0, int(Pr_B.g1g2g3.up_g1 + 0.5) + 1):
                    for g2 in range(0, int(Pr_B.g1g2g3.up_g2[g1] + 0.5) + 1):
                        for g3 in range(0, int(Pr_B.g1g2g3.up_g3[g1, g2] + 0.5) + 1):
                            for a1 in range(0, g1 + 1):
                                for a2 in range(0, g2 + 1):
                                    for a3 in range(0, g3 + 1):
                                        b1 = g1 - a1
                                        b2 = g2 - a2
                                        b3 = g3 - a3
                                        b0 = max(0, -(ip - g2 - g3))
                                        if b0 <= g2:
                                            for b0N in range(0, b0 + 1):
                                                # is nthunit on t-b-s?
                                                if n > a1 and b0N < n - a1:
                                                    proba = (Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                             * Pr_B.b0N_1(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                    Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                                # if not, is it on q-b-s?
                                                else:
                                                    q = n - (QNd - (b0N + a1 - n) % QNd - 1)
                                                    #for a2 in range(g2 + 1):
                                                    if 0 < q <= a1:
                                                        for Xb1 in range(0,b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                                         * Pr_B.b0N_1(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip-R0-1,1, n - 1] += proba
                                                    elif q > a1:
                                                        for x in range(0, b0 - b0N + 1):
                                                            # Xb2 = g2 - a2 - (b0-b0N) + x
                                                            # r = a1 + a2 + a3 - q + 1 + b3 + Xb2
                                                            r = g3 + g2 + a1 - q + 1 - (b0 - b0N) + x
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[x, a1 + b0N - q + 1, b0N, b0 - b0N]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_1(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                        elif b0 > g2:
                                            for b0N in range(0, b0 + 1):
                                                # is nthunit on t-b-s?
                                                if n > a1 and b0N < n - a1:
                                                    proba = (Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                             * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                    Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                                # if not, is it on q-b-s?
                                                else:
                                                    q = n - (QNd - (b0N + a1 - n) % QNd - 1)
                                                    if 0 < q <= a1:
                                                        for Xb1 in range(0,b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                                         * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip-R0-1,1, n - 1] += proba
                                                    elif a1 < q <= a1 + a2:
                                                        b2 = g2 - a2
                                                        for Xb2 in range(0, b2 + 1):
                                                            r = g3 + a2 + a1 - q + 1 + Xb2
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb2, a1 + a2 - q + 1, a2, b2]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif q > a1 + a2:
                                                        proba = (Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1, g2, g3)
                                                                 * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                        Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                if (Pr_name1[ip - R0 - 1, 1, n - 1] + Pr_name2[ip - R0 - 1, 1, n - 1]
                        > sum(sum(sum(Pr_B.g1g2g3.pmf_array))) * (1 - epsilon) and n_min <= n):
                    if ip == ip_min:
                        n_min = n
                    break

    elif R0 < -1:
        for ip in range(ip_min, ip_max + 1):
            for n in range(1, max_n + 1):
                print(n)
                for g1 in range(0, int(Pr_B.g1g2g3.up_g1 + 0.5) + 1):
                    for g2 in range(0, int(Pr_B.g1g2g3.up_g2[g1] + 0.5) + 1):
                        for g3 in range(0, int(Pr_B.g1g2g3.up_g3[g1, g2] + 0.5) + 1):
                            for a1 in range(0, g1 + 1):
                                for a2 in range(0,g2 + 1):
                                    for a3 in range(0,g3 + 1):
                                        b1 = g1 - a1
                                        b2 = g2 - a2
                                        b3 = g3 - a3
                                        b0 = max(0, -(ip - g2 - g3))
                                        if b0 <= g2:
                                            for b0N in range(0, b0 + 1):
                                                # is nthunit on t-b-s?
                                                if n > a1 and b0N < n - a1:
                                                    proba = (Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                             * Pr_B.b0N_1(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                    Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                                # if not, is it on q-b-s?
                                                else:
                                                    q = n - (QNd - (b0N + a1 - n) % QNd - 1)
                                                    if 0 < q <= a1:
                                                        for Xb1 in range(0, b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                                         * Pr_B.b0N_1(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif q > a1:
                                                        for x in range(0, b0 - b0N + 1):
                                                            # Xb2 = g2 - a2 - (b0-b0N) + x
                                                            # r = a1 + a2 + a3 - q + 1 + b3 + Xb2
                                                            r = g3 + g2 + a1 - q + 1 - (b0 - b0N) + x
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[x, a1 + b0N - q + 1, b0N, b0 - b0N]
                                                                         * Pr_B.a1(a1,g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1,g2,g3)
                                                                         * Pr_B.b0N_1(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                        elif g2 < b0 <= g2 + g3: #ip >= 0
                                            for b0N in range(a2, b0 + 1):
                                                # is nthunit on t-b-s?
                                                if n > a1 and b0N < n - a1:
                                                    proba = (Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3) * Pr_B.g1g2g3(g1, g2, g3)
                                                             * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                    Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                                # if not, is it on q-b-s?
                                                else:
                                                    q = n - (QNd - (b0N + a1 - n) % QNd - 1)
                                                    if 0 < q <= a1:
                                                        for Xb1 in range(0, b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif a1 < q <= a1 + a2:
                                                        for Xb2 in range(0, b2 + 1):
                                                            r = g3 + a2 + a1 - q + 1 + Xb2
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb2, a1 + a2 - q + 1, a2, b2]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif q > a1 + a2:
                                                        for x in range(0, b0 - b0N - b2 + 1):
                                                            # Xb3 = g2 - a2 + g3 - a3 - (b0-b0N) + x
                                                            # r = a1 + a2 + a3 - q + 1 + Xb3
                                                            r = g3 + g2 + a1 - q + 1 - (b0 - b0N) + x
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[x, a1 + b0N - q + 1, b0N - a2, b0 - b0N - b2]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_2(ip - R0 - 1, b0N, a2, b2, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                        elif b0 > g2 + g3: #ip < 0
                                            for b0N in range(a2 + a3, b0 + 1):
                                                a4 = b0N - a3 - a2
                                                b4 = -ip - a4
                                                # is nthunit on t-b-s?
                                                if n > a1 and b0N < n - a1:
                                                    proba = (Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                             * Pr_B.g1g2g3(g1, g2, g3)
                                                             * Pr_B.b0N_3(ip - R0 - 1, b0N, a2, a3))
                                                    Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                                # if not, is it on q-b-s?
                                                else:
                                                    q = n - (QNd - (b0N + a1 - n) % QNd - 1)
                                                    if 0 < q <= a1:
                                                        for Xb1 in range(0, b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_3(ip - R0 - 1, b0N, a2, a3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif a1 < q <= a1 + a2:
                                                        for Xb2 in range(0, b2 + 1):
                                                            r = g3 + a2 + a1 - q + 1 + Xb2
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb2, a1 + a2 - q + 1, a2, b2]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_3(ip - R0 - 1, b0N, a2, a3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif a1 + a2 < q <= a1 + a2 + a3:
                                                        for Xb3 in range(0, b3 + 1):
                                                            r = a1 + a2 + a3 - q + 1 + Xb3
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb3, a1 + a2 + a3 - q + 1, a3, b3]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_3(ip - R0 - 1, b0N, a2, a3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                    elif q > a1 + a2 + a3:
                                                        for Xb4 in range(0, -ip + 1):
                                                            r = a1 + a2 + a3 - q + 1 - Xb4
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g3:
                                                                proba = (Pr_BB[Xb4, q - a1 - a2 - a3, a4, b4]
                                                                         * Pr_B.a1(a1, g1) * Pr_B.a2(a2,g2) * Pr_B.a3(a3,g3)
                                                                         * Pr_B.g1g2g3(g1, g2, g3)
                                                                         * Pr_B.b0N_3(ip - R0 - 1, b0N, a2, a3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                if (Pr_name1[ip-R0-1,1, n - 1] + Pr_name2[ip-R0-1,1, n - 1] > sum(sum(sum(Pr_B.g1g2g3.pmf_array))) * (1- epsilon)
                    and n_min <= n):
                    if ip == ip_min:
                        n_min = n
                    break

    #average cdf of name1/nam2
    Pr_name1[:, 1, :] = Pr_name1[:, 1, :] / Q0
    Pr_name2[:, 1, :] = Pr_name2[:, 1, :] / Q0
    for i in range(Q0):
        Pr_B_name1[1, :] += Pr_name1[i, 1, :]
        Pr_B_name2[1, :] += Pr_name2[i, 1, :]

    t_analysis = time.time() - t_start

    return Pr_B_name1[1,:], Pr_B_name2[1,:], Pr_B.t_joint_pmf, t_analysis
