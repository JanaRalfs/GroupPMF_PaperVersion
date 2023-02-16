#%% Calculate the pmf for n outstanding orders at the retailer, Case A
import numpy as np
import scipy.stats
import scipy.special
import pmf_caseA_paperversion as pmfA
import time

# --- COMPUTATIONS OUTSTANDING ORDERS CASE A ---

def calc(R0, Q0, TN, L0, laX, La, QNd, max_n, epsilon):

    t_start = time.time()

    #grenze upper bound??
    z = pmfA.poisson_pmf(La, max(TN-L0,L0))
    max_size_A = z.maxa

    #pmfs for case A
    Pr_A = pmfA.joint_pmf_caseA(R0,Q0,TN,L0,laX,La,max_size_A)

    #bounds inventory position
    ip_min = R0 + 1
    ip_max = R0 + Q0

    # initiate matrices to save "cdf" of name1 / name2
    Pr_A_name1 = np.array([np.arange(0, max_n, 1, int), np.zeros(max_n)])
    Pr_A_name2 = np.array([np.arange(0, max_n, 1, int), np.zeros(max_n)])
    Pr_name1 = np.zeros((ip_max - ip_min + 1, 2,  max_n))
    Pr_name2 = np.zeros((ip_max - ip_min + 1, 2, max_n))
    for i in range(ip_max - ip_min + 1):
        Pr_name1[i, 0, :] = range(0, max_n)
        Pr_name2[i, 0, :] = range(0, max_n)
    n_min = 0

    #matrix beta-binomial
    #flexible bound?
    beta_ub = 75
    x_ub = beta_ub
    alpha_ub = 75
    k_ub = alpha_ub

    Pr_BB= np.zeros((x_ub + 1, k_ub +1, alpha_ub + 1, beta_ub + 1))
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
                for a3 in range(0, Pr_A.maxa3 + 1):
                    for b3 in range(0, Pr_A.maxb3 + 1):
                        g3 = a3 + b3
                        for a1 in range(0, Pr_A.maxa1 + 1):
                            for a2 in range(0, Pr_A.maxa2 + 1):
                                b0 = max(0, -(ip - a3 - b3))
                                for b0N in range(0, b0 + 1):
                                    # is nthunit on t-b-s?
                                    if n > a1 + a2 and b0N < n - a1 - a2:
                                        proba = (Pr_A.a1(a1) * Pr_A.a2(a2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                 * Pr_A.b0N_1(ip - R0 - 1, b0N, a3, b3))
                                        Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                    # if not, is it on q-b-s?
                                    else:
                                        q = n - (QNd - (b0N + a1 + a2 - n) % QNd - 1)
                                        if 0 < q <= a1:
                                            for b2 in range(0, Pr_A.maxb2 + 1):
                                                g2 = a2 + b2
                                                for b1 in range(0, Pr_A.maxb1 + 1):
                                                    for Xb1 in range(b1 + 1):
                                                        r = g3 + g2 + a1 - q + 1 + Xb1
                                                        phi = R0 + Q0 - (ip - r) % Q0
                                                        if r - phi <= g2 + g3:
                                                            proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                     * Pr_A.a1(a1) * Pr_A.b1(b1)
                                                                     * Pr_A.a2b2(a2, b2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                     * Pr_A.b0N_1(ip - R0 - 1, b0N, a3, b3))
                                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                        elif q > a1:
                                            proba = (Pr_A.a1(a1) * Pr_A.a2(a2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                     * Pr_A.b0N_1(ip - R0 - 1, b0N, a3, b3))
                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                    if ((Pr_name1[ip - R0 - 1, 1, n - 1] + Pr_name2[ip - R0 - 1, 1, n - 1] > sum(sum(Pr_A.a2b2.pmf_array)) * (1 - epsilon)
                        and n_min <= n)):
                        #or (Pr_name1[ip - R0 - 1, 1, n - 2] == Pr_name1[ip - R0 - 1, 1, n - 1] and Pr_name1[ip - R0 - 1, 1, n - 1] >= 1-epsilon)):
                        if ip == ip_min:
                            n_min = n
                        break

    elif R0 < -1:
        for ip in range(ip_min, ip_max + 1):
            for n in range(1, max_n + 1):
                print(n)
                for a3 in range(0, Pr_A.maxa3 + 1):
                    for b3 in range(0, Pr_A.maxb3 + 1):
                        g3 = a3 + b3
                        for a1 in range(0, Pr_A.maxa1 + 1):
                            for a2 in range(0, Pr_A.maxa2 + 1):
                                b0 = max(0,-(ip - g3))
                                if b0 <= g3:
                                    for b0N in range(0,b0 + 1):
                                        # is nthunit on t-b-s?
                                        if n > a1 + a2 and b0N < n - a1 - a2:
                                            proba = (Pr_A.a1(a1) * Pr_A.a2(a2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                     * Pr_A.b0N_1(ip - R0 - 1,b0N, a3, b3))
                                            Pr_name1[ip-R0-1,1, n - 1] += proba
                                        # if not, is it on q-b-s?
                                        else:
                                            q = n - (QNd - (b0N + a1 + a2 - n) % QNd - 1)
                                            for b2 in range(0, Pr_A.maxb2 + 1):
                                                g2 = a2 + b2
                                                if 0 < q <= a1:
                                                    for b1 in range(0, Pr_A.maxb1 + 1):
                                                        for Xb1 in range(b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g2 + g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_A.a1(a1) * Pr_A.b1(b1)
                                                                         * Pr_A.a2b2(a2, b2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                         * Pr_A.b0N_1(ip - R0 - 1, b0N, a3, b3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                elif a1 < q <= a1 + a2:
                                                    for Xb2 in range(b2 + 1):
                                                        r = g3 + a1 + a2 - q + 1 + Xb2
                                                        phi = R0 + Q0 - (ip - r) % Q0
                                                        if r - phi <= g2 + g3:
                                                            proba = (Pr_BB[Xb2, a1 + a2 - q + 1, a2, b2]
                                                                     * Pr_A.a1(a1) * Pr_A.a2b2(a2, b2)
                                                                     * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                     * Pr_A.b0N_1(ip - R0 - 1, b0N, a3, b3))
                                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                elif q > a1 + a2:
                                                    for x in range(0, b0 - b0N + 1):
                                                        # Xb3 = g3 - a3 - (b0-b0N) + x
                                                        # r = a1 + a2 + a3 - q + 1 + Xb3
                                                        r = a1 + a2 + g3 - q + 1 - (b0 - b0N) + x
                                                        phi = R0 + Q0 - (ip - r) % Q0
                                                        if r - phi <= g2 + g3:
                                                            proba = (Pr_BB[x, a1 + a2 + b0N - q + 1, b0N, b0 - b0N]
                                                                     * Pr_A.a1(a1) * Pr_A.a2b2(a2, b2)
                                                                     * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                     * Pr_A.b0N_1(ip - R0 - 1, b0N, a3, b3))
                                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                elif b0 > g3:   #ip < 0
                                    for b0N in range(0, b0 + 1):
                                        #for a3 in range(0, g3 + 1):
                                            #b3 = g3 - a3
                                        a4 = b0N - a3
                                        b4 = -ip - a4
                                        # is nthunit on t-b-s?
                                        if n > a1 + a2 and b0N < n - a1 - a2:
                                            proba = (Pr_A.a1(a1) * Pr_A.a2(a2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                     * Pr_A.b0N_2(ip - R0 - 1, b0N, a3))
                                            Pr_name1[ip - R0 - 1, 1, n - 1] += proba
                                        # if not, is it on q-b-s?
                                        else:
                                            q = n - (QNd - (b0N + a1 + a2 - n) % QNd - 1)
                                            for b2 in range(0, Pr_A.maxb2 + 1):
                                                g2 = a2 + b2
                                                if 0 < q <= a1:
                                                    for b1 in range(0, Pr_A.maxb1 + 1):
                                                        for Xb1 in range(b1 + 1):
                                                            r = g3 + g2 + a1 - q + 1 + Xb1
                                                            phi = R0 + Q0 - (ip - r) % Q0
                                                            if r - phi <= g2 + g3:
                                                                proba = (Pr_BB[Xb1, a1 - q + 1, a1, b1]
                                                                         * Pr_A.a1(a1) * Pr_A.b1(b1)
                                                                         * Pr_A.a2b2(a2, b2) * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                         * Pr_A.b0N_2(ip - R0 - 1, b0N, a3))
                                                                Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                elif a1 < q <= a1 + a2:
                                                    for Xb2 in range(b2 + 1):
                                                        r = g3 + a1 + a2 - q + 1 + Xb2
                                                        phi = R0 + Q0 - (ip - r) % Q0
                                                        if r - phi <= g2 + g3:
                                                            proba = (Pr_BB[Xb2, a1 + a2 - q + 1, a2, b2]
                                                                     * Pr_A.a1(a1) * Pr_A.a2b2(a2, b2)
                                                                     * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                     * Pr_A.b0N_2(ip - R0 - 1, b0N, a3))
                                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                elif a1 + a2 < q <= a1 + a2 + a3:
                                                    for Xb3 in range(0, b3 + 1):
                                                        r = a1 + a2 + a3 - q + 1 + Xb3
                                                        phi = R0 + Q0 - (ip - r) % Q0
                                                        if r - phi <= g2 + g3:
                                                            proba = (Pr_BB[Xb3, a1 + a2 + a3 - q + 1, a3, b3]
                                                                     * Pr_A.a1(a1) * Pr_A.a2b2(a2, b2)
                                                                     * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                     * Pr_A.b0N_2(ip - R0 - 1, b0N, a3))
                                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                                                elif q > a1 + a2 + a3:
                                                    for Xb4 in range(0,-ip + 1):
                                                        r = a1 + a2 + a3 - q + 1 - Xb4
                                                        phi = R0 + Q0 - (ip - r) % Q0
                                                        if r - phi <= g2 + g3:
                                                            proba = (Pr_BB[Xb4, q - a1 - a2 - a3, a4, b4]
                                                                     * Pr_A.a1(a1) * Pr_A.a2b2(a2, b2)
                                                                     * Pr_A.a3(a3) * Pr_A.b3(b3)
                                                                     * Pr_A.b0N_2(ip - R0 - 1, b0N, a3))
                                                            Pr_name2[ip - R0 - 1, 1, n - 1] += proba
                    if (Pr_name1[ip - R0 - 1, 1, n - 1] + Pr_name2[ip - R0 - 1, 1, n - 1] > sum(sum(Pr_A.a2b2.pmf_array)) * (1 - epsilon)
                            and n_min <= n):
                        if ip == ip_min:
                            n_min = n
                        break


    #average cdf of name1/nam2
    Pr_name1[:, 1, :] = Pr_name1[:, 1, :] / Q0
    Pr_name2[:, 1, :] = Pr_name2[:, 1, :] / Q0
    for i in range(Q0):
        Pr_A_name1[1, :] += Pr_name1[i, 1, :]
        Pr_A_name2[1, :] += Pr_name2[i, 1, :]

    t_analysis = time.time() - t_start

    return Pr_A_name1[1,:], Pr_A_name2[1,:], Pr_A.t_joint_pmf, t_analysis
