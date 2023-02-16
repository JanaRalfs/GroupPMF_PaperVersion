# %% main file which gets the data from the excel file "Input_Data"
# %% saves the pmf of the outstanding orders in a .csv file

import time
import numpy as np
import calc_caseB_paperversion as calcB
import calc_caseA_paperversion as calcA
import csv
import pandas as pd

# --- COMPUTATIONS OUTSTANDING ORDERS CASE B ---

# input from excel file
data = pd.read_excel("Input_Data_PaperVersion"
                     ".xlsx")
no_instance, no_input = data.shape
# to influence the accuracy of the number of outstanding orders
#epsilon = 0.000001
t = time.time()

#Create header and file
str_a = 'Pr_name1_'
str_b = 'Pr_name2_'
my_list_A = []
my_list_B = []
for i in range(75):
    my_list_A = my_list_A + [str_a + str(i)]
    my_list_B = my_list_B + [str_b + str(i)]
# creatae vector with input and results
header = np.concatenate(('ex', 'R_0', 'Q_0', 'T_M', 'L_0', 'la_m', 'La', 'Q_M', my_list_A, my_list_B, 't_joint_pmf_A', 't_analysis_A', 't_joint_pmf_B', 't_analysis_B'), axis=None)
# save results as .csv
name = 'new10.csv'
with open(name, 'a+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()


# loop through all examples
for instance in range(0,1):
    ex = data.at[instance, 'ex']
    R_0 = int(data.at[instance, 'R_0'])
    Q_0 = int(data.at[instance, 'Q_0'])
    T_M = data.at[instance, 'T_M']
    L_0 = int(data.at[instance, 'L_0'])
    la_m = data.at[instance, 'la_m']
    La = data.at[instance, 'La']
    Q_M = int(data.at[instance, 'Q_M'])
    epsilon = data.at[instance, 'epsilon']

    # max number of outstanding orders
    #z = pmfA.poisson_pmf(La, L_0 + T_M)
    max_n = 75

    # obtain pmd of outstanding orders
    if T_M <= L_0:
        Pr_name1, Pr_name2, t_joint_pmf_B, t_analysis_B = calcB.calc(R_0, Q_0, T_M, L_0, la_m, La, Q_M, max_n, epsilon)
        t_joint_pmf_A = 0
        t_analysis_A = 0
    else:
        probA = 1 - L_0 / T_M
        probB = L_0 / T_M
        Pr_A_name1, Pr_A_name2, t_joint_pmf_A, t_analysis_A = calcA.calc(R_0, Q_0, T_M, L_0, la_m, La, Q_M, max_n, epsilon)
        # input T_M = L_0, to reflect case B
        Pr_B_name1, Pr_B_name2, t_joint_pmf_B, t_analysis_B = calcB.calc(R_0, Q_0, L_0, L_0, la_m, La, Q_M, max_n, epsilon)
        #make sure that we have the same length of probability vectors from case A and case B
        n_A = Pr_A_name1.argmax()
        n_B = Pr_B_name1.argmax()
        if n_A > n_B:
            Pr_B_name1[n_B + 1:n_A + 1] = Pr_B_name1[n_B]
            Pr_B_name2[n_B + 1:n_A + 1] = Pr_B_name2[n_B]
        elif n_B > n_A:
            Pr_A_name1[n_A + 1:n_B + 1] = Pr_A_name1[n_A]
            Pr_A_name2[n_A + 1:n_B + 1] = Pr_A_name2[n_A]
        Pr_name1 = Pr_A_name1 * probA + Pr_B_name1 * probB
        Pr_name2 = Pr_A_name2 * probA + Pr_B_name2 * probB
        Pr_name1 = Pr_A_name1[:] * probA + Pr_B_name1[:] * probB
        Pr_name2 = Pr_A_name2[:] * probA + Pr_B_name2[:] * probB

    # creatae vector with input and results
    results = np.concatenate((ex, R_0, Q_0, T_M, L_0, la_m, La, Q_M, Pr_name1, Pr_name2, t_joint_pmf_A, t_analysis_A, t_joint_pmf_B, t_analysis_B), axis=None)
    # save results as .csv
    with open(name, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)
        f.close()
