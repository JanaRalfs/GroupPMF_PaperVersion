
import numpy as np
import calc_caseB_paperversion as calcB
import calc_caseA_paperversion as calcA
import csv
import pandas as pd
import multiprocessing
import concurrent.futures
num_processes =   2#multiprocessing.cpu_count()



def run(ex, R_0, Q_0, T_M, L_0, la_m, La, Q_M, epsilon):

    #max number of outstanding orders
    max_n = 75

    #obtain pmf of outstanding orders
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
        # make sure that we have the same length of probability vectors from case A and case B
        n_A = Pr_A_name1.argmax()
        n_B = Pr_B_name1.argmax()
        if n_A > n_B:
            Pr_B_name1[n_B + 1:n_A + 1] = Pr_B_name1[n_B]
            Pr_B_name2[n_B + 1:n_A + 1] = Pr_B_name2[n_B]
        elif n_B > n_A:
            Pr_A_name1[n_A + 1:n_B + 1] = Pr_A_name1[n_A]
            Pr_A_name2[n_A + 1:n_B + 1] = Pr_A_name2[n_A]
        # compute final
        Pr_name1 = Pr_A_name1[:] * probA + Pr_B_name1[:] * probB
        Pr_name2 = Pr_A_name2[:] * probA + Pr_B_name2[:] * probB

    #creatae vector with input and results
    results = np.concatenate((ex, R_0, Q_0, T_M, L_0, la_m, La, Q_M, Pr_name1, Pr_name2, t_joint_pmf_A,
                              t_analysis_A, t_joint_pmf_B, t_analysis_B), axis=None)
    #save results as .csv
    with open('pmf_14nov_old.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)
        f.close()



if __name__ == '__main__':

    # import data from excel sheet
    data = pd.read_excel("Input_Data_PaperVersion.xlsx")

    # Create header and file
    str_a = 'Pr_name1_'
    str_b = 'Pr_name2_'
    my_list_A = []
    my_list_B = []
    for i in range(75):
        my_list_A = my_list_A + [str_a + str(i)]
        my_list_B = my_list_B + [str_b + str(i)]
    # create header
    header = np.concatenate(('ex', 'R_0', 'Q_0', 'T_M', 'L_0', 'la_m', 'La', 'Q_M', my_list_A, my_list_B,
                             't_joint_pmf_A', 't_analysis_A', 't_joint_pmf_B', 't_analysis_B'), axis=None)

    # save header in .csv
    with open('pmf_14nov_old.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()

    # multi processing:
    # create dataframe
    df = pd.DataFrame()
    # max 61 processes in one pycharm
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(61,num_processes)) as executor:
        com = executor.map(run, data['ex'], data['R_0'], data['Q_0'], data['T_M'], data['L_0'],
                           data['la_m'], data['La'], data['Q_M'], data['epsilon'])
