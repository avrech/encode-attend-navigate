import csv
import numpy as np
import pickle
from tabulate import tabulate
import os
DIR = os.path.dirname(os.path.abspath(__file__))
TESTSET = 'nr_test_tsp_norm1e6'  # 'tsplib' | 'tsp2d' | 'nr_test_tsp_norm1e6'
if __name__ == "__main__":
    # test for 100 nodes:
    # with open(os.path.join(DIR,'nr_test_tsp_norm1e6_concorde_s2v_results.pkl'), 'rb') as f:
    #     s2v_concorde_res = pickle.load(f)
    # with open(os.path.join(DIR,'nr_test_tsp_norm1e6_concorde_s2v_40_50_results.pkl'), 'rb') as f:
    #     s2v4050_concorde_res = pickle.load(f)
    # # with open(os.path.join(DIR,'nr_results.pkl'), 'rb') as f:
    # #     nr_nr2opt_res = pickle.load(f)
    # with open(os.path.join(DIR,'nr_results100.pkl'), 'rb') as f:
    #     nr_nr2opt_res = pickle.load(f)

    # test for 200 nodes:
    with open(os.path.join(DIR, 'nr_test_tsp_norm1e6-all-results.pkl'), 'rb') as f:
        all_results100 = pickle.load(f)
    with open(os.path.join(DIR, 'nr_results100.pkl'), 'rb') as f:
        nr_nr2opt_res100 = pickle.load(f)
    # insert nr/nr2opt results into all_results:
    all_results100['nr'] = {}
    all_results100['nr2opt'] = {}
    for city in nr_nr2opt_res100.keys():
        all_results100['nr'][city] = {'len': nr_nr2opt_res100[city]['nr_len'],
                                           'time': nr_nr2opt_res100[city]['nr_time']}
        all_results100['nr2opt'][city] = {'len': nr_nr2opt_res100[city]['nr2opt_len'],
                                          'time': nr_nr2opt_res100[city]['nr2opt_time']}
    # Save unified all results:
    with open(os.path.join(DIR, 'all_results100.pkl'), 'wb') as f:
        pickle.dump(all_results100, f)
        print('saving all_results100 to {}'.format(os.path.join(DIR, 'all_results100.pkl')))

    # Calculate approximation/time ratio
    headers100 = ['Index', 'Model', 'Trained on', 'Nstep', 'gamma', 'epsilon', 'l2', 'Approximation', 'Time']
    tab100 = []
    for model_name in all_results100.keys():
        if model_name == 'concorde':
            continue
        aprx_ratio, time_ratio = [], []
        for city in all_results100[model_name].keys():
            aprx_ratio.append(all_results100[model_name][city]['len'] / all_results100['concorde'][city]['len'])
            time_ratio.append(all_results100[model_name][city]['time'] / all_results100['concorde'][city]['time'])

        model_split = model_name.split('-')
        if 'dqn' in model_split:
            trained_on = '-'.join(model_split[model_split.index('dqn') + 1:model_split.index('dqn') + 4])
            epsilon = '-'.join(model_split[model_split.index('epsilon') + 1:model_split.index('epsilon') + 4])
            gamma = model_split[model_split.index('gamma') + 1]
            l2 = model_split[model_split.index('l2') + 1]
            nstep = model_split[model_split.index('nstep') + 1]
            line = ['S2V', trained_on, nstep, gamma, epsilon, l2,'{:.3f}'.format(np.mean(aprx_ratio)), '{:.3f}'.format(np.mean(time_ratio))]
        else:
            line = [model_name, 'random-50', '-', '-', '-', '-','{:.3f}'.format(np.mean(aprx_ratio)), '{:.3f}'.format(np.mean(time_ratio))]
        tab100.append(line)
    # sort tab:
    tab100.sort(key=lambda x: float(x[-2]))
    [line.insert(0,idx+1) for idx, line in enumerate(tab100)]

    print('----------------------------------------------------------------------------')
    print('Average approximation and execution time ratio over 1000 graphs of 100 nodes')
    print('----------------------------------------------------------------------------')
    print(tabulate(tab100, headers=headers100))
    csv_results100 = [headers100]+tab100
    with open("all_results100.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_results100)
    print('all_results100 saved to csv file: all_results100.csv')

    # test for 200 nodes:
    with open(os.path.join(DIR,'nr_test_tsp200_norm1e6-all-results.pkl'), 'rb') as f:
        all_results200 = pickle.load(f)
    with open(os.path.join(DIR,'nr_results200.pkl'), 'rb') as f:
        nr_nr2opt_res200 = pickle.load(f)
    # insert nr/nr2opt results into all_results:
    all_results200['nr'] = {}
    all_results200['nr2opt'] = {}
    for city in nr_nr2opt_res200.keys():
        all_results200['nr'][city] = {'len': nr_nr2opt_res200[city]['nr_len'], 'time': nr_nr2opt_res200[city]['nr_time']}
        all_results200['nr2opt'][city] = {'len': nr_nr2opt_res200[city]['nr2opt_len'], 'time': nr_nr2opt_res200[city]['nr2opt_time']}
    # Save unified all results:
    with open(os.path.join(DIR,'all_results200.pkl'), 'wb') as f:
        pickle.dump(all_results200, f)
        print('saving all_results200 to {}'.format(os.path.join(DIR,'all_results200.pkl')))
    

    # Calculate approximation/time ratio
    headers200 = ['Index', 'Model', 'Trained on', 'Nstep',   'gamma', 'epsilon','l2', 'Approximation', 'Time']
    tab200 = []
    for model_name in all_results200.keys():
        if model_name == 'concorde':
            continue
        aprx_ratio, time_ratio = [], []
        for city in all_results200[model_name].keys():
            aprx_ratio.append(all_results200[model_name][city]['len'] / all_results200['concorde'][city]['len'])
            time_ratio.append(all_results200[model_name][city]['time'] / all_results200['concorde'][city]['time'])

        model_split = model_name.split('-')
        if 'dqn' in model_split:
            trained_on = '-'.join(model_split[model_split.index('dqn')+1:model_split.index('dqn')+4])
            epsilon = '-'.join(model_split[model_split.index('epsilon')+1:model_split.index('epsilon')+4])
            gamma = model_split[model_split.index('gamma')+1]
            l2 = model_split[model_split.index('l2')+1]
            nstep = model_split[model_split.index('nstep')+1]
            line = ['S2V', trained_on, nstep, gamma, epsilon, l2, '{:.3f}'.format(np.mean(aprx_ratio)), '{:.3f}'.format(np.mean(time_ratio))]
        else:
            line = [model_name, 'random-50', '-', '-', '-', '-','{:.3f}'.format(np.mean(aprx_ratio)), '{:.3f}'.format(np.mean(time_ratio))]
        tab200.append(line)

    # sort tab:
    tab200.sort(key=lambda x: float(x[-2]))
    [line.insert(0,idx+1) for idx, line in enumerate(tab200)]

    print('----------------------------------------------------------------------------')
    print('Average approximation and execution time ratio over 1000 graphs of 200 nodes')
    print('----------------------------------------------------------------------------')
    print(tabulate(tab200, headers=headers200))
    csv_results200 = [headers200] + tab200
    with open("all_results200.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_results200)
    print('all_results100 saved to csv file: all_results200.csv')
    print('Finished')

