#-*- coding: utf-8 -*-
import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA


# Compute a sequence's reward
def reward(tsp_sequence):
    tour = np.concatenate((tsp_sequence, np.expand_dims(tsp_sequence[0],0))) # sequence to tour (end=start)
    inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1,:2]-tour[1:,:2]),axis=1)) # tour length
    return np.sum(inter_city_distances) # reward

# Swap city[i] with city[j] in sequence
def swap2opt(tsp_sequence,i,j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j+1] = np.flip(tsp_sequence[i:j+1], axis=0) # flip or swap ?
    return new_tsp_sequence

# One step of 2opt = one double loop and return first improved sequence
def step2opt(tsp_sequence):
    seq_length = tsp_sequence.shape[0]
    distance = reward(tsp_sequence)
    for i in range(1,seq_length-1):
        for j in range(i+1,seq_length):
            new_tsp_sequence = swap2opt(tsp_sequence,i,j)
            new_distance = reward(new_tsp_sequence)
            if new_distance < distance:
                return new_tsp_sequence, new_distance
    return tsp_sequence, distance


class DataGenerator(object):

    def __init__(self):
        pass

    def gen_instance(self, max_length, dimension, seed=0): # Generate random TSP instance
        if seed!=0: np.random.seed(seed)
        sequence = np.random.rand(max_length, dimension) # (max_length) cities with (dimension) coordinates in [0,1]
        pca = PCA(n_components=dimension) # center & rotate coordinates
        sequence = pca.fit_transform(sequence) 
        return sequence

    def train_batch(self, batch_size, max_length, dimension): # Generate random batch for training procedure
        input_batch = []
        for _ in range(batch_size):
            input_ = self.gen_instance(max_length, dimension) # Generate random TSP instance
            input_batch.append(input_) # Store batch
        return input_batch

    def test_batch(self, batch_size, max_length, dimension, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        input_ = self.gen_instance(max_length, dimension, seed=seed) # Generate random TSP instance
        for _ in range(batch_size): 
            sequence = np.copy(input_)
            if shuffle==True: 
                np.random.shuffle(sequence) # Shuffle sequence
            input_batch.append(sequence) # Store batch
        return input_batch

    def loop2opt(self, tsp_sequence, max_iter=2000): # Iterate step2opt max_iter times (2-opt local search)
        best_reward = reward(tsp_sequence)
        new_tsp_sequence = np.copy(tsp_sequence)
        for _ in range(max_iter): 
            new_tsp_sequence, new_reward = step2opt(new_tsp_sequence)
            if new_reward < best_reward:
                best_reward = new_reward
            else:
                break
        return new_tsp_sequence, best_reward

    def visualize_2D_trip(self, trip): # Plot tour
        plt.figure(1)
        colors = ['red'] # First city red
        for i in range(len(trip)-1):
            colors.append('blue')
            
        plt.scatter(trip[:,0], trip[:,1],  color=colors) # Plot cities
        tour=np.array(list(range(len(trip))) + [0]) # Plot tour
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--")

        plt.xlim(-0.75,0.75)
        plt.ylim(-0.75,0.75)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def visualize_sampling(self, permutations): # Heatmap of permutations (x=cities; y=steps)
        max_length = len(permutations[0])
        grid = np.zeros([max_length,max_length]) # initialize heatmap grid to 0

        transposed_permutations = np.transpose(permutations)
        for t, cities_t in enumerate(transposed_permutations): # step t, cities chosen at step t
            city_indices, counts = np.unique(cities_t,return_counts=True,axis=0)
            for u,v in zip(city_indices, counts):
                grid[t][u]+=v # update grid with counts from the batch of permutations

        fig = plt.figure(1) # plot heatmap
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(grid, interpolation='nearest', cmap='gray')
        plt.colorbar()
        plt.title('Sampled permutations')
        plt.ylabel('Time t')
        plt.xlabel('City i')
        plt.show()

def coord2tspfile(record, dirname, tsp_comment='random'):
    '''
    Generate for data standard TSP folder,
    containing .tsp files of each instance
    :param dict record: dict of cities and nr/nr2opt results
    :param str dirname: path to save the tsp files.
    :param str tsp_comment: description to be written in the tsp files
    :return: None
    '''
    norm = 1000000 # normalize for evaluation by concorde
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    city_id = 0
    nr_results = {}
    for city in tqdm(record['test_data'], 'Generating tsp files...'):
        tsp_fname = 'city{}-{}-{}nodes.tsp'.format(city_id, tsp_comment, len(city[0]))
        with open(os.path.join(dirname, tsp_fname), 'w') as f:
            f.write('NAME: '+tsp_fname[:-4]+'\n')
            f.write('TYPE: TSP\n')
            f.write('COMMENT: '+tsp_comment+'\n')
            f.write('DIMENSION: {}\n'.format(len(city[0])))
            f.write('EDGE_WEIGHT_TYPE: EUC_2D\n')
            f.write('NODE_COORD_SECTION\n')
            for n_id, node in enumerate(city[0]):
                f.write('{:d} {} {}\n'.format(n_id+1, node[0]*norm, node[1]*norm))
            f.write('EOF')
        nr_results[tsp_fname[:-4]] = {
            'nr_len': record['predictions_length'][city_id],
            'nr_time': record['test_nr_time'][city_id],
            'nr2opt_len': record['predictions_length_w2opt'][city_id],
            'nr2opt_time': record['test_nr2opt_time'][city_id]
        }
        city_id += 1

    with open('nr_results100.pkl', 'wb') as f:
        pickle.dump(nr_results, f)

if __name__ == '__main__':
    with open('save/2D_TSP50_b256_e128_n512_s3_h16_q360_u256_c256_lr0.001_d5000_0.96_T1.0_steps20000_i7.0/nr-test-results-100nodes-from-2019-02-07_09-21-49.pkl', 'rb') as f:
        record = pickle.load(f)
    coord2tspfile(record, 'nr_test_tsp100_norm1e6', 'random100_norm-1e6')
    print('Finished')