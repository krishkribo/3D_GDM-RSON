from heapq import nsmallest
import numpy as np
import math
try:
    from preprocessing import processing
except Exception:
    from .preprocessing import processing
    
from tqdm import tqdm
import warnings

class EpisodicGWR(processing):

    def __init__(self):
        self.iterations = 0
        self.count = 0 
        self.dist_type = None
        self.data_pre_process = False
        self.debug = False
    
    def init_network(self, ds, e_labels, random, learning_type, num_context) -> None:
        
        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"
        
        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2
        
        if learning_type == 0:
            ds = ds
        elif learning_type == 1:
            ds = ds[0] # the first label to be learnt is initialised

        self.dimension = ds.shape[1]
        #self.dimension = np.shape(ds[0])
        self.num_context = num_context
        self.depth = self.num_context + 1
        empty_neuron = np.zeros((self.depth, self.dimension))
        self.weights = [empty_neuron, empty_neuron]
        
        # Global context
        self.g_context = np.zeros((self.depth, self.dimension))        
        
        # Create habituation counters
        self.habn = [1, 1]
        
        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))
        
        # Temporal connections
        self.temporal = np.zeros((self.num_nodes, self.num_nodes))

        # Label histogram
        self.num_labels = e_labels
        self.alabels = []
        for l in range(0, len(self.num_labels)):
            self.alabels.append(-np.ones((self.num_nodes, self.num_labels[l])))
            
        # Initialize weights
        self.random = random
        if self.random: 
            init_ind = np.random.randint(0, ds.shape[0], self.num_nodes)
        else:
            init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            self.weights[i][0] = ds[init_ind[i]]
            
        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)

    def compute_alphas(self, coeff):
        alpha_w = np.zeros(coeff)
        for a in range(0,len(alpha_w)):
            alpha_w[a] = np.exp(-a)
        alpha_w[:] = alpha_w[:]/ np.sum(alpha_w)
        return alpha_w

    def compute_distance(self, x1, x2, dist_type, **kwargs):
        preprocess = kwargs.get('data_process', False)
        d_len = len(self.alphas) # aplhas, x1 and x2 be of same length
        if preprocess:
            if self.debug: print(f"Data preprocessing -->")
            # filter the zeros elements in the input matrix 
            x1 = super().filter_zeros(x1)
            x2 = super().filter_zeros(x2)
            # get the arrays of equal size 
            x1, x2 = super().get_equal_array(x1=x1, x2=x2)
        s = super()
        d_j = np.sum(np.array(
            [self.alphas[i]*s.get_distance(x=x1[i], y=x2[i], type=dist_type) for i in range(d_len)]
        ))
        
        return d_j

    def find_bmus(self, d_input, **kwargs):
        s_best = kwargs.get('s_best', False)
        n_best = kwargs.get('no_best', 2)
        dist_type = kwargs.get('dist_type', 'euclidean')
        data_process = kwargs.get('data_process', False)
        distance = np.zeros(self.num_nodes)
        for d in range(0,self.num_nodes):
            distance[d] = self.compute_distance(self.weights[d], d_input, dist_type=dist_type, data_process=data_process)

        if s_best:
            bmus = nsmallest(n_best,((k,i) for i,k in enumerate(distance))) 
            return bmus[0][1], bmus[0][0], bmus[1][1]
        else:
            return np.argmin(distance), distance[np.argmin(distance)]
        

    def expand_matrix(self, matrix):
        e_matrix = np.hstack((matrix, np.zeros((matrix.shape[0],1))))
        e_matrix = np.vstack((e_matrix, np.zeros((1, e_matrix.shape[1]))))
        return e_matrix

    def add_node(self, b_index, **kwargs):
        new_node = kwargs.get('new_node', True)
        if new_node:
            n_neuron = np.array((self.weights[b_index]+self.g_context)/2)
            self.weights.append(n_neuron)
            self.num_nodes += 1
    
    def update_weight(self, n_index, epsilon):
        n_weight = np.dot( 
                    (self.g_context - self.weights[n_index]), (epsilon * self.habn[n_index]))
        self.weights[n_index] += n_weight 

    def habituate_node(self, index, tau, **kwargs):
        new_node = kwargs.get('new_node', False)
        kappa = kwargs.get('kappa', 1.05)
        if not new_node:
            habn = tau * kappa * (1-self.habn[index]) - tau
            self.habn[index] += habn
        else:
            self.habn.append(1)
        
    def update_neighbors(self, n_index, epsilon, tau_n):
        neighbours = np.nonzero(self.edges[n_index])
        for n in range(0, len(neighbours[0])):
            idx = neighbours[0][n]
            self.update_weight(idx, epsilon)
            self.habituate_node(idx, tau_n, new_node=False)

    def update_edges(self, f_index, s_index, **kwargs):
        new_index = kwargs.get('new_index', False)
        self.ages += 1
        if self.debug:
            print(self.ages)
            print(self.ages.shape)
        '''for n in range(self.num_nodes):
            if self.habn[n] > 0.2:
                self.ages[n]+=1
            else:
                self.ages[n] = self.ages[n]'''
        if new_index:
            self.edges = self.expand_matrix(self.edges)
            self.ages = self.expand_matrix(self.ages)
            self.edges[f_index, s_index] = 0
            self.edges[s_index, f_index] = 0
            self.ages[f_index, s_index] = 0
            self.ages[s_index, f_index] = 0
            self.edges[f_index, new_index] = 1
            self.edges[new_index, f_index] = 1
            self.edges[s_index, new_index] = 1
            self.edges[new_index, s_index] = 1
        else:
            self.edges[f_index, s_index] = 1
            self.edges[s_index, f_index] = 1
            self.ages[f_index, s_index] = 0
            self.ages[s_index, f_index] = 0    
    
    def update_temporal(self, current_ind, previous_ind, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if new_node:
            self.temporal = self.expand_matrix(self.temporal)
        if previous_ind != -1 and previous_ind != current_ind:
            self.temporal[previous_ind, current_ind] += 1

    def update_labels(self, bmu, label, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)        
        if not new_node:
            for l in range(0, len(self.num_labels)):
                for a in range(0, self.num_labels[l]):
                    if a == label[l]:
                        self.alabels[l][bmu, a] += self.a_inc
                    else:
                        if label[l] != -1:
                            self.alabels[l][bmu, a] -= self.a_dec
                            if (self.alabels[l][bmu, a] < 0):
                                self.alabels[l][bmu, a] = 0              
        else:
            for l in range(0, len(self.num_labels)):
                new_alabel = np.zeros((1, self.num_labels[l]))
                if self.debug: print(f"labels in the update labels : {int(label[l])}")
                if label[l] != -1:
                    new_alabel[0, int(label[l])] = self.a_inc
                if self.debug: print(f"labels in the update labels : {new_alabel}")
                self.alabels[l] = np.concatenate((self.alabels[l], new_alabel), axis=0)
        
        if self.debug: print(f"alabels : {self.alabels}")       
    
    def remove_old_edges(self):
        removed_nodes = []
        ages_of_nodes = []
        habn_nodes = []
        for i in range(self.num_nodes):
            neighbours = np.nonzero(self.edges[i])
            if self.habn[i] > self.rm_threshold:
                for j in neighbours[0]:
                    if self.max_age != None and self.ages[i,j] > self.max_age:
                        if self.debug: ages_of_nodes.append(self.ages[i])
                        self.edges[i,j] = 0
                        self.edges[j,i] = 0
                        self.ages[i,j] = 0
                        self.ages[j,i] = 0
                        if self.debug:
                            removed_nodes.append(i) # for debugging
                            habn_nodes.append(self.habn[i])
            else:
                for j in neighbours[0]:
                    if self.max_age != None and self.ages[i,j] > self.max_age:
                        self.ages[i,j] = 0
                        self.ages[j,i] = 0

        if self.debug:
            print(f"ages of isolated nodes : {ages_of_nodes}")
            print(f"removed_nodes : {removed_nodes}")
            print(f"habituation counter of the removed nodes : {habn_nodes}")


    def remove_isolated_nodes(self) -> None:
        if self.num_nodes > 2:
            ind_c = 0
            rem_c = 0
            while (ind_c < self.num_nodes):
                neighbours = np.nonzero(self.edges[ind_c])            
                if len(neighbours[0]) < 1:
                    if self.num_nodes > 2:
                        self.weights.pop(ind_c)
                        self.habn.pop(ind_c)
                        for d in range(0, len(self.num_labels)):
                            d_labels = self.alabels[d]
                            self.alabels[d] = np.delete(d_labels, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=1)
                        self.ages = np.delete(self.ages, ind_c, axis=0)
                        self.ages = np.delete(self.ages, ind_c, axis=1)
                        self.temporal = np.delete(self.temporal, ind_c, axis=0)
                        self.temporal = np.delete(self.temporal, ind_c, axis=1)
                        self.num_nodes -= 1
                        rem_c += 1
                    else: return
                else:
                    ind_c += 1
            print ("(-- Removed %s neuron(s))" % rem_c)

    def run_egwr(self,epoch,ds_vectors,ds_labels,w,w_logs) -> None:
        # Start training
        self.error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))
        previous_ind = -1

        samples = ds_vectors.shape[0]
        print(f"\nNumber of neurons before training: {self.num_nodes}")
        for iteration in tqdm(range(0, samples)):
            # Generate input sample
            self.g_context[0] = ds_vectors[iteration]
            label = ds_labels[:, iteration]

            # Update global context
            for z in range(1, self.depth):
                self.g_context[z] = (self.beta * previous_bmu[z]) + ((1-self.beta) * previous_bmu[z-1])
            
            # Find the best and second-best matching neurons
            b_index, b_distance, s_index = self.find_bmus(self.g_context, s_best = True, 
                data_process=self.data_pre_process, dist_type=self.dist_type)
            
            b_label = np.argmax(self.alabels[0][b_index])
            if self.debug: print(f"Best matching label : {b_label}")
            misclassified = b_label != label[0]

            if self.debug: 
                print(f"b_label : {b_label}, label : {label}, misclassified : {misclassified}")
                w.log({"b_label" : b_label, "label" : label[0]})
            
            # Quantization error
            self.error_counter[epoch] += b_distance
            
            # Compute network activity
            a = math.exp(-b_distance)
            
            w.log({ w_logs[0] : a})
            w.log({w_logs[9]:self.habn[b_index]})
            
            # Store BMU at time t for t+1
            previous_bmu = self.weights[b_index]

            if (not self.regulated) or (self.regulated and misclassified):
                
                if self.max_nodes == 0 or self.max_nodes == None:
                    c_check = a < self.a_threshold and self.habn[b_index] < self.hab_threshold
                
                else: 
                    c_check = a < self.a_threshold and self.habn[b_index] < self.hab_threshold and self.num_nodes < self.max_nodes

                if (c_check):
                    # Add new neuron
                    n_index = self.num_nodes
                    self.add_node(b_index)
                   
                    # Add label histogram           
                    self.update_labels(n_index, label, new_node = True)                   

                    # Update edges and ages
                    self.update_edges(b_index, s_index, new_index = n_index)
                    
                    # Update temporal connections
                    self.update_temporal(n_index, previous_ind, new_node = True)

                    # Habituation counter                    
                    self.habituate_node(n_index, self.tau_b, new_node = True)

                    w.log({w_logs[2] : self.epsilon_n * self.habn[n_index]})
                
                else:
                    # Habituate BMU
                    self.habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    b_rate, n_rate = self.epsilon_b, self.epsilon_n
                    if self.regulated and misclassified:
                        b_rate *= self.mod_rate
                        n_rate *= self.mod_rate
                    else:
                        # Update BMU's label histogram
                        self.update_labels(b_index, label)

                    self.update_weight(b_index, b_rate)

                    # Update BMU's edges // Remove BMU's oldest ones
                    self.update_edges(b_index, s_index)

                    # Update temporal connections
                    self.update_temporal(b_index, previous_ind)

                    # Update BMU's neighbors
                    self.update_neighbors(b_index, n_rate, self.tau_n)
                    
                    w.log({w_logs[1] : self.epsilon_b * self.habn[b_index]})

            self.iterations += 1
                
            previous_ind = b_index

            w.log({w_logs[3] : self.num_nodes})
            
        return b_index


         
    def train_egwr(self, ds_vectors, ds_labels, params, w, w_logs, **kwargs) -> (list,list,list):
        
        assert not self.locked, "Network is locked. Unlock to train."
        
        self.dist_type = kwargs.get('dist_type', None)
        self.data_pre_process = kwargs.get('data_pre_process', False)
        self.debug = kwargs.get('debug', False)
        t_test = kwargs.get('t_test', False)

        print(f"Distance type used for training : {self.dist_type} with data preprocess {self.data_pre_process},{type(self.data_pre_process)}")

        self.samples = ds_vectors.shape[0]
        self.max_epochs = params['epochs']
        self.a_threshold = params['a_threshold']   
        self.epsilon_b, self.epsilon_n = params['l_rates']
        self.beta = params['beta']
        self.regulated = params['regulated']
        self.context = params['context']
        self.hab_threshold = params['habn_threshold']
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.rm_threshold = params['node_rm_threshold']
        #self.max_neighbors = 10 # optional
        self.max_age = params['max_age']
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1
        self.mod_rate = 0.001

        if not self.context:
            self.g_context.fill(0)

        # learning type 0-batch, 1-incremental
        #self.max_nodes = self.samples # OK for batch, bad for incremental
        if params['learning_type'] == 0:
            if params['batch']: self.max_nodes = self.samples
            else: self.max_nodes = self.samples
        elif params['learning_type'] == 1:
            self.max_nodes = 0

        # wandb - configurations and updates
        config = w.config
        config.max_nodes = self.max_nodes
        config.max_age = self.max_age
        config.new_node = self.new_node
        config.distance_type = self.dist_type
        config.update({"max_nodes":self.max_nodes},allow_val_change=True)
        config.update({"regulated":self.regulated},allow_val_change=True)
        config.update({"samples":self.samples},allow_val_change=True)
        
        num_nodes_list = []
        q_error_list = []
        acc_list = []
        update_rate_list = []

        try:
            for epoch in tqdm(range(1, self.max_epochs+1)):

                # run
                b_index = self.run_egwr(epoch-1, ds_vectors, ds_labels, w, w_logs)

                # Remove old edges
                self.remove_old_edges()

                # Average quantization error (AQE)
                self.error_counter[epoch-1] /= self.samples

                # compute accuracy per epoch 
                if t_test:
                    # testing on the training data
                    # print(test_data)
                    # print(test_d_lables)
                    print(f"\nTesting ---->\n")
                    self.test(ds_vectors, ds_labels, test_accuracy=True, dist_type=self.dist_type, data_pre_process=self.data_pre_process)
                    print(f"-"*10)
                    print ("\nEpoch: %s \n NN: %s \n ATQE: %s \n Accuracy: %s \n Loss: %s \n" % 
                        (epoch, self.num_nodes, self.error_counter[epoch-1], self.test_accuracy[0], 1-self.test_accuracy[0]))
                    acc_list.append(self.test_accuracy[0])

                else:
                    print(f"-"*10)
                    print ("\nEpoch: %s \n NN: %s \n ATQE: %s \n" % 
                        (epoch, self.num_nodes, self.error_counter[epoch-1]))
                
                num_nodes_list.append(self.num_nodes)
                q_error_list.append(self.error_counter[epoch-1])
                update_rate_list.append(self.habn[b_index]*self.epsilon_b)
                w.log({w_logs[4]:self.error_counter[epoch-1]})
                
                if self.debug: w.log({w_logs[5]:self.num_nodes, w_logs[6]:self.ages, w_logs[7]:self.edges})
                self.count+=1

        finally:   
            if params['learning_type'] == 1: 
                if self.count == self.max_epochs:
                    # Remove isolated neurons
                    self.remove_isolated_nodes()
                    #self.num_nodes_list.append(self.num_nodes)
                    w.log({w_logs[8]:self.num_nodes})
                    self.count = 0
            else:
                # Remove isolated neurons
                self.remove_isolated_nodes()
                #self.num_nodes_list.append(self.num_nodes)
                w.log({w_logs[8]:self.num_nodes})
                self.count = 0

        # restting context vector for each classes (incremental)
        self.g_context = np.zeros((self.depth, self.dimension))         
                
        return num_nodes_list, q_error_list, acc_list, update_rate_list

    def test(self, ds_vectors, ds_labels, **kwargs):

        test_accuracy = kwargs.get('test_accuracy', False)
        test_vecs = kwargs.get('ret_vecs', False)
        data_pre_process = kwargs.get('data_pre_process', False)
        dist_type = kwargs.get('dist_type', 'manhattan')

        test_samples = ds_vectors.shape[0]
        print(f"\nNumber of test samples : {test_samples}")
        print(f"Distance type used for testing : {self.dist_type} with data preprocess {self.data_pre_process}, type: {type(self.data_pre_process)}")

        self.re_trainflag = False
        self.bmus_index = -np.ones(test_samples)
        self.bmus_weight = np.zeros((test_samples, self.dimension))
        self.bmus_label = -np.ones((len(self.num_labels), test_samples))
        self.bmus_activation = np.zeros(test_samples)
        
        input_context = np.zeros((self.depth, self.dimension))
        
        if test_accuracy:
            acc_counter = np.zeros(len(self.num_labels))

        for i in tqdm(range(0, test_samples)):
            input_context[0] = ds_vectors[i]
            # Find the BMU
            b_index, b_distance = self.find_bmus(input_context, data_process=data_pre_process, dist_type=dist_type)
            self.bmus_index[i] = b_index
            self.bmus_weight[i] = self.weights[b_index][0]
            self.bmus_activation[i] = math.exp(-b_distance)
            
            for l in range(0, len(self.num_labels)):
                #print(np.argmax(self.alabels[l][b_index]))
                self.bmus_label[l, i] = np.argmax(self.alabels[l][b_index])
                if ds_labels is not None and ds_labels[l,i] not in np.arange(0, self.num_labels[0],1):
                    self.bmus_label[l,i] = None
                    print(f"Unknow label detected")
                    inp = input('Teach new labels (y/n):')
                    while inp:
                        if inp == 'y' or inp == 'n': break
                        inp = input("Enter the correct input (y/n):")
                    if inp == 'y':
                        self.re_trainflag = True
                    elif inp == 'n':
                        self.re_trainflag = False
                
                elif ds_labels is None:
                    print(f"warning: input label not found")
                #print(f"l : {self.bmus_label[l, i]}")
            #print(f"BMU label : {self.bmus_label}")

            for j in range(1, self.depth):
                input_context[j] = input_context[j-1]
            
            if test_accuracy:
                if ds_labels is None:
                    raise UserWarning('input label not found')
                for l in range(0, len(self.num_labels)):
                    #print(f"GT label : {ds_labels[l,i]}")
                    #print(f"predicted label : {self.bmus_label[l,i]}")
                    if self.bmus_label[l, i] == ds_labels[l, i]:
                        acc_counter[l] += 1
        
        if test_accuracy: print(f"Accuracy counter : {acc_counter}")                
        
        if test_accuracy: self.test_accuracy =  acc_counter / ds_vectors.shape[0]
            
        if test_vecs:
            s_labels = -np.ones((1, test_samples))
            s_labels[0] = self.bmus_label[1]
            return self.bmus_weight, s_labels

        else:
            return self.bmus_label
