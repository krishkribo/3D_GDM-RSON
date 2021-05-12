# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@author: German I. Parisi (german.parisi@gmail.com)
@edited: Krishna
"""

import gtls
import numpy as np
from episodic_gwr import EpisodicGWR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def replay_samples(net, size) -> (np.ndarray, np.ndarray):
    samples = np.zeros(size)
    r_weights = np.zeros((net.num_nodes, size, net.dimension))
    r_labels = np.zeros((net.num_nodes, len(net.num_labels), size))
    for i in range(0, net.num_nodes):
        for r in range(0, size):
            if r == 0: samples[r] = i
            else: samples[r] = np.argmax(net.temporal[int(samples[r-1]), :])
            #print(type(net.weights))
            r_weights[i, r] = net.weights[int(samples[r])][0]
            for l in range(0, len(net.num_labels)):
                r_labels[i, l, r] = np.argmax(net.alabels[l][int(samples[r])])
    return r_weights, r_labels

def plot_error(x,y,xlabel,ylabel,color,title,m_title,epochs)->None:
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.plot(x[0],y[0],color=color[0])
    ax1.set_title(title[0])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend([str(epochs)+" epochs"])
    ax2.plot(x[1],y[1],color=color[1])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(title[1])
    ax2.legend([str(epochs)+" epochs"])
    f.suptitle(m_title)
    plt.show()

if __name__ == "__main__":

    train_flag = True
    train_type = 1 # 0:Batch, 1: Incremental
    train_replay = False
    
    ds_iris = gtls.IrisDataset(file='iris.csv', normalize=True)
    print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    assert train_type < 2, "Invalid type of training."
    
    '''
    Episodic-GWR supports multi-class neurons.
    Set the number of label classes per neuron and possible labels per class
    e.g. e_labels = [50, 10]
    is two labels per neuron, one with 50 and the other with 10 classes.
    Setting the n. of classes is done for experimental control but it is not
    necessary for associative GWR learning.
    '''
    e_labels = [3, 3]
    s_labels = [3]
    ds_vectors = ds_iris.vectors
    ds_labels = np.zeros((len(e_labels), len(ds_iris.labels)))
    ds_labels[0] = ds_iris.labels
    ds_labels[1] = ds_iris.labels
    
    num_context = 1 # number of context descriptors
    epochs = 20 # epochs per sample for incremental learning
    a_threshold = [0.95, 0.9]
    beta = 0.7
    learning_rates = [0.2, 0.001]
    context = True
    
    print(f"ds_vectors : {ds_vectors}")
    print(f"shape ds_vectors:{np.shape(ds_vectors)}")
    print(ds_vectors[0])
    print(ds_iris.labels)
    #exit()
    g_episodic = EpisodicGWR()
    g_episodic.init_network(ds_iris, e_labels, num_context)
    
    g_semantic = EpisodicGWR()
    g_semantic.init_network(ds_iris, s_labels, num_context)
    
    if train_type == 0:
        # Batch training
        # Train episodic memory
        print('------ episodic -------')
        e_NN,e_qerr = g_episodic.train_egwr(ds_vectors, ds_labels, epochs, a_threshold[0],
                              beta, learning_rates, context, regulated=0)
                              
        e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels,
                                              ret_vecs=True)
        # Train semantic memory
        print('------ semantic -------')
        s_NN, s_qerr = g_semantic.train_egwr(e_weights, e_labels, epochs, a_threshold[1], beta, 
                          learning_rates, context, regulated=1)        
    else:
        # Incremental training
        n_episodes = 0
        batch_size = 5 # number of samples per epoch
        # Replay parameters
        replay_size = (num_context * 2) + 1 # size of RNATs
        replay_weights = []
        replay_labels = []
        
        # Train episodic memory
        for s in range(0, ds_vectors.shape[0], batch_size):
            print('------ episodic -------')
            e_NN,e_qerr = g_episodic.train_egwr(ds_vectors[s:s+batch_size],
                                  ds_labels[:, s:s+batch_size],
                                  epochs, a_threshold[0], beta, learning_rates,
                                  context, regulated=0)
            
            e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels,
                                                  ret_vecs=True)
            # Train semantic memory
            print('------ semantic -------')
            s_NN, s_qerr = g_semantic.train_egwr(e_weights[s:s+batch_size],
                                  e_labels[:, s:s+batch_size],
                                  epochs, a_threshold[1], beta, learning_rates,
                                  context, regulated=1)
                                  
            if train_replay and n_episodes > 0:
                # Replay pseudo-samples
                for r in range(0, replay_weights.shape[0]):
                    print('------ episodic -------')
                    e_NN,e_qerr = g_episodic.train_egwr(replay_weights[r], replay_labels[r, :],
                                          epochs, a_threshold[0], beta,
                                          learning_rates, 0, 0)
                    print('------ semantic -------')
                    s_NN, s_qerr = g_semantic.train_egwr(replay_weights[r], replay_labels[r],
                                          epochs, a_threshold[1], beta, 
                                          learning_rates, 0, 1)
            # Generate pseudo-samples
            if train_replay: 
                replay_weights, replay_labels = replay_samples(g_episodic, 
                                                               replay_size)
            
            n_episodes += 1


    print(f"EpisodicGWR shape :{np.shape(g_episodic.weights)}")
    print(f"SemanticGWR shape :{np.shape(g_semantic.weights)}")

    g_episodic.test(ds_vectors, ds_labels, test_accuracy=True)
    g_semantic.test(e_weights, e_labels, test_accuracy=True)
        
    print("Accuracy episodic: %s, semantic: %s" % 
          (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]))
    
    print(e_NN)
    print(e_qerr)
    plot_error([e_NN,s_NN],[e_qerr,s_qerr],"No of Neurons","Quantization error",
        ['r','g'],["episodic memory","semantic memory"],'NN-Quantization error',epochs)

    # PCA process 
    """pca = PCA(n_components=2)
    pca.fit(g_episodic.weights[0])
    g_episodic_pca = pca.fit_transform(g_episodic.weights[0])
    
    pca.fit(g_semantic.weights[0])
    g_semantic_pca = pca.fit_transform(g_semantic.weights[0])

    print(f"EpisodicGWR shape PCA transformed : {np.shape(g_episodic_pca)}")
    print(f"SemanticGWR shape PCA transformed : {np.shape(g_semantic_pca)}")
    """
    if train_type == 0:
        gtls.plot_network(g_episodic,edges=True,labels=False,title='episodic memory - batch training')
        gtls.plot_network(g_semantic,edges=True,labels=False,title='semantic memory - batch training')
    else:
        gtls.plot_network(g_episodic,edges=True,labels=False,title='episodic memory - incremental training')
        gtls.plot_network(g_semantic,edges=True,labels=True,title='semantic memory - incremental training')
