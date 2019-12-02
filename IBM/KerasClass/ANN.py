import numpy as np
from matplotlib import pyplot as plt
class Node():

    def __init__(self,layer,name,weights,bias):
        
        #Which layer is this node in
        self.layer = layer
        
        #An integer name
        self.name = name
        
        #Weights should be an array of the size of the previous input layer
        self.weights = weights
        
        ##Define Bias
        self.bias = bias
        
        ##Valuse
        self.value = np.zeros(1)

        
    def set_node_value(self,z):
        self.value = np.array(z)
        
    def __str__(self):
        return " {} {} {} {}" .format(self.layer, self.name,self.weights,self.bias)


class Layer():

    def __init__(self,name,num_nodes,previous,outlayer=False):
        #Name of the layer
        self.name = name
        
        #How many nodes are in this layer
        self.num_nodes = num_nodes
        
        #How many nodes in the next layer
        self.previous_nodes = previous
        
        #Store a dictionary object of nodes
        self.nodes = {}
        for i in range(0,self.num_nodes):
            #Node takes in layer name, the ith number associated and an array of previous nodes.
            self.nodes[i] = Node(self.name,
                                 i,
                                 np.around(
                                    np.random.uniform(size=self.previous_nodes), decimals=2),
                                 np.around(
                                     np.random.uniform(size=1),decimals=2))
            
    def set_layer_by_array(self,X):
        ##X - [1,2,3] will set nodes0,1,2 to [1,2,3]
        assert(len(X) == len(self.nodes))
        for x in range(0,len(X)):
            self.nodes[x].set_node_value(np.array(X[x]))
            
        
    def __getitem__(self,x):
        return self.nodes[x]

    def __iter__(self):
        for node in self.nodes:
            yield self.nodes[node]
            
    def __len__(self):
        return len(self.nodes)
        
        

class ANNNetwork():
    
    def __init__(self,topology=[]):
        
        #This will define our network topology
        self.topology = {}
        
        #This will define our network.
        self.network = {}

    def set_layer_topology(self,topology):
        self.topology = {k:v for k,v in enumerate(topology,start=0)}
        self.setup_layers()

    def setup_layers(self):
        for layer in self.topology:
            if layer == 0:
                ##Input Layer
                
                ##Input Layer so no previous nodes1
                previous_nodes = 0
                #Layer takes in name, num nodes, and the number of nodes in previous layer
                self.network[layer] = Layer(
                    'Input',
                    self.topology[layer],previous_nodes)
                continue
            elif layer == len(self.topology)-1:
                ##Output Layer
                
                ##Previous nodes
                previous_nodes = self.topology[layer-1]
                
                self.network[layer] = Layer('Output', self.topology[layer],previous_nodes)
                continue
            
            ##We are in a hidden layer
            previous_nodes = self.topology[layer-1]
            name_ = 'Hiden_{}'.format(layer-1)
            self.network[layer]= Layer(name_,self.topology[layer],previous_nodes)
            
                
    def set_input(self,X):
        
        if len(X) != len(self.network[0]):
            raise Exception("Must input {} values for input".format(len(self.network[0])))
        else:
            X = np.asarray(X)
            self.network[0].set_layer_by_array(X)
        
            
    def __getitem__(self,x):
        
        return self.network[x]
    
    def __repr__(self):
       return self.network.__repr__()
            
    def __str__(self):
        rep_str = ""
        for layer in self.network:
            
            rep_str += "Layer {}: Name - {}\n".format(layer,self.network[layer].name)
            for node in self.network[layer]:
               rep_str+="\tNode -{}\n\t\tWeights - {}\n\t\tBias - {}\n\t\tValue - {}\n".format(
                   node.name,
                   node.weights,
                   node.bias,
                   node.value)
        return rep_str
    
    def sigmoid_function(self,value):
        return np.around(1/(1+np.exp(-value)),decimals=3)
    
    def forward_propogate(self):
        '''
        Forward propogate all values
        
        ex. Hidden layer node 1 value will be all the previous input nodes [X1,X2..XN]
        
        and times them by all the weights of the nodes [W1,W2..WN]
        
        (X1*W1 + X2*W2 ... XN+WN)+Bias
        
        Since all the weights of the nodes are the same size as the number of inputs they should be the same
        
        
        '''
        for layer in list(self.network.keys())[1:]:
            ##Start in the hidden layers
            for node in self.network[layer]:
                #Go through all nodes
                X = []
                #Take in the previous_layer_input
                #Look at all the values and put them in an array
                for previous_layer_node in self.network[layer-1]:
                    X.append(previous_layer_node.value)
                X = np.asanyarray(X)
                ##Weights is also an ordered array
                weights = node.weights
                ##Value is their dot product
                value = self.sigmoid_function(
                    np.around(np.sum(X*weights)+node.bias,decimals=2))
                node.set_node_value(value)
                
                
    def draw_neural_net(self):
        '''plotting function for AN'''
        left=0.1
        right=0.91
        bottom=0.1
        top=0.91
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        layer_sizes = list(self.topology.values())
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                value = float(np.around(Ann[n][m].value,decimals=2))
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
                label = ax.annotate(
                    value, xy=(n*h_spacing + left, layer_top - m*v_spacing), fontsize=12, ha="center",zorder=5)


        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                    ax.add_artist(line)
        return ax

Ann = ANNNetwork()
Ann.set_layer_topology([4,10,10,4,1])
Ann.set_input([0.2,0.01,0.01,0.2])
Ann.forward_propogate()
Ann.draw_neural_net()
plt.show()
