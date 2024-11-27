import numpy as np
from utils.utils import load_data
import sys
import requests
requests.packages.urllib3.disable_warnings()
import yaml
import json
from txtai.embeddings import Embeddings


class agent_classify:
    def __init__(self,parafile):
        # read YAML
        with open(parafile, 'r') as file:
            parameters = yaml.safe_load(file)
        ssize = parameters['samplesize']
        self.load_embedding = parameters['load_embedding']
        self.dataname = parameters['dataset_name']
        self.abs_len = parameters['abstract_len']
        self.shotnum = parameters['fewshot_num']
        ## create embed

        ## load data
        self.data, self.text = load_data(self.dataname, use_text=True, seed=42)
        if ssize == "all":
            self.sample_size = int(len(self.data.test_id)/2)
        else:
            self.sample_size = ssize
        ##
        idx_list = list(range(self.sample_size))
        node_indices = self.sample_test_nodes(self.data, self.text, self.sample_size, self.dataname)
        self.node_index_list = [node_indices[idx] for idx in idx_list]

        if parameters['emb'][0]:
            self.creatembed(parameters['emb'][1])
        else:
            self.embeddings = None
            
            
    def logger(self,experName):
        self.experName = experName
        self.f = open('logresult/'+self.dataname+"__"+self.experName+'.log', 'a')
        sys.stdout = self.f

    def creatembed(self,index_file):
        if self.load_embedding:
            self.embeddings = Embeddings()
            self.embeddings.load(index_file)
        else:
            self.embeddings = self.createIndex(self.text)
            self.embeddings.save(index_file)
            
    def createIndex(self,alltext):
        alltxt = []
        for i in range(len(alltext['title'])):
            alltxt.append(alltext['title'][i]+"|| "+alltext['abs'][i])
        # Create embeddings model, backed by sentence-transformers & transformers
        embeddings = Embeddings(path="allenai/scibert_scivocab_uncased")
        # Index the list of te
        embeddings.index(alltxt)
        return embeddings
    
    
    def sample_test_nodes(self,data, text, sample_size):
        """
        Parameters:
            data:  data object.
            text: Textual information associated with nodes.
            sample_size (int): Number of test nodes to sample.
            dataset (str): Name of the dataset being used.
        """

        np.random.seed(42)
        test_indices = np.where(data.test_mask.numpy())[0]

        # Sample 2 times the sample size
        # node_indices = sample_test_nodes(data, 2 * sample_size)
        sampled_indices_double = np.random.choice(test_indices, size=2*sample_size, replace=False)

        # Filter out the indices of nodes with title "NA\n"
        sampled_indices = [node_idx for i, node_idx in enumerate(sampled_indices_double) 
                    if text['title'][node_idx] != "NA\n"]
        sampled_indices = sampled_indices[:sample_size]

        # sanity check
        count = 0
        for node_idx in sampled_indices:
            if text['title'][node_idx] == "NA\n":
                count += 1
        assert count == 0
        assert len(sampled_indices) == sample_size

        return sampled_indices