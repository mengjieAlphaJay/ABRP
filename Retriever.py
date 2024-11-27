from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.cluster import KMeans
import numpy as np
import random
import json
from rouge_score import rouge_scorer

class dp:
    def __init__(self,idx,hop,tflabel):
        self.idx = idx
        self.hop = hop
        self.tf_label = tflabel
        self.new_label = 'None'

    def set_label(self,nb):
        self.new_label = nb

class ClustRetr:
    def __init__(self,documents_dic):
        # 初始化SciBERT模型和分词器
        model_name = 'allenai/scibert_scivocab_uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.documents = list(documents_dic.values())
        self.keys = list(documents_dic.keys())
        self.embeddings = np.vstack([self.get_scibert_embeddings(doc) for doc in self.documents])
    def find_most_similar_documents(self,prek=0):
        # 创建一个结果字典
        result = {}
        num_docs = len(self.embeddings)

        self.similarity_matrix = np.zeros((num_docs, num_docs))
        
        # 计算余弦相似度并存储到矩阵中
        for i in range(num_docs):
            for j in range(i + 1, num_docs):  # 只计算上三角部分
                similarity = np.dot(self.embeddings[i], self.embeddings[j]) / (
                    np.linalg.norm(self.embeddings[i]) * np.linalg.norm(self.embeddings[j])
                )
                self.similarity_matrix[i][j] = similarity
                self.similarity_matrix[j][i] = similarity  # 对称性

        # 找到最相似的文档
        for i in range(num_docs):
            most_similar_ids = np.argsort(self.similarity_matrix[i])[::-1]  # 找到最大相似度的索引
            most_idx = most_similar_ids[prek]
            result[self.keys[i]] = self.keys[most_idx]
        if prek == 0:
            self.sim_dic = result
        else:
            self.sim_dic2 = result
        return result


    def find_most_rouge_documents(self):
        result = {}
        num_docs = len(self.keys)
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

        self.rouge_matrix = np.zeros((num_docs, num_docs))
        
        for i in range(num_docs):
            for j in range(i + 1, num_docs):  # 
                rsim = scorer.score(self.documents[i],self.documents[j])['rouge2'].recall
                self.rouge_matrix[i][j] = rsim
                self.rouge_matrix[j][i] = rsim  # 

        for i in range(num_docs):
            most_similar_idx = np.argmax(self.rouge_matrix[i])  # 
            result[self.keys[i]] = self.keys[most_similar_idx]
        self.rouge_dic = result
        return result

    # 
    def get_scibert_embeddings(self,text):
        # 
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # 获取输出
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings

    def cluster(self,k):
        self.n_clusters = k
        n_clusters = k  # 聚类的数量
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(self.embeddings)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

    def select_doc(self,top_percentile):
        def invert_dict(original_dict):
            inverted_dict = {}
            for key, values in original_dict.items():
                for value in values:
                    inverted_dict[value] = key
            return inverted_dict
        selected_documents = {}
        clust_dic = {}
        clost_idxlist = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels == i)[0]
            cluster_embeddings = self.embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_embeddings - self.centroids[i], axis=1)
            clost_idx = np.argmin(distances)
            clost_idxlist.append(self.keys[clost_idx])
            cutoff_distance = np.percentile(distances, top_percentile * 100)
            top_docs_indices = cluster_indices[distances <= cutoff_distance]
            print(len(top_docs_indices),"len ....")
            selected_documents[i] = [self.documents[idx] for idx in top_docs_indices]
            clust_dic[self.keys[clost_idx]] = [self.keys[idx] for idx in top_docs_indices]
        self.selected_documents = selected_documents#
        self.clust_dic = invert_dict(clust_dic)
        return self.clust_dic 
    
    def find_closest_documents(self,idx):
        target_vector = self.embeddings[idx] 
        closest_documents = []
        closest_indices = {}

        # Iterate over each cluster's selected documents using items from the dictionary
        for cluster_index, docs in self.selected_documents.items():
            # Retrieve the indices of the documents in the current cluster
            cluster_indices = [i for i, label in enumerate(self.labels) if label == cluster_index and self.documents[i] in docs]

            # Get the embeddings of selected documents in this cluster
            cluster_embeddings = self.embeddings[cluster_indices]

            # Calculate distances from the target vector to each selected document in the cluster
            distances = np.linalg.norm(cluster_embeddings - target_vector, axis=1)

            # Find the index of the minimum distance
            #min_index = np.argmin(distances)
            
            # Map the closest document to the cluster index in the dictionary
            #closest_documents.append(docs[min_index])

            #closest_indices[cluster_index] = cluster_indices[min_index]
            # Sort distances and get indices sorted by closest first
            sorted_indices = np.argsort(distances)

            # Find the index of the closest or second closest document based on the condition
            if cluster_indices[sorted_indices[0]] == idx and len(sorted_indices) > 1:
                min_index = sorted_indices[1]
            else:
                min_index = sorted_indices[0]
            closest_indices[cluster_index] = cluster_indices[min_index]  # Choose the closest
            closest_documents.append(docs[min_index])
        return closest_indices,closest_documents


class Retriever:
    ## for one node
    def __init__(self,agent):
        self.agent = agent
        self.data = agent.resources['data']
        self.text = agent.resources['text']
        self.node_index_list = agent.resources['node_list']
        self.embeddings = agent.resources['embeddings']
        self.maxp1 = agent.maxp1
        self.maxp2 = agent.maxp2
        self.shotnum = agent.shotnum

    def set_idx(self,nidx):
        #examples:dp list
        self.neibor_indices = []
        self.nidx = nidx
    
    def get_subgraph(self, edge_index, hop=1):
        current_nodes = torch.tensor([self.nidx])
        all_hops = []

        for _ in range(hop):
            mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
            
            # Add both the source and target nodes involved in the edges 
            new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))

            # Remove the current nodes to get only the new nodes added in this hop
            diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
            diff_nodes = torch.tensor(list(diff_nodes_set))  
            
            all_hops.append(diff_nodes.tolist())

            # Update current nodes for the next iteration
            current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))
        return all_hops
    
    def get_hop(self, hop):
        """
        Handle neighbors when attention is not used.
        Returns:
            str: String containing information about standard neighbors.
        """
        all_hops = self.get_subgraph(self.data.edge_index, hop)

        for h in range(0, hop):
            neighbors_at_hop = all_hops[h]
            neighbors_at_hop = np.array(neighbors_at_hop)
            neighbors_at_hop = np.unique(neighbors_at_hop)
            if h == 0:
                neighbors_at_hop = neighbors_at_hop[:self.maxp1]
            else:
                neighbors_at_hop = neighbors_at_hop[:self.maxp2]

            if len(neighbors_at_hop) > 0:
                self.neibor_indices = neighbors_at_hop
            else:
                self.neibor_indices = [self.nidx]
        return
    
    
    def embed_knn(self,k):
        #"pub_index"
        all_neighbor = self.embeddings.search(self.text['title'][self.nidx]+"|| "+self.text['abs'][self.nidx], k)
        self.neibor_indices = [all_neighbor[i][0] for i in range(k)]

    def get_embed(self):
        emb_neib = {}
        try:
            for n in self.node_index_list:
                emb_neib[n] = []
                self.set_idx(n)
                self.embed_knn(15)
                emb_neib[str(n)].extend(self.neibor_indices)
            with open('neibor/'+self.agent.dataname+'_embed.json','w') as f:
                json.dump(emb_neib,f,indent=4)
            return emb_neib
        except Exception as e:
            print("Error .. ",e)
            return emb_neib

    def cluster_closet(self,k):
        alltext = self.text
        closedic = {}
        alltxt = []
        for i in range(len(alltext['title'])):
            alltxt.append(alltext['title'][i]+"|| "+alltext['abs'][i])
        clustdoc = ClustRetr(alltxt)
        clustdoc.cluster(k)
        clustdoc.select_doc(0.5)
        for i,node in enumerate(self.node_index_list):
            closedic[node] = list(clustdoc.find_closest_documents(node)[0].values())
        with open('neibor/'+self.agent.dataname+'_cluster.json','w') as f:
            json.dump(closedic,f,indent=4)
        return closedic
    
    def select_dp4citation(self,candi):
        #candi:[[hop-1],[hop-2]]
        def has_label(idx):
            return (self.data.train_mask[idx] or self.data.val_mask[idx])
        labeled_data = []
        unlabeled_data_0 = []
        unlabeled_data_1 = []
        selection = []

        # 从candi[0]找带标签的数据点
        for idx in candi[0]:
            if has_label(idx):
                labeled_data.append(dp(idx,1,True))
            else:
                unlabeled_data_0.append(dp(idx,1,False))
            if len(labeled_data) == self.shotnum:
                return labeled_data  # 如果已经找够5个，则直接返回

        # 如果candi[0]中不足5个，尝试从candi[1]继续找
        hop2n = 0
        for idx in candi[1]:
            if has_label(idx):
                labeled_data.append(dp(idx,2,True))
                hop2n+=1
                ## dp from candi[1] must <=3
                if hop2n== int(self.shotnum/2)+1:
                    break
            else:
                unlabeled_data_1.append(dp(idx,2,False))
            if len(labeled_data) == self.shotnum:
                return labeled_data  # 如果找够5个，则返回

        # 如果仍未找够5个带标签的数据点，从剩余的未标记数据中随机选取足够的数据点
        needed = self.shotnum - len(labeled_data)
        selection.extend(labeled_data)

        # 先尝试从candi[0]的未标记数据中随机选取
        random.shuffle(unlabeled_data_0)
        selection.extend([i for i in unlabeled_data_0[:needed]])

        # 如果candi[0]的未标记数据不足，从candi[1]的未标记数据中补充
        if len(selection) < self.shotnum:
            needed = self.shotnum - len(selection)
            random.shuffle(unlabeled_data_1)
            selection.extend([ i for i in unlabeled_data_1[:needed]])

        return selection

    def select_citations(self,candi,shotnum,if_unlab=False):
        #candi:[[hop-1],[hop-2]]
        def has_label(idx):
            return (self.data.train_mask[idx] or self.data.val_mask[idx])
        labeled_data = []
        unlabeled_data_0 = []
        unlabeled_data_1 = []
        selection = []

        # 从candi[0]找带标签的数据点
        for idx in candi[0]:
            if has_label(idx):
                labeled_data.append(dp(idx,1,True))
            else:
                unlabeled_data_0.append(dp(idx,1,False))
            if len(labeled_data) == shotnum:
                return labeled_data  # 如果已经找够n个，则直接返回

        # 如果candi[0]中不足5个，尝试从candi[1]继续找
        for idx in candi[1]:
            if has_label(idx):
                labeled_data.append(dp(idx,2,True))
            else:
                unlabeled_data_1.append(dp(idx,2,False))
            if len(labeled_data) == shotnum:
                return labeled_data  # 如果找够n个，则返回
        needed = shotnum - len(labeled_data)
        selection.extend(labeled_data)

        if if_unlab:
            # 先尝试从candi[0]的未标记数据中随机选取
            random.shuffle(unlabeled_data_0)
            selection.extend([i for i in unlabeled_data_0[:needed]])

            if len(selection) < shotnum:
                needed = shotnum - len(selection)
                random.shuffle(unlabeled_data_1)
                selection.extend([ i for i in unlabeled_data_1[:needed]])
        return selection
    def convert(self,data):
        converted_data = {}
        for key, value in data.items():
            converted_sublist = []
            for arr in value:
                if isinstance(arr, np.ndarray):
                    converted_sublist.append(arr.tolist())
                else:
                    print('attention ... ',key,arr)
                    converted_sublist.append(arr)
            converted_data[str(key)] = converted_sublist
        return converted_data

    def get_hops(self,j):
        cit_nei = []
        self.set_idx(j)
        self.get_hop(1)
        cit_nei.append(self.neibor_indices)
        self.get_hop(2)
        cit_nei.append(self.neibor_indices)
        return cit_nei
        

    def get_citation_neib(self):
        cite_neib = {}
        for j in self.agent.node_index_list:
            cite_neib[j] = self.get_hops(j)
        citation_neib = convert(cite_neib)
        return citation_neib

    def select_citation_data(self,citation_neib):
        select_data = {}
        for id_ in self.node_index_list:
            self.set_idx(id_)
            points = self.select_dp4citation(citation_neib[str(id_)])
            select_data[id_] = points
        return select_data

    def get_all_citation(self):
        citation_neib = self.get_citation_neib()
        dp_cita = self.select_citation_data(citation_neib)
        try:
            self.agent.dumppkl(self.agent.dataname+"_shot"+str(self.shotnum)+"_citation"+".pkl",dp_cita,'neibor/')
            return dp_cita
        except Exception as e:
            print("Error .. ",e)
            return  dp_cita