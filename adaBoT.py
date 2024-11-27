import itertools
from predata.datax import dataClassif,predres
from prompthub import DoT_classify,PST_proximity,REL_proximity
import json
import numpy as np
import random
import pickle
import dill
from contextlib import redirect_stdout
import time
import sys

def sub_list(a,b):
    return list(set(a) - set(b))

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
a1 = "(GOAL) The research objective, question, or task that this paper aims to address. "
a2 = "(METHOD) The research methodology, design or main method(s) proposed in the paper. "
a3 = "(IDEA) The overall idea, motivation or core concept of the paper. " 
a4 = "(THEORY/EXPER) The theory, experimental design, implementation, or tool proposed by the paper. "
aspect_dict = {
"GOAL": a1,
"METHOD": a2,
"IDEA": a3,
"THEORY/EXPER": a4
}
A1 = "GOAL"
A2 = "METHOD"
A3 = "IDEA"
A4 = "THEORY/EXPER"

def pair_combinations(lst,num=2):
    return [list(comb) for comb in itertools.combinations(lst, num)]

def flatten(two_d_list):
    flattened_list = [item for sublist in two_d_list for item in sublist]
    return flattened_list


## load exmaple pairs
class Boosterlar:
    def __init__(self,agent,retr=""):
        aspli = ["GOAL","METHOD","IDEA","THEORY/EXPER"]
        self.learners = {}
        allpairs2 = pair_combinations(aspli,2)
        allpairs3 = pair_combinations(aspli,3)
        for al in allpairs2:
            self.learners[al[0]+" "+al[1]] = {k: aspect_dict[k] for k in al}
        for al in allpairs3:
            self.learners[al[0]+" "+al[1]+" "+al[2]] = {k: aspect_dict[k] for k in al}
        self.dname = agent.dataname
        if self.dname == 'pubmed' or 'cora':
            self.dataset = dataClassif(agent.data,agent.text,agent.node_index_list,agent.dataname)
            self.examples = self.dataset.sample_citanode(retr)
            train_ids = [n.idx for n in flatten(self.examples.values())]
            self.train_list = list(set(train_ids))
            self.test_data = list(self.examples.keys())
        else:
            self.dataset = agent.pstdata
            self.train_list = self.dataset.node_index_list
            self.test_data = self.dataset.node_index_list
        self.n_estim = len(self.learners)
        self.alphas = []
        self.models= []
        self.learnlist = list(self.learners.keys())
        self.mid_res = []
        self.all_res = {}
        self.none_ratio = 0
        self.agent = agent

    def Boosting(self,lmodel,start,beta,name,learner_slice,ifsave,load_state,pre_state):
        self.NoneAlert = 0
        print("boost .... start")
        outputfl = name+"_"+lmodel+".txt"
        beta = beta
        if self.dname == 'relish':
            doter =  REL_proximity(self.agent)
        else:
            doter =  DoT_classify(self.dataset)
        if load_state==False:
            start = start
            train_data = np.array(self.train_list[:start]).astype(int)
            print("len train_data ...",len(train_data))
            n_samples = len(train_data)
            w = np.ones(n_samples)/n_samples
        else:
            start = pre_state["start"]
            train_data = pre_state["train_data"]
            print("len train_data ...",len(train_data))
            n_samples = len(train_data)
            w = pre_state["weight"]
            self.alphas = pre_state['alphas']
            self.models= pre_state['models']
            self.mid_res = pre_state['mid_res']
            self.all_res = pre_state['all_res']
        for key in learner_slice:
            learner = self.learners[key]
            doter.set_aspect(learner)
            keyname = key.replace("/","_")
            keyname = keyname.replace(" ","|")
            with open("out_terminal/"+keyname+outputfl,'w') as f,redirect_stdout(Tee(sys.stdout,f)):
                result = self.train_single_model(doter,train_data,lmodel)
            res = predres(result)
            self.mid_res.append(res)
            res.countacc()
            print("acc ********* ",doter.name,res.acc)
            wr = res.get_pred(train_data)
            error = np.dot(w,wr)/w.sum()
            print("error ********* ",error)

            alpha = np.log((1.0 - error) / (error + 1e-10))+np.log(self.dataset.nclass-1)
            indi = np.array([-1 if x==0 else x for x in wr])
            w*=np.exp(alpha*indi)
            w/=w.sum()
            remain_num = int((n_samples)*beta)
            add_num = n_samples-remain_num
            diffcult_indices = np.argsort(w)[-remain_num:]
            train_data = np.concatenate([train_data[diffcult_indices],self.train_list[start:start+add_num]]).astype(int)
            start += add_num
            w = np.concatenate([w[diffcult_indices],np.ones(add_num)*(1-beta)/n_samples])
            w/=w.sum()
            self.models.append(doter.name)
            self.alphas.append(alpha)
            self.all_res[doter.name] = res
            time.sleep(20)
        state = {"start":start,"train_data":train_data,"weight":w,"alphas":self.alphas,"mid_res":self.mid_res,
        "models":self.models,"all_res":self.all_res,"none":self.none_ratio,"error":error,"wr":wr}
        with open("msg/"+keyname+"_midstate.pkl",'wb') as f:
            dill.dump(state, f)
        boost_res = {'alphas':self.alphas,'models':self.models}
        if ifsave:
            with open('msg/'+name+'_boostresponse.pkl', 'wb') as f:
                dill.dump(self.all_res, f)
            with open('msg/'+name+'_boost_alpha.json',"w") as f:
                json.dump(boost_res,f)
        print(boost_res)
        return self.all_res,state
    def train_single_model(self,prompter,train_list,model):
        self.NoneAlert = 0
        boostdict = {}
        print("========================================")
        ### construct whole prompt
        for i,id_ in enumerate(train_list):
            if (i+1)%10==0:
                self.none_ratio=0
                time.sleep(1.5)
            expoints = []#
            prompter.set_idx(id_,expoints)
            if self.dname in ['cora','pubmed']:
                message = prompter.zero_shot()
                self.dataset.set_id(id_)
            else:
                message,label = prompter.zero_shot_aspect()
                self.dataset.set_id(id_,label)
            response = Olla.LLMsInfer(message,model,False)
            rt = self.dataset.verify(response)
            response = Olla.run_localmodel(message,["mistral-ai","qwen","mistral-nemo","phi3:14b"])
            results = {"message":message,"response":response,"ori_answ":rt[0],"ideal_answ":rt[1],"tf":rt[2]}
            boostdict[str(id_)] = results
        return boostdict
       
