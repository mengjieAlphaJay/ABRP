import random
import os
import numpy as np
import yaml
import json
a1 = "(GOAL) The research objective, question, or task that this paper aims to address;"
a2 = "(METHOD) The research methodology, design or main method(s) proposed in the paper."
a3 = "(IDEA) The overall idea, motivation or core concept of the paper." 
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

def generate_system_prompt(source, arxiv_style="subcategory", include_options=False):
    """
    Generate a system prompt based on the given content type and source.
    
    Args:
    - content_type (str): Specifies the type of content (e.g., title, abstract, neighbors).
    - source (str): Specifies the data source (e.g., arxiv, cora, pubmed, product).
    - use_original_arxiv (bool, optional): If set to True, a special prompt for 'arxiv' is used.
    
    Returns:
    - str: Generated system prompt.
    """

    categories = {
        'cora': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory", "Reinforcement Learning", "Probabilistic Methods"],
        'pubmed': ["Type 1 diabetes", "Type two diabetes", "Experimentally induced diabetes"]
    }

    #arxiv_prompts = generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping)
    
    prompts = {
        #'arxiv': arxiv_prompts[arxiv_style],
        #'arxiv_2023': arxiv_prompts[arxiv_style],
        'cora': "Please predict the most appropriate category for the paper. Choose from the following categories:\n\n{}",
        'pubmed': "Please predict the most likely type of the paper. Your answer should be chosen from:\n\n{}",
        'product': "Please predict the most likely category of this product from Amazon. Your answer should be chosen from the list:\n\n{}"
    }

    # Fetch the appropriate prompt
    prompt = prompts[source]

    if source in ['cora', 'pubmed']:
        categories_list = "\n".join(categories[source])
        return prompt.format(categories_list)
    else:
        return format(prompt)

class BoT_proximity:
    def __init__(self,agent):
        self.agent = agent
        self.abs_len = 5000#agent.abs_len
        self.main_papers = agent.main_papers
        self.pstdata = agent.pstdata
        self.prompt = {}
    def Example_create_prompting(self):
        userpt = """
*Step 1: Evaluation and Reasoning*
For each Reference Paper, you can evaluate and reason from the following aspects. When reasoning their relationship, you should identify the specific evaluated aspects, discuss how the Query's aspects either aligns with, diverges from, or just similar with the Reference:
(METHOD) Research Methodology: Assign 'm_score' {0, 1, 2} to assess whether it adopts or expands the new method M introduced in the Reference. 
(THEORY/EXPER) Theoretical Foundation/Experimental Design: Assign 'e_score' {0, 1, 2} to evaluate if the Query uses the new theory, experimental design, implementation, or tool proposed by the Reference.
(IDEA) Idea:  Assign 'i_score' {0, 1, 2} to evaluate if the Query's overall idea, motivation or core concept are 'clearly' inspired by the Reference's methods/ideas/concepts.
2 point: if Query is inspired/uses/adopts/extends the 'new' method/idea/theory/design/tool proposed by the Reference; 
1 point: if they are just similar but not the source, 0 point: if not original or dissimilar.

Attention!! If the label is a 'ref-source', then at least one aspect must score 2;
The response in the step should be JSON with key 'step1out'.
Format for Step 1 response:
Step-1:
Reference Paper-1 (ref-source):
     * METHOD:m_score:{0}; REASON:{} 
     * THEORY/EXPER:e_score:{0}; REASON:{}
     * IDEA:i_score:{0}; REASON:{}
Reference Paper-2 (not-ref-source)：
    ...
Reference Paper-3 (not-ref-source)：
    ...
Reference Paper-4 (not-ref-source)：
    ...
*Step 2: Comprehensive Analysis*
Analyze the reasoning and scoring process from Step 1. Compare and summarize the differences in scores, prioritizing the distinctions between scores of 2 and 1. You should also prioritize comparing between ref-source and not-ref-source, focusing on their specific aspects in relation to the Query.
Example for 'm_score',if you find a ref-source receives m_score of 2 and another reference earns a 1, compare these two and give this analysis:
    - ANALYSIS: The Query clearly the Transformer architecture based solely on attention mechanisms introduced in a Reference (ref-source), resulting in the m_score of 2. However, the method of Query is just similar to the self-attention mechanisms in another Reference but not directly derived from it. Consequently, the m_score is 1.
When making comparisons, you must mention the specific aspects being evaluated (for example, the methods used), and you can omit the sequence numbers of the reference (for instance, refer other references/the another reference, rather than saying Reference Paper-1). There is no need for a final summary,
In Step 2, the structure should be the following,
    - 'm_score':{Analysis & Reason}. Consequently, the m_score is {}. However, {Analysis & Reason}. Consequently, the m_score is {}.
The response in the step should be the JSON with three keys,"Analysis_m_score","Analysis_e_score","Analysis_i_score". Each analysis should not exceed 60 words.
{
'Analysis_m_score':{Analysis & Reason}. Consequently, the m_score is {}. However, {Analysis & Reason}. Consequently, the m_score is {}.
'Analysis_e_score':...
'Analysis_i_score':...
}
"""
        userpt+= "\nQuery Paper: "+ self.prompt['query_paper']
        for i,ref in enumerate(self.prompt['references']):
            userpt+= f"Reference Paper-{i+1}:"+ ref
        message = [{'role': 'system', 'content':self.system_prompt_Reasoning_v2()},
            {'role':'user', 'content': userpt}]
        return message

    def set_aspect(self,aspects):
        self.aspects = aspects
        keys = aspects.keys()
        self.keys = keys
        self.name = "|".join(keys)
        self.name = self.name.replace("/","_")
        self.name = self.name.replace(" ","-")


    def system_prompt_zero_EnglishBase_simply(self):
        strpt = """Your task is to identify the source of paper ('ref-sources') from a given paper. Whether a reference qualifies as a “ref-source” is based on one of the following criteria:
1. Does the main idea of paper p draw inspiration from the reference? 2. Is the fundamental methodology of paper p derived from the reference? 
Namely, is the reference indispensable to paper p? Without the contributions of the reference, would the completion of paper p be impossible?
Please evaluate whether the references of the given Query Paper qualify as ref-source, assigning scores.
"""
        return strpt
    def set_idx(self,nidx,refids,examples):
        #examples:dp list
        self.nidx = nidx
        self.prompt['query_paper'] = self.concatPaper(self.nidx,'main',False)   
        self.prompt['ref_labels'] = [self.pstdata.get_type(self.nidx,ref) for ref in refids]
        self.prompt['references'] = [self.concatPaper(ref,sets,False,False) for sets,ref in zip(self.prompt['ref_labels'],refids)]
        self.examples = examples

    def COT_DCOM_promting_Examp(self):
        userpt = "For each Reference Paper, determine whether it is the source paper of the Query Paper. You can evaluate and reason from the following aspects. When reasoning their relationship, you should identify the specific evaluated aspects, discuss how the Query's aspects either aligns with, diverges from, or just similar with the Reference:\n\n"
        p1 = "(METHOD) Research Methodology: Assign 'm_score' {0, 1, 2} to assess whether it adopts or expands the new method M introduced in the Reference.\nExample:"+self.examples["m_score"]+"\n\n"
        p2 = "(THEORY/EXPER) Theoretical Foundation/Experimental Design: Assign 'e_score' {0, 1, 2} to evaluate if the Query uses the new theory, experimental design, implementation, or tool proposed by the Reference.\nExample:"+self.examples["e_score"]+"\n\n"
        p3 = "(IDEA) Idea:  Assign 'i_score' {0, 1, 2} to evaluate if the Query's overall idea, motivation or core concept are 'clearly' inspired by the Reference's methods/ideas/concepts.\nExample:"+self.examples["i_score"]+"\n\n"
        p4 = "(GOAL) Research Goal: Assign 'g_score' {0, 1, 2} to evaluate if the Query's research objective or task is proposed by the Reference.\n\n"
        key2prompt = {"METHOD":p1,"THEORY/EXPER":p2,"IDEA":p3,"GOAL":p4}
        key2score = {"METHOD":"2*m_score","THEORY/EXPER":"e_score","IDEA":"i_score","GOAL":"g_score"}
        for key in self.keys:
            userpt += key2prompt[key]
    
        userpt+= "Finally, combine these scores for a final assessment,"
        if A2 in self.keys:
            userpt+= "with a greater weight on METHOD (m_score): FINAL_SCORE = "
        else:
            userpt+="FINAL_SCORE = "
        for key in self.keys:
            userpt += key2score[key]
            userpt += " + "
        userpt = userpt[:-3]
        userpt+="""\nThe response should follow the format: 
    Reference Paper-1:
    - REASON FOR ASPECTS：
        * METHOD:m_score:{0}; REASON:{} 
        * ...
        * ... 
    - FINAL_SCORE:{0}
    Reference Paper-2：
    ...
    Reference Paper-3：
    ...\n"""
        #userpt+=self.example+'\n\n'
        userpt+= "Query Paper: "+ self.prompt['query_paper']
        for i,ref in enumerate(self.prompt['references']):
            userpt+= f"Reference Paper-{i+1}:"+ ref
        message = [{'role': 'system', 'content':self.system_prompt_zero_EnglishBase_simply()},
            {'role':'user', 'content': userpt}]
        return message
    def COT_DCOM_promting_base(self):
        userpt = "For each Reference Paper, determine whether it is the source paper of the Query Paper. You can evaluate and reason from the following aspects. When reasoning their relationship, you should identify the specific evaluated aspects, discuss how the Query's aspects either aligns with, diverges from, or just similar with the Reference:\n"
        p1 = "(METHOD) Research Methodology: Assign 'm_score' {0, 1, 2} to assess whether it adopts or expands the new method M introduced in the Reference.\n\n"
        p2 = "(THEORY/EXPER) Theoretical Foundation/Experimental Design: Assign 'e_score' {0, 1, 2} to evaluate if the Query uses the new theory, experimental design, implementation, or tool proposed by the Reference.\n\n"
        p3 = "(IDEA) Idea:  Assign 'i_score' {0, 1, 2} to evaluate if the Query's overall idea, motivation or core concept are 'clearly' inspired by the Reference's methods/ideas/concepts.\n\n"
        p4 = "(GOAL) Research Goal: Assign 'g_score' {0, 1, 2} to evaluate if the Query's research objective or task is proposed by the Reference.\n\n"
        key2prompt = {"METHOD":p1,"THEORY/EXPER":p2,"IDEA":p3,"GOAL":p4}
        key2score = {"METHOD":"2*m_score","THEORY/EXPER":"e_score","IDEA":"i_score","GOAL":"g_score"}
        for key in self.keys:
            userpt += key2prompt[key]
    
        userpt+= "Finally, combine these scores for a final assessment,"
        if A2 in self.keys:
            userpt+= "with a greater weight on METHOD (m_score): FINAL_SCORE = "
        else:
            userpt+="FINAL_SCORE = "
        for key in self.keys:
            userpt += key2score[key]
            userpt += " + "
        userpt = userpt[:-3]
        userpt+="""\nThe response should follow the format: 
    Reference Paper-1:
    - REASON FOR ASPECTS：
        * METHOD:m_score:{0}; REASON:{} 
        * ...
        * ... 
    - FINAL_SCORE:{0}
    Reference Paper-2：
    ...
    Reference Paper-3：
    ...\n"""
        #userpt+=self.example+'\n\n'
        userpt+= "Query Paper: "+ self.prompt['query_paper']
        for i,ref in enumerate(self.prompt['references']):
            userpt+= f"Reference Paper-{i+1}:"+ ref
        message = [{'role': 'system', 'content':self.system_prompt_zero_EnglishBase_simply()},
            {'role':'user', 'content': userpt}]
        return message

    def concatPaper(self,pid,sets,if_lable=False):
        if sets == 'main':
            title = self.main_papers[pid].title
            abstract = self.main_papers[pid].abs
            prompt_str = ""
        else:
            if if_lable:
                paper = self.pstdata.get_paperinfo(pid)
                if sets=='trace':
                    prompt_str = "(Label: ref-source)"
                else:
                    prompt_str = "(Label: not-ref-source)"
            else:
                prompt_str = ""
                paper = self.pstdata.get_paperinfo(pid)
            title = paper['title']
            abstract = paper['abs']
        prompt_str += f"<Title: {title}||| "
        prompt_str = prompt_str+f"Abstract: {abstract[:self.abs_len]}>\n" 
        return prompt_str
    
    
    
class BoT_classify:
    def __init__(self,datax):
        self.abs_len = 2200
        self.datax = datax
        self.sys_prompt = generate_system_prompt(self.datax.dataname)

    def set_aspect(self,aspects):
        self.aspects = aspects
        keys = aspects.keys()
        self.keys = keys
        self.name = "|".join(keys)
        self.name = self.name.replace("/","_")
        self.name = self.name.replace(" ","-")
    def set_idx(self,nidx,examples):
        self.nidx = nidx
        self.examples = examples
        self.query = self.concatPaper(self.nidx,'No Category',False)

    def concatPaper(self,nidx,label,if_lable=False):
        title = self.datax.get_data(nidx,'title') #self.text['title'][nidx]
        prompt_str = f"<Title: {title[6:]}.   "
        # Include abstract if required
        abstract = self.datax.get_data(nidx,'abs')
        prompt_str = prompt_str+f"Abstract: {abstract[10:self.abs_len]}>\n" 
        if if_lable:
           prompt_str +=  f"Category: {label}\n"
        return prompt_str

    def zero_shot(self):
        userpt = """First, extract the following aspects of the paper. Then, for each aspect deduce the REASONING process using the extracted information. When reasoning, you can also optionally refer to information from neighbor papers. The hop-1 neighbor directly cites or is cited by the query paper. hop-2 neighbors would include citations of the citations.\n"""
        for i,key in enumerate(self.keys):
            userpt += "  "+(str(i+1)+". "+aspect_dict[key])+ f"  First, extract the {key}, then give the reasoning process.\n"
        userpt += "Finally, integrate these aspects above and the relevant papers, determine the Category for the Query Paper.\n\n"
        userpt+= ("""If there isn't enough information to extract specific aspetcs, make an educated guess based on the title and provide reasoning.\n\nThe response should follow the format: \n""")
        userpt+= "ASPECTS & REASONING:\n  1. METHOD & Reason: {};\n  2. ...;"
        userpt += "\nFINAL_reasoning: {fianl_reason}"+"\nCategory:{}\n\n"
        userpt +=  "The Query paper has the following neibor papers: \n"+self.concatExamples()
        userpt += "\nQuery Paper:"+self.query+"\n"
        return [{'role':'system', 'content': self.sys_prompt}, {'role':'user', 'content': f"{userpt}"}]

    def few_shot(self,reason):
        reason = {k.lower(): v for k, v in reason.items()}
        userpt = """First, extract the following aspects of the paper. Then, for each aspect deduce the REASONING process using the extracted information. When reasoning, you can also optionally refer to information from neighbor papers. The hop-1 neighbor directly cites or is cited by the query paper. hop-2 neighbors would include citations of the citations.\n"""
        for i,key in enumerate(self.keys):
            userpt += "  "+(str(i+1)+". "+aspect_dict[key])+ f"  First, extract the {key}, then give the reasoning process.\n"
            try:
                userpt += "- Example: "+reason[key.lower()]+"\n\n"
            except:
                userpt += "\n"
        userpt += "Finally, integrate these aspects above and the relevant papers, determine the Category for the Query Paper.\n"
        userpt += "- Example: "+reason['final']+" Moreover, there are also papers in the neighbor that are consistent with this category. \n\n"
        userpt+= ("""If there isn't enough information to extract specific aspetcs, make an educated guess based on the title and provide reasoning.\n\nThe response should follow the format: \n""")
        userpt+= "ASPECTS & REASONING:\n  1. GOAL & Reason: {};\n  2. ...;"
        userpt += "\nFINAL_reasoning: {fianl_reason}"+"\nCategory:{}\n\n"
        userpt +=  "The Query paper has the following neibor papers: \n"+self.concatExamples()
        userpt += "\nQuery Paper:"+self.query+"\n"
        #prompt+= "ASPECTS & REASONING:\nFINAL_reasoning:\nCategory:"
        return [{'role':'system', 'content': self.sys_prompt}, {'role':'user', 'content': f"{userpt}"}]

        userpt_reason = ""
        for i,key in enumerate(self.keys):
            userpt_reason += "  "+(str(i+1)+"."+self.aspects[key])+"\n"#+ "First, extract the information. Based on it, give the reason why it falls the category.\n"
        prompt = """First, extract the following aspects of the paper. Then, for each aspect, articulate the reasoning process that supports the determination of category using the extracted information and the given category.\n""" 
        prompt += userpt_reason
        #1. (GOAL) The research objective, question, or task that this paper aims to address; Then give the reason why the goal falls the category.
        #2. (METHOD) The research methodology, design or main method(s) used in the paper. Then give the reason why the method falls the category.
        prompt += """You should extract the information and give reasoning for each aspect (e.g.,'METHOD-reason','GOAL-reason'). If there isn't enough information to extract a METHOD or GOAL, make an educated guess based on the title and provide reasoning.
The structure of reasoning process is {Extracted Information}. {REASON supporting the Category};
    - Example for 'METHOD-reason': The paper proposes a subsymbolic approach to natural language processing using neural networks. The new method falls under the category of Neural Networks. Additionaly, some neighboring papers also use neural networks for NLP tasks, which supports the credibility of this classification.
Next, integrate the two aspects above and give the final reasoning process.
The response should be the JSON with three keys,\n{\n"""
        pt = ""
        for key in self.keys:
            pt += f"{key}_Reason: "+" {Extracted Information}. {REASON supporting the Category}; (limit to 40 words)\n"
        prompt += pt
        prompt += """FINAL_reasoning:""\n}\n"""
        prompt += '\nQuery paper : '+ self.query + 'Category: '+ self.datax.get_data(self.nidx,'label')
        sys_prompt = "Given a paper and its label, please provide the reasoning process that supports the determination of the label."
        #articulate the diagnostic reasoning process that supports the sentiment determination of the input.
        message = [{'role': 'system', 'content':sys_prompt},{'role':'user', 'content': prompt}]
        return message


    def concatExamples(self):
        init_str = ""
        examples = self.examples[:4]
        for i, nei in enumerate(examples):
            init_str+= f"(hop-{nei.hop})"+self.concatPaper(nei.idx,self.datax.get_data(nei.idx,'label'),if_lable=nei.tf_label)
        return init_str