#!/usr/bin/env python
# coding: utf-8

# # Influential data identification - Llama2 - Sentence
# 
# This notebook demonstrates how to efficiently compute the influence functions using DataInf, showing its application to **influential data identification** tasks.
# 
# - Model: [llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) trained on a mix of publicly available online datasets.
# - Fine-tuning dataset: Synthetic Sentence transformation dataset
# 
# References
# - `trl` HuggingFace library [[Link]](https://github.com/huggingface/trl).
# - DataInf is available at this [ArXiv link](https://arxiv.org/abs/2310.00902).

# In[1]:


import sys
sys.path.append('../src')
from lora_model import LORAEngineGeneration
from influence import IFEngineGeneration

import warnings
warnings.filterwarnings("ignore")


# ## Fine-tune a model
# - We fine-tune a llama-2-13b-chat model on the `sentence transformation` dataset. We use `src/sft_trainer.py`, which is built on HuggingFace's [SFTTrainer](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py). It will take around 30 minutes.

# In[2]:


"""
python src/sft_trainer.py \
    --model_name /workingdir/models_hf/llama7b-chat \
    --dataset_name datasets/grammars_train.hf \
    --output_dir /workingdir/models_hf/grammars_13bf \
    --dataset_text_field text \
    --load_in_8bit \
    --use_peft
"""

# !python /YOUR-DATAINF-PATH/DataInf/src/sft_trainer.py \
#     --model_name /YOUR-LLAMA-PATH/llama/models_hf/llama-2-13b-chat \
#     --dataset_name /YOUR-DATAINF-PATH/DataInf/datasets/grammars_train.hf \
#     --output_dir /YOUR-DATAINF-PATH/DataInf/models/grammars_13bf \
#     --dataset_text_field text \
#     --load_in_8bit \
#     --use_peft


# ## Load a fine-tuned model

# Please change the following objects to  "YOUR-LLAMA-PATH" and "YOUR-DATAINF-PATH"
base_path = "/workingdir/models_hf/llama7b-chat"
project_path ="/nethome/yjin328/Workspace/NLP/DataInf"
adapter_path = "/workingdir/models_hf/grammars_7b"
lora_engine = LORAEngineGeneration(base_path=base_path, 
                                   project_path=project_path,
                                   adapter_path=adapter_path,
                                   dataset_name='grammars')


# ## Compute the gradient
#  - Influence function uses the first-order gradient of a loss function. Here we compute gradients using `compute_gradient`
#  - `tr_grad_dict` has a nested structure of two Python dictionaries. The outer dictionary has `{an index of the training data: a dictionary of gradients}` and the inner dictionary has `{layer name: gradients}`. The `val_grad_dict` has the same structure but for the validationd data points. 

import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)


# ## Compute the influence function
#  - We compute the inverse Hessian vector product first using `compute_hvps()`. With the argument `compute_accurate=True`, the exact influence function value will be computed. (it may take an hour to compute).
# <!--  - Here, we take a look at the first five validation data points. -->

# In[6]:


influence_engine = IFEngineGeneration()
influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
influence_engine.compute_hvps()
influence_engine.compute_IF()


# ## Attributes of influence_engine
# There are a couple of useful attributes in `influence_engine`. For intance, to compare the runtime, one can use `time_dict`.

# In[7]:


influence_engine.time_dict


# In[8]:


influence_engine.IF_dict.keys()


# ## Application to influential data detection task
# - We inspect the most and the least influential data points for validation data loss. Here, the most (and the least) influential data points are determined by the absolute value of influence function values.
# - Why? the least influential data points will have near zero values, which means the training data point does not affect the validation loss. 

# In[9]:


most_influential_data_point_proposed=influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmax(), axis=1)
least_influential_data_point_proposed=influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmin(), axis=1)


# In[10]:


val_id=0
print(f'Validation Sample ID: {val_id}\n', 
      lora_engine.validation_dataset[val_id]['text'], '\n')
print('The most influential training sample: \n', 
      lora_engine.train_dataset[int(most_influential_data_point_proposed.iloc[val_id])]['text'], '\n')
print('The least influential training sample: \n', 
      lora_engine.train_dataset[int(least_influential_data_point_proposed.iloc[val_id])]['text'])


# # AUC and Recall 

# In[11]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

identity_df=influence_engine.IF_dict['identity']
proposed_df=influence_engine.IF_dict['proposed']

n_train, n_val = 900, 100
n_sample_per_class = 90 
n_class = 10

identity_auc_list, proposed_auc_list=[], []
for i in range(n_val):
    gt_array=np.zeros(n_train)
    gt_array[(i//n_class)*n_sample_per_class:((i//n_class)+1)*n_sample_per_class]=1
    
    # The influence function is anticipated to have a big negative value when its class equals to a validation data point. 
    # This is because a data point with the same class is likely to be more helpful in minimizing the validation loss.
    # Thus, we multiply the influence function value by -1 to account for alignment with the gt_array. 
    identity_auc_list.append(roc_auc_score(gt_array, -(identity_df.iloc[i,:].to_numpy())))
    proposed_auc_list.append(roc_auc_score(gt_array, -(proposed_df.iloc[i,:].to_numpy())))
    
print(f'identity AUC: {np.mean(identity_auc_list):.3f}/{np.std(identity_auc_list):.3f}')
print(f'proposed AUC: {np.mean(proposed_auc_list):.3f}/{np.std(proposed_auc_list):.3f}')


# In[12]:


# Recall calculations
identity_recall_list, proposed_recall_list=[], []
for i in range(n_val):
    correct_label = i // 10

    # Similar to AUC computation, we consider the first 90 data points with the smallest influence function values 
    # These data points with the smallest influence function values likely have the same class with the validation data point.
    sorted_labels = np.argsort(identity_df.iloc[i].values)// 90 # ascending order
    recall_identity = np.count_nonzero(sorted_labels[0:90] == correct_label) / 90.0
    identity_recall_list.append(recall_identity)
    
    sorted_labels = np.argsort(proposed_df.iloc[i].values)// 90 # ascending order
    recall_proposed = np.count_nonzero(sorted_labels[0:90] == correct_label) / 90.0
    proposed_recall_list.append(recall_proposed)
    
print(f'identity Recall: {np.mean(identity_recall_list):.3f}/{np.std(identity_recall_list):.3f}')
print(f'proposed Recall: {np.mean(proposed_recall_list):.3f}/{np.std(proposed_recall_list):.3f}')


# In[ ]:





# In[ ]:




