## Improving Your Model Ranking on Chatbot Arena by Vote Rigging
This repository contains the official implementation of [Improving Your Model Ranking on Chatbot Arena by Vote Rigging](https://arxiv.org/abs/2501.17858)


----
<div align=center><img src=pics/demo.png  width="80%" height="60%"></div>

### Initialize your rigging environment
Run the following command to set up your initial rigging environment, we separate the complete voting records into the historical votes (90%) and other users' votes (10%).
```cmd
python initial_env.py
```

### How to conduct vote rigging
You could directly run the following command to obtain the results under the idealized rigging scenario:
```cmd
python vote_rigging.py --rigging_mode omni_bt_diff
```
The default rigging strategy is Omni-BT, and we also support other rigging strategies specified by ```--rigging_mode```. Besides, you could set ```--classifier_acc``` to control the classification performance of de-anonymizing functions and set ```--beta``` to control the marginal probability of sampling the target model. If you want to explore the impact of concurrent votes from other users, you may run the following command:

```
python rigging_with_vo.py --rigging_mode omni_bt_diff
```

### How to train the multi-class classifier
First, switch to the [classifier](classifier) directory. To generate the training corpus, you could run the following example command that queries [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using the prompt from the [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) dataset:
```
python dataset_cur.py --output_dir hc3 --model_id meta-llama/Meta-Llama-3-8B-Instruct
```
With the prepared training corpus, try to run the following script to fine-tune a RoBERTa-based model:
```
python train.py --dataset hc3
```

### Defense against vote rigging
To detect malicious users, you can run the following command:
```
python detect_malicious_users.py --rigging_mode omni_bt_diff
```
For vote filtering, you can run the following command and specify the parameter ```--filter_threshold``` to control the filtering threshold.
```
python vote_filtering.py --rigging_mode omni_bt_diff --filter_threshold 0.8
```
