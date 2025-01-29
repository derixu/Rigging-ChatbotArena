## Improving Your Model Ranking on Chatbot Arena by Vote Rigging

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
python vote_rigging.py
```
The default rigging strategy is Omni-BT, and we also support other rigging strategies specified by ```--rigging_mode```. Besides, you could set ```--classifier_acc``` to control the classification performance of de-anonymizing functions and set ```--beta``` to control the marginal probability of sampling the target model.
