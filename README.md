# LLM A* 



[arXiv](https://arxiv.org/abs/2312.01797) | supplyment
## Requirements
There are two necessary factors for running the experiment code: 
  1. Please ensure that openai (v0.28) is installed for the compatibility of GPT-3.5；
  ```
  pip install openai==0.28
  ```  
  2. An available OpenAI API key is required.
## Running


## Potencial Problems
This is the code of primary version so that there are possibly two problems during running. When the problems below appear, please just re-run the code:
  1. For both LLM A* and LLM Greedy, an error with 'empty arg' is thrown out.
  2. For PPO, an error with [nan, nan] is thrown out.


Additionally, considering the fact that there is a certain degree of randomness in the result of LLM A*, LLM Greedy and PPO, it is normal for the reimplementation results to be different from those in the paper.

## BiblioTeX
