# LLM A* 

LLM A* is a novel algorithm that integrates traditional heuristic algorithms together with large language model (LLM), in order to achieving a lower search complexity and higher quality of planned path in comparison with data-drived path planning algorithms. Based on GPT-3.5-turbo interaction sessions, LLM A* is able to meet a near-A* degree in the light of the matrics of search complexity, path steps and maximum deviation times (MDT) under special obstacle grid-map environments with different sizes.


[arXiv](https://arxiv.org/abs/2312.01797) | supplyment

## Requirements

There are two necessary factors for running the experiment code: 
  1. Please ensure that openai (v0.28) is installed for the compatibility of GPT-3.5；
  ```
  pip install openai==0.28
  ```  
  2. An available OpenAI API key is required. (fix with the code on google colab)

## Running


## Potencial Problems
There are possibly two problems during running. When the problems below appear, please just re-run the code:
  1. For both LLM A* and LLM Greedy, an error with 'empty arg' is thrown out.
  2. For PPO, an error with [nan, nan] is thrown out.


Additionally, considering the fact that there is a certain degree of randomness in the result of LLM A*, LLM Greedy and PPO, it is normal for the reimplementation results to be different from those in the paper.
## BiblioTeX
