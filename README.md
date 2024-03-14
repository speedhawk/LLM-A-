# LLM A* 

LLM A* is a novel algorithm that integrates traditional heuristic algorithms together with large language model (LLM), in order to achieving a lower search complexity and higher quality of planned path in comparison with data-drived path planning algorithms. Based on GPT-3.5-turbo interaction sessions, LLM A* is able to meet a near-A* degree in the light of the matrics of search complexity, path steps and maximum deviation times (MDT) under special obstacle grid-map environments with different sizes.


[arXiv](https://arxiv.org/abs/2312.01797) | supplyment

## Requirements & Running

We saved each codenote of LLM A*, LLM Greedy and PPO on GoogleColab as a corresponding gist that can be immediately run. However, an available OpenAI API key is required in LLM-based algorithms. (fix with the code on google colab) please change the code below with your API key before running:


```
self.openai_key = 'YOUR_KEY'
```
Moreover, please be aware about the path which should be self-defined in your Google drive. 
For LLM-based algorithms, you should be care about:
  The path saving environments
For PPO, you should be extraly care about:
  1. path of statistic data (about the score and steps)
  2. path of network checkpoints

  Especially, you can manually modify the number in 'episodes.txt' to 0 when starting a new training.

## Potencial Problems
There are possibly two problems during running. When the problems below appear, please just re-run the code:
  1. For both LLM A* and LLM Greedy, an error with 'empty arg' is thrown out.
  2. For PPO, an error with [nan, nan] is thrown out.


Additionally, considering the fact that there is a certain degree of randomness in the result of LLM A*, LLM Greedy and PPO, it is normal for the reimplementation results to be different from those in the paper. 

## BiblioTeX
