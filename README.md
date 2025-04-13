# LLM A* 

LLM A* is a novel algorithm that integrates traditional heuristic algorithms together with large language model (LLM), in order to achieving a lower search complexity and higher quality of planned path in comparison with data-drived path planning algorithms. Based on GPT-3.5-turbo interaction sessions, LLM A* is able to meet a near-A* degree in the light of the matrics of search complexity, path steps and maximum deviation times (MDT) under special obstacle grid-map environments with different sizes.

## Notes>_<
Our project was initially completed and implemented on Google Colab. We are currently continuously updating and improving the source code and the setup documents. If you have difficulty installing the source code, please run the Ipynb file first. Thank you!


[arXiv](https://arxiv.org/abs/2312.01797) | [supplement](https://stummuac-my.sharepoint.com/personal/55141653_ad_mmu_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F55141653%5Fad%5Fmmu%5Fac%5Fuk%2FDocuments%2FFaculty%2Ddoc%2FResearch%2FPeng%2DWang%2FIROS%5F2024%2FSupplemental%20Materials%20to%20%20LLM%20A%20Human%20in%20the%20Loop%20Large%20Language%20Models%20Enabled%20A%20Star%20Search%20for%20Robotics%2Epdf&parent=%2Fpersonal%2F55141653%5Fad%5Fmmu%5Fac%5Fuk%2FDocuments%2FFaculty%2Ddoc%2FResearch%2FPeng%2DWang%2FIROS%5F2024&ga=1)

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

```
  @misc{xiao2023llm,
        title={LLM A*: Human in the Loop Large Language Models Enabled A* Search for Robotics}, 
        author={Hengjia Xiao and Peng Wang},
        year={2023},
        eprint={2312.01797},
        archivePrefix={arXiv},
        primaryClass={cs.RO}
  }
```
