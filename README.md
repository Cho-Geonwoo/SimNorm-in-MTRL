<p align="center" width="100%">
</p>

<div id="top" align="center">

Evaluating Simplicial Normalization in Multi-Task Reinforcement Learning
-----------------------------

<!-- **Authors:** -->

_**Geonwoo Cho<sup>1</sup>, Subi Lee<sup>1</sup>, Jaemoon Lee<sup>2</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> Gwangju Institute of Science and Technology,
<sup>2</sup> Seoul National University

</div>


## Overview

Multi-task reinforcement learning (MTRL) addresses key limitations of existing reinforcement learning (RL) methods, notably in generalization and sample efficiency. However, managing multiple tasks simultaneously remains a significant challenge. Various approaches, including curriculum learning, the mixture of experts, and parameter-sharing strategies, have been explored to improve MTRL performance. On the other hand, one of the recent research suggests that Simplicial Normalization (SimNorm), rather than ReLU, is an effective activation function for modeling the objective function on single task RL. In this paper, we investigate whether this claim extends to MTRL. We conducted experiments on two types of agents—one using ReLU and the other using SimNorm—within the Meta-world environments, comparing their total return and success rates. Our findings show that SimNorm appears to underperform compared to ReLU in the MTRL environments.



## Quick Start

Download the dataset MT50 via this [Google Drive link](https://drive.google.com/drive/folders/1Ce11F4C6ZtmEoVUzpzoZLox4noWcxCEb).

When your environment is ready, you could run the following script:
``` Bash
python main.py --seed 123 --data_path ./MT50 --prefix_name MT5 # MT30, MT50
```


## Acknowledgments

This repo benefits from [HarmoDT](https://github.com/charleshsc/HarmoDT) and [PWM](https://github.com/imgeorgiev/PWM). Thanks for their wonderful works!