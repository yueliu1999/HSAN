[stars-img]: https://img.shields.io/github/stars/yueliu1999/HSAN?color=yellow
[stars-url]: https://github.com/yueliu1999/HSAN/stargazers
[fork-img]: https://img.shields.io/github/forks/yueliu1999/HSAN?color=lightblue&label=fork
[fork-url]: https://github.com/yueliu1999/HSAN/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=yueliu1999.HSAN
[adgc-url]: https://github.com/yueliu1999/HSAN

# Hard Sample Aware Network

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://aaai.org/Conferences/AAAI-23/" alt="Conference">
        <img src="https://img.shields.io/badge/AAAI'23-brightgreen" /></a>
<p/>



[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]


An official source code for paper Hard Sample Aware Network for Contrastive Deep Graph Clustering, accepted by AAAI 2023. Any communications or issues are welcomed. Please contact yueliu19990731@163.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

-------------

### Overview

<p align = "justify"> 
We propose a novel contrastive deep graph clustering method dubbed Hard Sample Aware Network (HSAN) by introducing a comprehensive similarity measure criterion and a general dynamic sample weighing strategy. Concretely, in our algorithm, the similarities between samples are calculated by considering both the attribute embeddings and the structure embeddings, better-revealing sample relationships and assisting hardness measurement. Moreover, under the guidance of the carefully collected high-confidence clustering information, our proposed weight modulating function will first recognize the positive and negative samples and then dynamically up-weight the hard sample pairs while down-weighting the easy ones. In this way, our method can mine not only the hard negative samples but also the hard positive sample.
<div  align="center">    
    <img src="./assets/HSAN_model.png" width=80%/>
</div>


<div  align="center">    
      Figure 1: Illustration of the proposed Hard Sample Aware Network (HSAN).
</div>

### Requirements





### Quick Start

The code will be released soon. 



### Clustering Result

<div  align="center">    
    <img src="./assets/HSAN_result.png" width=100%/>
</div>

<div  align="center">    
    Table 1: Clustering results of our proposed HSAN and thirteen baselines on six datasets.
</div>




<div  align="center">    
    <img src="./assets/HSAN_tsne.png" width=100%/>
</div>

<div  align="center">    
    Figure 2: 2D <i>t</i>-SNE visualization of seven methods on two datasets.
</div>

### Citation

```

```








