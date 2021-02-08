# Overview
Some basic Code implementation of the paper: Capturing Rich Relations of Users and Items with Multi-Graph Attention Networks for Social Recommendation, which has been submitted to KDD 2021 for blind review. The full code will be released once the paper is accepted.
# AtNE-Trust: Attributed Trust Network Embedding for Trust Predicrion in Online Social Networks


## Basic information
- Version: 1.0
- Date: 11 Feburary 2021


## Requirements
 - python ==3.7
 - Pytorch ==1.1.0
 - numpy (We recommend you to use [Anaconda](https://anaconda.org/anaconda/numpy).)
 

## Usage
### Input
- Input graph files: (trustor, trustee, trust-value) trust-value is 0 or 1.
                     (user, item, rating)
                     (item, item, link)


## Baselines:
Traditional Social Recommendations: https://github.com/hongleizhang/RSAlgorithms
GraphRec: https://github.com/wenqifan03/GraphRec-WWW19
DANSER: https://github.com/qitianwu/DANSER-WWW-19
