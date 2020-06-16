# AtNE-Trust
Code implementation of the paper: AtNE-Trust: Attributed Trust Network Embedding for Trust Prediction in Online Social Networks, which has been submitted to ICDM 2020 for blind review. AtNE-Trust
# AtNE-Trust: Attributed Trust Network Embedding for Trust Predicrion in Online Social Networks


## Basic information
- Version: 1.0
- Date: 11 June 2020


## Overview
This package is an simple implementation of the paper "AtNE-Trust: Attributed Trust Network Embedding for Trust Predicrion
in Online Social Networks" submitted to ICDM 2020.


## Requirements
 - python 3.7
 - tenforflow 1.14.0
 - numpy (We recommend you to use [Anaconda](https://anaconda.org/anaconda/numpy).)
 

## Usage
### Input
- Input graph files: (trustor, trustee, trust-value) trust-value is 0 or 1.
  The input data is stored in database in our code, so we read the input data from database directly. It is easily to be 
  changed if other data files is needed.

- Input embedding set:
  Four different views of data are pre-generated and stored. And then we can use the generated embeddings of different views 
  in the AtNE-Trust code.

### How to run
 1. Run trust network embedding to obtain pre-generated trust network embedding.
   (1) Complie .cc codes
       For mac users, please execute a command: ``make mac``.
       For linux users, please execute a command: ``make linux``.
   (2) Give parameters that you want (Optional)
 2. Run MF, doc2vec to obtain pre-generated attributes embedding
 3. Save embedding files.
 3. Run atne-trust.
 
Note: In our code, this is a version that employ attributes embedding for trust prediction, it is easy to add the 
"trust network embedding" view to obtain the full code. We make our code such way due to that different methods may have different 
ways to deal with network embedding. So we want to provide a framework-like code for variation convenience.

## Baselines:
ASNE: https://github.com/lizi-git/ASNE
SIDE: https://datalab.snu.ac.kr/side/
