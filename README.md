#### [Comparing Distributions by Measuring Differences that Affect Decision Making](https://openreview.net/forum?id=KB5onONJIAU):
What:
- New approach to measure discrepancy between distributions by calculating optimal loss for specific decision task
- Existing approaches: f-measure, IPM overlaps and extended by novel approach H-Divergence

Why to read:
-  new model for loss function analysis
-  learn about relevant algorithms on similar topic


#### [CycleMLP: A MLP-like Architecture for Dense Prediction](https://openreview.net/forum?id=NMEceG4v69Y):
What:
- New MLP-based architecture for Images
- Main features:
	- input can be various size
	- linear computational complexity using local windows

Why to read:
- See development of CV and MLP architectures
- Adding small portion of inductive bias helps a lot (Hierarchical Pierceiver)


#### [Language modeling via stochastic processes](https://openreview.net/forum?id=pMQwKL1yctf):
What:
- Use Stochastic process to mitigate problem in "classic" text generation
- Introduced Language model (Time Control) explicitly models latent structure with  
Brownian bridge dynamics learned using a novel contrastive objective 

Why to read:
- better text generation
- Seems fun

#### [Resolving Training Biases via Influence-based Data Relabeling](https://openreview.net/forum?id=EskfH0bwNVn):
What:
- Combine influence function with data relabeling for reduce training bias
- Proposed approach increase model's robustness to label noise

Why to read:
- Numerous aplication in Allegro

#### [Representational Continuity for Unsupervised Continual Learning](https://openreview.net/forum?id=9Hrka5PA7LW):
What:
- Connecting CL and Representation learning
- Propose impovement to UCL to fix catastrophics forgetting and  better qualitative intterpretations.

Why to read:
- Seems to be useful for practitioners (Allegro)



#### [A Fine-Grained Analysis on Distribution Shift](https://openreview.net/forum?id=Dl4LetuLdyK):
What:
- Framework of analysis distribution shift
- Comprehensive analysis of different methods

Why to read:
- See how different approaches can be used to mitigate distribution shift


#### [Weighted Training for Cross-Task Learning](https://openreview.net/forum?id=ltM1RMZntpu):
What:
- New algorithm for cross-tasks learning.
- Representation-based task distance depending on the quality of representations of each tasks.

Why to read:
- Comparison with other modern approaches
- Better domain adaptation pre-training for MailBERT


#### [F8Net: Fixed-Point 8-bit Only Multiplication for Network Quantization](https://openreview.net/forum?id=_CfpJazzXT2):
What:
- 8-bit fixed-point number is able to represent a wide range of values with negligible  
relative error
- New training approach unifying PACT and fixed-point quantization

Why to read:
- References to previous work in field
- Analysis and comparison to previous approaches


#### [Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation](https://openreview.net/forum?id=oapKSVM2bcj):
What:
- Implemented einops notation
- Comparison on CPU and CUDA

Why to read:
- einstein notation very convenient operation that simplify writing and reading code.
- PyTorch implementation of einstein operations is rather slow. 


#### [Finetuned Language Models are Zero-Shot Learners](https://openreview.net/forum?id=gEZrGCozdqR):
What:
- Instruction tuning - impove zero-shot learning capabilities via LM FT on various datasets in specific way. 

Why to read:
- Capability of zero-shot 
- See how we can combine SSL and FT with labeled data.


#### [Coordination Among Neural Modules Through a Shared Global Workspace](https://openreview.net/forum?id=XzTtHjgPDsT):
What:
- I don't understand. Some kind of environment for different models communication

Why to read:
- Interesting reference works
- New domain


#### [DISCOVERING AND EXPLAINING THE REPRESENTATION BOTTLENECK OF DNNS](https://openreview.net/forum?id=iRCUlgmdfHJ):
What:
- Discover of *representation bottleneck*
- Design loss functions to mitigate this phenomenon

Why to read:
- Intuition on differences between  DNN and human visual cognition


#### [Compositional Attention: Disentangling Search and Retrieval](https://openreview.net/forum?id=IwJPj2MBcIa):
What:
- Analysis of search/retrieval machanism  in MHA and highlighting problems.
- Solution to the problem: new attention mechanism, that disentangles search and retrieval

Why to read:
- Better understand attention mechanism 


#### [8-bit Optimizers via Block-wise Quantization](https://openreview.net/forum?id=shpkpVXzo3h):
What:
- Using dynamic & block-wise quantization, stable embadding layer new 8-bit optimizer has similar performance to 32-bit Adam by using only fraction of memory.
- Results obtained on multiple tasks 

Why to read:
- Can be directly replacement for 32-bit Adam without drop in metrics

#### [ViTGAN: Training GANs with Vision Transformers](https://openreview.net/forum?id=dwg5rXg1WS_):
What:
- Combining GAN and ViT gives results on par with CNN-based GANs
- Novel techniques regularize ViT.

Why to read:
- Just for fun


#### [Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models](https://openreview.net/forum?id=Nfl-iXa-y7R)
What:
- Using sparse matrix operation to effectively train NN
- Proposed method based on butterfly matrices and simplification useful to train on modern hardware.

Why to read:
- Learn about sparse model training
- To faster training of GPT-2 and ViT


#### [Linking Emergent and Natural Languages via Corpus Transfer](https://openreview.net/forum?id=49A1Y6tRhaq):
What:
- Analysis of EC and NL

Why to read:
- Learn about emergent communication
- just for fun


#### [Memorizing Transformers](https://openreview.net/forum?id=TrjbxzRcnf-):
What:
- Use non-differentiable memory and approximate kNN to add new information at inference time

Why to read:
- Learn more about memory in LM


#### [EntQA: Entity Linking as Question Answering](https://openreview.net/forum?id=US2rTP5nm_):
What:
- Solve EL as QA
- Invert problem: given document, search "questions" candidate and apply reader to get candidate mentions in document.

Why to read:
- Interesting problem formulation


#### [Strength of Minibatch Noise in SGD](https://openreview.net/forum?id=uorVGbWV5sw):
What:
- Theretical analysis of SGD noise with implication

Why to read:
- New methods: approximation of noise and using LinReg as minimal model for DL. 
- Comprehensive related works section
- Can be useful in practice - sec 6 ( but after review of the section I am not sure :) )


#### [The MultiBERTs: BERT Reproductions for Robustness Analysis](https://openreview.net/forum?id=K0E_F0gFDgA):
What:
- Analysis of BERT models for robustness using novel approach
- Some practical examples of using their method

Why to read:
- Method for quantifying uncertainty of experimental result can be useful


#### [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://openreview.net/forum?id=9Vrb9D0WI4):
What:
- T0

Why to read:
- Analysis of many NLP tasks 


#### [Tuformer: Data-driven Design of Transformers for Improved Generalization or Efficiency](https://openreview.net/forum?id=V0A5g83gdQ_)
What:
- MHSA analysis using tensor diagram
- method for flexible wieght tuning across heads based on data

Why to read:
- Interesting idea (at first glance)
