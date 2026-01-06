# Previous Research

This repository contains a collection of research papers developed during my Master's program at NYU, focusing on  topics in Machine Learning, Deep Learning, Natural Language Processing, Information Retrieval, and Intelligent Systems. Each project explores fundamental challenges in AI and demonstrates practical applications of advanced computational techniques to real world problems at scale.

## Research Papers

### 1. Deep Residual Learning: Analyzing Model Performance on CIFAR-10 Images

**Paper:** [example_1.pdf](example_1.pdf)  
**Repository:** https://github.com/aaronbengochea/cifar10-evaluation

This research conducts a comprehensive study of residual convolutional neural networks on the CIFAR-10 classification task, exploring the performance characteristics of shallow, mid-depth, and deep residual architectures ranging from 10 to 102 layers. The main objective is to investigate how variations in network depth affect convergence behavior, overfitting, and overall classification accuracy while focusing on the BasicBlock design from ResNet18. The study analyzes the impact of critical hyperparameters including optimizer selection (SGD vs. ADAM), learning rate scheduling, and regularization techniques through pre-training and training-time image augmentations. By evaluating ResNet10, ResNet40, and ResNet102 variants, the research aims to determine the optimal balance between model capacity and generalizability for networks with fewer than 5M trainable parameters. The findings provide practical guidelines for efficiently training deep residual networks in resource-constrained scenarios, revealing that mid-depth architectures (ResNet40) achieve the best trade-off between accuracy and computational efficiency on hidden test sets.

### 2. Exploring White Box Attack Design and Its Transferability to Black Box Settings

**Paper:** [example_2.pdf](example_2.pdf)  
**Repository:** https://github.com/timothycao/adversarial-attacks

This research presents a systematic study of adversarial attacks on pre-trained ImageNet classifiers, focusing on ResNet-34 and DenseNet-121 models. The main objective is to explore the design of white-box adversarial attacks and assess their transferability to black-box settings by implementing and comparing multiple attack variants under controlled conditions. The study begins by establishing baseline accuracies on a curated 500-image subset from ImageNet-1K, then implements the Fast Gradient Sign Method (FGSM) with an L∞ perturbation budget of ε=0.02 to generate initial adversarial examples. Building on this foundation, the research develops both multi-step (PGD) and momentum-based (MI-FGSM) attack variants to further degrade model performance by at least 70%. To demonstrate vulnerability under sparse, high-magnitude perturbations, the study adapts PGD and MI-FGSM attacks to localized 32×32 patches with increased epsilon values. The ultimate goal is to assess how these diverse attack configurations transfer from ResNet-34 (white-box) to DenseNet-121 (black-box), revealing key patterns in cross-model attack transferability and highlighting the brittleness of deep classifiers to both subtle and localized adversarial perturbations.

### 3. Finetuning with LoRA: Analyzing Category Classification Performance on Agnews Headlines

**Paper:** [example_3.pdf](example_3.pdf)  
**Model Repository:** https://github.com/timothycao/agnews-classifier  
**Experiments Repository:** https://github.com/aaronbengochea/agnews-classifier-experiments

This research evaluates the effectiveness of Low-Rank Adaptation (LoRA) for fine-tuning transformer-based language models on the AG News category classification task using RoBERTa-base. The main objective is to identify LoRA adapter configurations with fewer than 1M trainable parameters that optimally balance classification performance, computational efficiency, and generalization while keeping all 125.3M pre-trained RoBERTa-base parameters frozen. The study systematically compares LoRA configurations spanning between 600K-1M trainable parameters, exploring both low-rank and high-rank adapter settings to determine the minimum capacity required to sufficiently capture the complexity of the AG News dataset. By conducting structured experiments over LoRA's most important hyperparameters, the research aims to provide practical guidelines for selecting LoRA configurations in resource-constrained environments. The findings demonstrate that lower-rank LoRA adapters (~665K-705K parameters) achieve marginally superior generalization on hidden test sets compared to higher-rank counterparts (~1M parameters), while using only ~70% of the trainable parameter budget, highlighting the effectiveness of compact low-rank adapters for resource-efficient fine-tuning of large language models.

### 4. A Comparative Analysis of Sparse, Dense, and Hybrid Ranking Pipelines for MS MARCO

**Paper:** [example_4.pdf](example_4.pdf)  
**Repository:** https://github.com/timothycao/search-systems

This research conducts a systematic comparison of modern information retrieval components to quantify their relative and complementary strengths in hybrid architectures. The main objective is to evaluate how sparse lexical retrieval (BM25), dense vector search (FAISS HNSW), rank fusion strategies (Linear Score Fusion and Reciprocal Rank Fusion), and neural reranking (BERT-based bi-encoder) perform individually and in combination on the MS MARCO passage retrieval task. The study implements baseline systems consisting of a BM25 inverted index and an HNSW approximate nearest neighbor graph, then evaluates two rank fusion-based reranking strategies that merge rankings from sparse and dense retrieval pipelines. To assess the impact of semantic similarity modeling, a BERT-based bi-encoder reranker performs cascading reranking over all previous system outputs (BM25, HNSW, LSF, and RRF), isolating and measuring the incremental benefit provided by neural embeddings. Additionally, the research analyzes performance as a function of query length to understand how different ranking paradigms behave under varying query complexity. The goal is to provide practical guidance for designing high-performance retrieval architectures by demonstrating clear performance differences across retrieval paradigms and highlighting the value of layered reranking in hybrid pipelines.

### 5. A Comprehensive Analysis of Dynamic Tiered BM25 Indexing and Query Routing using MSMARCO

**Paper:** [example_5.pdf](example_5.pdf)  
**Repository:** https://github.com/timothycao/search-systems

This research conducts a comprehensive study of a production-inspired pipeline for dynamic BM25 indexing with tiered partitions and delta shards that handle freshly ingested documents. The main objective is to evaluate the quality versus cost trade-offs of a learned query routing system that dynamically decides between searching only high-quality documents (T1 tier) or searching both T1 and lower-quality documents (T2 tier) in parallel based on query characteristics. Documents are assigned to tiers using static BM25 scores derived from query-term frequencies, with delta shards periodically rolled into base indexes when thresholds are exceeded. At query time, the system over-fetches from tiered and delta shards, rescores with global statistics, and employs a learned query router to optimize retrieval decisions. The study compares tiered retrieval against a non-tiered BM25 baseline built on the same corpus, evaluating multiple learned query routing models with varying classification routing thresholds. By separating the corpus into training and working subsets and ensuring all evaluation documents are included in the working set, the research aims to quantify how dynamic tiering and intelligent query routing can balance retrieval quality with computational efficiency in large-scale information retrieval systems.

### 6. Code Sensei - Distributed AI Programming Assessment Handler

**Paper:** [example_6.pdf](example_6.pdf)  
**Repository:** https://github.com/allan-jt/CodeSensei

This paper introduces Code Sensei, an AI-driven programming assessment platform that addresses critical limitations in traditional technical interviews and programming assessments. The main objective is to develop a next-generation solution that dynamically adapts to each candidate's performance through large language model-powered recommendation engines, replacing static one-size-fits-all question flows with personalized, adaptive problem selection. The platform aims to provide immediate, detailed feedback beyond simple pass/fail signals by delivering fine-grained performance metrics including execution time, memory usage, and test case coverage in real-time. Built on a modular, cloud-native architecture, Code Sensei combines serverless AWS Lambda functions for orchestration, AWS Fargate for safe containerized code execution, and VPC-resident Bedrock endpoints to ensure scalability, security, and low latency. The system indexes a question bank in OpenSearch for sub-second retrieval, issues problems via an asynchronous API-driven workflow, and evaluates submitted code through strict isolation mechanisms. By unifying adaptive problem selection, immediate performance-driven feedback, and scalable infrastructure, the research demonstrates how Code Sensei can seamlessly scale to thousands of concurrent sessions while transforming programming assessments from static tests into interactive learning experiences.

### 7. Building a Disk-Based Search Engine for the MS MARCO Dataset

**Paper:** [example_7.pdf](example_7.pdf)  
**Repository:** https://github.com/timothycao/search-system

This research presents the design and implementation of a modular, disk-based search engine that indexes and retrieves passages from the MS MARCO dataset using BM25 ranking. The main objective is to build a fully functional, efficient retrieval system from first principles, emphasizing modularity, compression, and disk-based data access without reliance on external frameworks. The system is composed of three independent executables that operate in sequence: a parser that streams and tokenizes raw data producing sorted posting chunks, an indexer that merges postings into a single compressed inverted index stored on disk with accompanying lexicon and page table, and a query processor that loads index metadata and supports interactive ranked retrieval. The research aims to handle millions of documents efficiently through I/O-efficient external sorting, chunked parsing, and block-based compression while storing docIDs and term frequencies separately using VarByte encoding and gap compression. To optimize query performance, the system implements the BM25 scoring model with tunable parameters and incorporates advanced pruning algorithms including MaxScore and Block-Max WAND to skip low-impact postings and accelerate disjunctive queries. The goal is to demonstrate how fundamental components of an information retrieval system can be implemented from the ground up, yielding a scalable, transparent, and explainable prototype search engine that operates efficiently on disk while maintaining clear separation between parsing, index construction, and query processing stages.

## Research Domains

- **Deep Learning & Computer Vision:** Residual neural networks, image classification, model optimization
- **Adversarial Machine Learning:** White-box attacks, transferability analysis, model robustness
- **Natural Language Processing:** Fine-tuning strategies, parameter-efficient learning, text classification
- **Information Retrieval:** Hybrid ranking pipelines, dense and sparse retrieval, neural reranking
- **Search Systems:** Dynamic indexing, tiered architectures, learned query routing
- **Intelligent Systems:** AI-powered platforms, adaptive learning systems, distributed computing

## Contact

Aaron Bengochea  
New York University  
[ab6503@nyu.edu](mailto:ab6503@nyu.edu)
