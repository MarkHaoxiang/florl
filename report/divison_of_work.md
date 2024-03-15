Hao Xiang Li
- Report writeup - entire report and all analysis, including appendix. An edit history can be provided on request.
- Report figures and tables: all figures, tables and visualisations.
- Florl Library
- - Proposed and designed library
- - Design and implementation of Knowledge, KnowledgeShard and related tests.
- - Design and implementation of FlorlClient abstractions. Implementation of all clients, including GymClient, KittenClient
- - Implementing all FRL algorithms (QTOptAvg, DQNAvg and TD3Avg) including Prox versions.
- Federated Reinforcement Learning Experiments
- - Entirety of federated vs baseline comparisons (DQNAvg, QTOptAvg)
- - Entirety of data heterogeneity experiments and visualisations
- - Proposed motivation (and prototyped) the experiment for sending different modules of TD3.
- - Proximal FRL experiments
- ROS Flower RL
- - Designed architecture (Both FL and FRL)
- - Implemented FL classifer for data downloading, publishing and substantial section of client, launch script. Implemented server, evaluator, and classifier.
- - Implemented FeatureLabelPair, FloatTensor, Transition, PolicyService, SampleFeatureLabelPair, SampleTransition and related Python utilities.
- - Implemented dual thread approach for bridging ROS and Flower.
- - Implemented FRL gym_controller, C++ replay_buffer, and sections of RosKittenClient (parts of train function, and sample_request)

