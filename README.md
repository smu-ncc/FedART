# FedART: A Neural Model Integrating Federated Learning with Adaptive Resonance Theory
This is the source code for FedART (paper under review in IEEE TNNLS).

## Table of Contents
1. [Introduction](#introduction)
2. [Code Organization](#code-org)
3. [Communication Method](#comms)
4. [What you need to change for your own dataset](#customization)
5. [How to run everything manually](#manual-run)
6. [How to run everything using automated scripts](#auto-run)
7. [License](#license)

## Introduction <a name="introduction"></a>
Federated Learning (FL) is a privacy-aware machine learning paradigm wherein multiple clients combine their locally learned models into a single global model without divulging their private data. However, current FL methods typically assume the use of a fixed network architecture across all the local and global models and they are unable to adapt the architecture of the individual models according to the local data, which is especially important for data that is not Independent and Identically Distributed (non-IID) across different clients. To address this limitation, we propose a novel FL method called Federated Adaptive Resonance Theory (FedART) which leverages the adaptive abilities of self-organizing Adaptive Resonance Theory (ART) neural network models. Based on ART, the client and global models in FedART dynamically adjust and expand their internal structure without being restricted to a predefined static architecture, providing architectural adaptability. In addition, FedART employs a universal learning mechanism that enables both federated clustering, by associating inputs to automatically growing categories, as well as federated classification by coassociating data and class labels. Our experiments conducted on various federated classification and clustering tasks show that FedART consistently outperforms state-of-the-art FL methods for data with non-IID distribution across clients.

**FedART can be run for single or multiple rounds.**

![FedART Federated Learning Architecture](FedART.png)

## Code Organization <a name="code-org"></a>
In the following discussion, `<dataset>` is used as a placeholder for dataset name.
- `fedart_supervised_learning` directory contains the data and source code related to supervised learning (classification).
     - `data/<dataset>` contains the dataset in .csv or .hd5 format. `data/<dataset>/prep_data.py` is used to extract data and save in the .csv file. If you add a new dataset, please implement `data/<dataset>/prep_data.py` for it.
     - `partitioned_data/<dataset>` directory saves the data from `data/<dataset>` after it has been partitioned among different clients.
     - `learned_models/<dataset>` directory saves the local models learned by different clients and the aggregated global model learned after federated learning.
     - `saved_args/<dataset>` directory saves the arguments or parameters related to the given dataset.
     - `src` directory contains the federated learning code.
       - `setup_fl.py` contains the arguments or parameters corresponding to different datasets. It also calls the `run_ccordinator` function to start the `experiment_coordinator` (described below).
       - `experiment_coordinator.py` contains code for loading data from `data/<dataset>`, normalizing it, partitioning it among different clients, doing train-test splits, and preparing data for global testing and training a baseline non-FL centralized model. This is where _nonIID_ or _IID_ partitioning happens (see `prep_client_data` function). Furthermore, it creates the directories `partitioned_data`, `learned_models`, `evaluation_results`, and `saved_args`. It saves the partitioned data and the dataset-related arguments while the models are saved later by the clients and server. It also implements functions for evaluating the models.
       - `clients_runner.py` loads the partitioned data from `partitioned_data/<dataset>` directory and runs multiple parallel client processes. The client processes connect to the server using sockets.
       -  `server_runner.py` runs the federated learning server process. The server connects to the clients using sockets.
     - `FedART` directory contains the implementation of the FedART server, FedART clients, and the underlying Fusion ART model (see `Base` directory).

## Communication Method <a name="comms"></a>
We use simple socket communication for bi-directional send and receive between various clients and server. The clients run in parallel using multiprocessing.

## What you need to change for your own dataset <a name="customization"></a>
1. Add your dataset to the `fedart_supervised_learning/data/<dataset>` directory. Extract the data as a Pandas dataframe and save as a .csv or .hd5 file.
2. Provide the arguments or parameters corresponding to the dataset in the `setup_fl.py` file under `get_args` function by using an _if_ statement `if args.dataset == '<dataset>'`.

That's it! You are good to go!

## How to run everything manually <a name="manual-run"></a>
First, make sure `pandas, multiprocessing, scokets, and threading` packages are installed and Python version >= 3.5.0.
For each new experiment run, the following three commands need to be executed in the given sequence.

1. Open two terminals. 
2. `cd src` in both terminals.
3. In terminal 1, call
   
        python setup_fl.py --dataset=<dataset> --split_type=<split> --random_seed=67
   
   The `<dataset>` name should be without quotation marks. `<split>` should be either `nonIID` or `IID`.
   
4. In terminal 1, call

        python server_runner.py --dataset=<dataset> --fl_rounds=<R>

   Here, `<R>` is the number of federated learning rounds between server and clients.
   
5. Wait until you see "Server is listening..." in terminal 1. This means we can now start client processes.
   
6. In terminal 2, call

        python clients_runner.py --dataset=<dataset>

7. After both the server and client processes finish, calculate the evaluation scores (precision, recall, accuracy, etc.) and save them by calling

         python evaluator.py --dataset=<dataset>

This will run the clients and server in parallel to execute federated learning. 

During the execution, following records are kept:
1. The global and partitioned client data is saved in the directory `partitioned_data/<dataset>`.
2. The learned server and client models are saved in the directory `learned_models/<dataset>`.
3. The dataset-specific arguments or parameters are saved in the directory `saved_args/<dataset>`. This is meant for running server and clients multiple times for different experiment trials without having to rerun `setup_fl.py`.
4. The model evaluation results are saved in `evaluation_results/<dataset>`.

## How to run everything using automated scripts <a name="auto-run"></a>
Comming soon.

## License <a name="license"></a>

## To-do 
1. Setup an alternative for file based communication instead of sockets - backup option for experimentation.
2. Add hyper-parameter search.
3. Re-organize the fedart clustering code and upload here.
4. Add other datasets if size permits.
5. Add scripts for running all the programs automatically
