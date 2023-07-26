# fastsync
A Fast Edge-Based Synchronizer for Tasks in Real-Time Artificial Intelligence Applications - Codes and Dataset

### Simulation Configuration ###
I use a task graph (DAG) with a mixture of synchronous,
asynchronous and local tasks. The task graph is similar to
those used in Bulk Synchronous Parallel (BSP), Stale Synchronous Parallel (SSP) [11] and Dynamic Stale Synchronous
Parallel (DSSP) [13] approaches for synchronizing parameter
updates in distributed machine learning and neural networks.
The models usually have the following four steps. (i) Compute gradients using local weights. (ii) Push gradients to
parameter server to compute global weights. (iii) Pull new
computed global weights from parameter server. (iv) Update
local weights using global weights.

The execution times of a single task on workers is based
on a mixture distribution. One for the fast execution and
the other for slow execution. The execution times used in
the simulations are gotten from traces from the clustering
8
experiments. The times are split into two to depict short
(µ = 25ms) and long tasks (µ = 80ms).
The parameters in the simulations are as follows. (i) Synchronization degree: ratio of the total machines required to
pass quorum. (ii) Worker size: The maximum number of
workers present in the system at any point in time. (iii)
Simulation rounds: The number of times the task graph is
continuously run. (iv) Clustering frequency: This is the rate at
which re-clustering is done by the controller.

### Implementation ###
I evaluate the performance of our algorithm compared
to three other parameter server synchronization models (ASP,
BSP and SSP). I aim to find how our algorithm compares to
the other models in terms of accuracy and training time. We
implement all the parameter server synchronization models in
Ray. I train a simple 2D convolution neural network model
with a batch size of 16 on the KMNIST dataset2
consisting of
70, 000 28x28 gray scale images (60, 000 training and 10, 000
testing set examples). Each model is trained on a machine
with 2.4GHz Intel Core i5 machine with 12GB of memory
dedicated to the workers and 3GB dedicated to the parameter
server.
I compare the training accuracy and training time of our
algorithm, ASP, BSP and SSP models for different number
of workers. Fig 18 and 19 shows the training time versus
training iterations for our algorithm (Fast Sync), ASP, BSP,
SSP-3 (staleness threshold = 3 iterations) and SSP-4 (staleness
threshold = 4 iterations) for 5 and 20 workers respectively
