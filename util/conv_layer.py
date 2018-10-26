"""
This is where I want to implement my ConvNet for CIFAR-10 data.

Currently, my idea of a 5 layer network is composed of:


0   : Input : 32-by-32-by-3 CIFAR-10 data

1   : Conv  : 3 * 5-by-5-by-3 kernel    => 28-by-28-by-3 data
    - # of parameters   : 228

2   : Conv  : 3 * 5-by-5-by-3 kernel    => 24-by-24-by-3 data
    - # of parameters   : 228

3   : Pool  : max of 2-by-2 window      => 12-by-12-by-3 data

4   : FC    : fully connected           => 10 scores
    - # of weights      : 4320


At training time:

5   : Softmax   : Softmax loss function => 1 loss

TODO
"""
