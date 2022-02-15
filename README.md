# dropCluster
PyTorch implementation of dropCluster. Citations and references are listed and importable from each file.


# Implementation Details
### Module Location: 

The paper only uses dropCluster in the convolutional stem (after Conv(in_channels=3, out_channels=64)). I suggest using it before the maxpooling operation. The paper also linearly increases the dropout rate from 0 to 0.1/0.15 until the end of training. They don't start until epoch 40 to give the network time to learn stable representations of the data. Too early and the clustered feature maps won't stay relevant for very long. This should be determined experimentally by evaluating the training loss curve.


### ArgParse Options:
```python
    self.parser.add_argument('--use_cluster_dropout', type=float, default=False, help='use a cluster-wise dropout')
    self.parser.add_argument('--dropCluster_warmup', type=int, default=False, help='number of epochs to use to raise keep_prob from 0.0 to use_cluster_dropout (0.1)')
    self.parser.add_argument('--dropCluster_start', type=int, default=20, help='Training epoch to start implementing dropCluster')
    self.parser.add_argument('--dropCluster_update_freq', type=int, default=10, help='Number of epochs between updating the clusters')
```


### Training script: train.py
```python
    dataset = training_data
    
    if opt.use_cluster_dropout:
	# Define scheduler for updating the dropout rate
        dropCl_scheduler = scheduler(param=opt.use_cluster_dropout,
                                     starting_epoch=opt.dropCluster_start,
                                     total_epochs=opt.dropCluster_warmup,
                                     total_batches=len(dataset),method='linear')

    # Begin training
    for epoch in range(1, max_epoch):
        for i, data in enumerate(dataset):
            # dropCluster
            if opt.use_cluster_dropout:
	    	# Update dropout rate
                update_DropCluster(model.netEst, prob=dropCl_scheduler(epoch=epoch-1, batch=i))
                if (epoch % opt.dropCluster_update_freq == 0):
		    # Initialize or update the cluster segmentation maps
                    update_DropCluster(model.netEst, activate=True)
                    model.set_input(data)
                    with torch.no_grad():
                        model.forward()

            # Forward pass + optimization step
            model.set_input(data)
	    model.forward()
            model.optimize()
```

JTL 10.22.2020

Updated: JTL 02.15.2022
