<!-- 1. use tiktoken. -->
<!-- 1. use a proper TensorDataset and Dataloader. -->
<!-- 1. Use config instead of huge method declarations. -->
<!-- 1. Use GeLU. -->
<!-- 1. Use better initialization -->
1. Tie embedding and unembedding layers wieghts.
1. use flash attention.
1. use bfloat with amp.
1. set the weights to TFloat32?.
1. increase the size of the model.
1. Use torch.compile.
1. Use gradient accumulation and increase batch size.
1. Use a learning rate scheduler.
1. Better handling of optimizer hyperparameters.
1. Use DDP.
1. use fineWeb-EDU.