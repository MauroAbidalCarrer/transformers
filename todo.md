<!-- 1. use tiktoken. -->
<!-- 1. use a proper TensorDataset and Dataloader. -->
<!-- 1. Use config instead of huge method declarations. -->
<!-- 1. Use GeLU. -->
<!-- 1. Use better initialization -->
<!-- 1. Tie embedding and unembedding layers weights. -->
<!-- 1. use flash attention. -->
<!-- 1. set the weights to TFloat32 or bfloat16?. -->
<!-- 1. use bfloat with amp and "high" precision for float32 matmul. -->
<!-- 1. Use torch.compile. -->
<!-- 1. Use gradient accumulation and increase batch size. -->
<!-- 1. Use gradient clipping. -->
<!-- 1. Use fused AdamW -->
<!-- 1. Use a learning rate scheduler. -->
<!-- 1. Better handling of optimizer hyperparameters. -->
1. Use DDP.
1. Refacto data loader
1. use fineWeb-EDU.