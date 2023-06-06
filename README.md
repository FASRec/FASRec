# FASRec

This is the official code for Future Augmentation with Self-Distillation in Recommendation

## Reference Code：

https://github.com/pmixer/SASRec.pytorch
https://github.com/fadel/pytorch_ema/tree/master/torch_ema

## Commands：

To train our FASRec on the Yelp data with default parameters:

```
python main.py --save_dir=Yelp/our+ --teaching_epoch=300 --gpu=4 --dataset=Yelp
```

To train the baseline model on the Yelp data with default parameters:

```
python main.py --save_dir=Yelp/orgin --teaching_epoch=1000 --gpu=5 --dataset=Yelp
```

## Options:

The training of the FASRec model is handled by the main.py script that provides the following command line arguments.

```
--dataset            STR           Name of dataset.               Default is "Beauty".
--save_dir           STR           Path to save checkpoint.       Default is "tmp_out".
--batch_size         INT           Batch size.                    Default is 128.    
--lr                 FLOAT         Learning rate.                 Default is 0.001.
--maxlen             INT           Maxmum length of sequence.     Default is 50.
--hidden_units       INT           Number of hidden units.        Default is 50.
--num_blocks         INT           Number of blocks.              Default is 2.
--num_epochs         INT           Number of epochs to run.       Default is 201.
--num_heads          INT           Number of heads.               Default is 1.
--dropout_rate       FLOAT         Dropout rate value.            Default is 0.5.
--device             STR           Device for training.           Default is 'cuda'.
--l2_emb             FLOAT         L2 regularization value.       Default is 0.0.
--gpu                STR           Name of GPU to use.            Default is "0".
--alpha              FLOAT         Weight for teacher predict.    Default is 0.7.
--teaching_epoch     INT           When to use teacher predict.   Default is 300.
--onlyP              BOOL          Only use teacher's postive     Default is False.
																	 predict.
--decay              FLOAT         Decay rate for EMA.            Default is 0.999.
```
