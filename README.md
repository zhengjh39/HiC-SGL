### HiC-SGL

Subgraph decomposition and graph representation learning for single cell Hi-C imputation and clustering.

![model](C:\Users\93595\Pictures\model.png)

The structure of the **HiC-SGL** model. It consists of an encoder and a decoder. The encoder extracts the local feature of each edge and the global feature of the cell graph. The decoder estimates the likelihood of each edge being present in the cell map using the encoded feature.



#### requirements

* python==3.9.6
* pandas==1.4.4
* pytorch==1.12.1
* numpy==1.23.1

* torch-geometric==2.1.0



#### Data Process

**required data**

**scHiC dataset:**“data.txt" is a table file, which needs to contain attributes:：

* cell_id: Cell ID
* chrom1: the chromosome where the initial position of the connection is located
* pos1: The relative position of the initial position of the connection on the chromosome (base sequence number)
* chrom2: Chromosome where the end position of the connection is located
* pos1: The relative position of the end position of the connection on the chromosome (base sequence number)
* count: the number of connections



**"label_info.pickle"**: file, the attribute "cell type" records the category corresponding to the cell id

**”config.JSON”**: The configuration file of the dataset, including properties:

* **data_dir**: The folder where the data is located, and the subsequently generated cell graph, cell features, model weights and other data will also be saved in it.
* **genome_reference_path**: chromosome size file
* **chrom_list**: list, chromosomes to process (corresponding to the name in 'data.txt')
* **resolution**: the fragment length of chromosome division (the length of a single bin)
* **dense_thre**: threshold of the number of readcount in the cell map
* **cell_dim** : dimension of the cell embedding
* **node_dim**: dimension of node embedding
* **neg_num** : number of negative samples during training
* **drop_ratio**: the proportion of edges to mask as test set.
* **batch_size**:  training batch size
* **num_workers**: the number of cpus used for dataloader
* **train_epoch**: number of train epoch 
* **pretrain_epoch**: number of pretrain epoch



#### dataprocess

```python
python HicProcess.py --dir [dir]
```

* [dir]：The path of the data folder

Get cell map (torch_geomertic.data), save in dir/cellgraph directory；cell feature(torch.tensor), save in dir/cellatr directory



#### Pretrain[option]

```
python pretrain.py --cuda [cuda] --dir [dir]
```

* [cuda] GPU used

* [dir] the folder where the data is located

Get the pre-training weight, save it in the /GCLweight directory



#### Train

```
python train.py --cuda [cuda] --dir[dir] [--pre]
```

* [cuda] GPU used
* [dir] The folder where the data is located

If --pre is included, the model uses pre-trained weights, otherwise it does not

Get the trained model parameters and save them in the dir/weight directory



#### utils

The utils module implements some methods for conveniently obtaining data, models, calling models for cell imputation, and obtaining cell embedding.

| function         |description                              | parameters                                                         | return                                                        |
| -------------- | --------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| get_cells      | obtain cell map and cell features | **data_dir**(str): data directory path; **c** (int) chromosome number | cellgraph(list[torch_geometric.data]), cell feature(tensor) |
| get_model      | obtain model                      | **data_dir**(str): data directory path;  **c** (int) chromosome number; **state**(str)( 'init', 'pretrained', 'trained'); **device**(str)：model device | model (torch.nn.Module)                                     |
| get_cell_embed | Get trained cell embeddings       | **data_dir**(str): data directory path                       | cell embed (numpy.array)                                    |
| impute_cell    | impute cell                       | **raw_cell**(list[torch_geometric.data]：raw cell， **model**(torch.nn.Module): model | imputed cell(torch.tensor)                                  |





