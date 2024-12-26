# Applying GNN to BIM graphs for semantic enrichment

<center><img src="fig/layout.jpg" style="zoom:100%;"/> 

We present a novel approach of semantic enrichment, where we represent BIM models as graphs and apply GNNs to BIM graphs for semantic enrichment. 

We select a typical semantic enrichment task -- apartment room type classification -- to test our approach.

To achieve this goal, we created a BIM graph dataset, named **RoomGraph**, and modified a classic GNN algorithm to leverage both node and edge features, **SAGE-E**.

The RoomGraph dataset and the source codes of SAGE-E  are open to public research use. Enjoy!


# Requirements
- PyTorch
- DGL
- numpy
- pandas
- scikit-learn
- time


Training and testing SAGE-E does not need special configurations. The basic environment including the required libraries will be fine. 


### Folder structure
The following shows the basic folder structure.
```
├── code
│   ├── SAGEE.py # The architecture of the GNN algorithm.
│   ├── best_default.py # The selected model weight by authors.
│   ├── node_evaluation.py # The supplementary code for training process  
│   └── train&test.ipynb # The main code about training and test
├── dataset
    └──roomgraph.bin # The constructed graph dataset.
```

# Usage
Go to "code/train&test.ipynb". The code is explained step by step. 

## Bibtex
@article{WANG2022104039,
    title = {Exploring graph neural networks for semantic enrichment: Room type classification},
    journal = {Automation in Construction},
    volume = {134},
    pages = {104039},
    year = {2022},
    issn = {0926-5805},
    doi = {https://doi.org/10.1016/j.autcon.2021.104039},
    author = {Zijian Wang and Rafael Sacks and Timson Yeung}
}

## Contact
Welcome to contact Zijian Wang (zijianwang1995@gmail.com) if you have any questions. 

If you want to know more about my work, please visit: https://zijianwang-zw.github.io/
