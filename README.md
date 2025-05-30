# Applying GNN to BIM graphs for semantic enrichment

<img src="fig/layout.jpg" style="zoom:100%;"/> 

We present a novel approach of semantic enrichment, where we represent BIM models as graphs and apply GNNs to BIM graphs for semantic enrichment. 

We select a typical semantic enrichment task -- apartment room type classification -- to test our approach.

To achieve this goal, we created a BIM graph dataset, named **RoomGraph**, and modified a classic GNN algorithm to leverage both node and edge features, **SAGE-E**.

The RoomGraph dataset and the source codes of SAGE-E  are open to public research use. Enjoy!


# Installation

## Install Dependencies
1. Clone or download this repository:
```bash
git clone https://github.com/ZijianWang-ZW/SAGE-E.git
cd SAGE-E
```

2. Install [anaconda](https://www.anaconda.com/download) and follow the following steps

3. Create a new conda enviroment and install all required packages

```bash
conda create -n gnntutorial python=3.8 -y
```

```bash
conda activate gnntutorial
```

Install the jupyter notebook
```bash
conda install jupyter notebook -y
```

Install PyTorch. Training and testing SAGE-E does not need special GPU configurations. CPU processing is sufficient for the provided dataset.
```bash
conda install pytorch=2.3.0 torchvision torchaudio cpuonly -c pytorch
```

```bash
conda install -c pytorch torchdata
```

```bash
conda install pydantic -y
```

Install DGL
```bash
conda install -c dglteam dgl
```

Install all other required libs
```bash
conda install numpy, pandas, scikit-learn     
```


## Folder Structure
The following shows the basic folder structure:
```
├── code/
│   ├── SAGEE.py                 # The SAGE-E GNN architecture implementation
│   ├── best_default.pt          # Pre-trained model weights (default configuration)
│   ├── best_user.pt            # Pre-trained model weights (user configuration)
│   ├── node_evaluation.py      # Utility functions for model evaluation
│   └── train&test.ipynb        # Main training and testing notebook
├── dataset/
│   └── roomgraph.bin           # RoomGraph dataset (BIM graph data)
├── fig/
│   └── layout.jpg              # Project illustration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

# Usage

## Quick Start
1. **Open the main notebook**: Navigate to `code/train&test.ipynb`
2. **Run the cells step by step**: The notebook contains detailed explanations for each step

## Step-by-Step Guide

### 1. Load the Dataset
The RoomGraph dataset is provided in `dataset/roomgraph.bin`. The notebook will automatically load this dataset:
```python
from dgl.data.utils import load_graphs
bg = load_graphs("../dataset/roomgraph.bin")[0]
```

### 2. Model Architecture
The SAGE-E model is defined in `code/SAGEE.py`. It consists of:
- **SAGEELayer**: Individual layer that processes both node and edge features
- **SAGEE**: 4-layer network optimized for room type classification

### 3. Training
Run the training cells in the notebook to:
- Split data into train/validation/test sets
- Initialize the SAGE-E model
- Train with specified hyperparameters
- Monitor training progress

### 4. Evaluation
The notebook includes comprehensive evaluation:
- F1-score calculation
- Confusion matrix generation
- Model performance metrics

### 5. Using Pre-trained Models
Two pre-trained models are provided:
- `best_default.pt`: Model with default hyperparameters
- `best_user.pt`: Model with optimized hyperparameters

Load a pre-trained model:
```python
model = SAGEE(ndim_in, ndim_out, edim, activation, dropout)
model.load_state_dict(torch.load('best_default.pt'))
```

## Customization

### Using Your Own Data
To use SAGE-E with your own BIM graph data:
1. Convert your data to DGL graph format
2. Ensure node and edge features are properly formatted
3. Modify the data loading section in the notebook
4. Adjust model parameters if needed

### Hyperparameter Tuning
Key hyperparameters you can modify:
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `dropout`: Dropout rate for regularization
- Network architecture (layer dimensions in `SAGEE.py`)

## Output
The model performs room type classification on BIM graphs, outputting:
- Predicted room types for each node
- Classification confidence scores
- Performance metrics (F1-score, accuracy, confusion matrix)

## Citation

If you use this code or the RoomGraph dataset in your research, please cite our paper:

```bibtex
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
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Welcome to contact Zijian Wang (zijianwang1995@gmail.com) if you have any questions. 

If you want to know more about my work, please visit: https://zijianwang-zw.github.io/
