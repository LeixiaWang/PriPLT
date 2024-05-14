# PriPL-Tree: Accurate Range Query for Arbitrary Distribution under Local Differential Privacy

## Contents

* 🌸 **PriPL-Tree (Full Version).pdf**: The complete manuscript, including additional appendices.
* 🌸 **main.py**: Examples demonstrating how to use our provided functions to build PriPL-Trees and respond to range queries.
* **frequency_oracle.py**: The basic SW and OUE mechanisms.
* **pl_approximation.py**: The piecewise linear fitting function.
* **pripl_tree.py**: The PriPL-Tree algorithm for 1-D range queries.
* **grids.py**: The algorithm combines PriPL-Trees with adaptive grids for multi-dimensional range queries.
* **tool.py**: Tools for reading datasets and queries and computing the MSE of estimates.
* **parameters.py**: Default parameters for the algorithms.
* **datasets (folder)**: Contains four real-world datasets evaluated in our manuscript.
* **query (folder)**: Contains query files, each containing 1,000 random queries with varying query volumes from 0.1 to 0.9.

## Requirements

> Python==3.11.0
>
> numpy==1.23.5
> 
> treelib==1.6.1
> 
> scipy==1.10.1
> 
> h5py==3.8.0
