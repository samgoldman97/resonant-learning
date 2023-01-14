# Resonant Learning

This github repository contains code for the paper, written as part of a Harvard Undergraduate Senior Thesis in Computer Science:
[Resonant Learning in Scale-free Networks](https://www.biorxiv.org/content/10.1101/2021.11.10.468065v1.full.pdf)

## Setup

To install the enviornment

```
conda env create -f enviornment.yml
pip install -r requirements.txt
python setup.py develop
```

## Experiments

Sample networks can be constructed and visualized in Gephi:

```
python scripts/draw_sf_graph.py
```


To create ordered chaos phase transition diagrams:

```
python scripts/order_chaos_transition.py
```


TODO: 
- Create ordered chaos phase transiiton diagram (line above)
- Work on ordered-chaos phase transition (in presence and not presence of oscillations)
- Get PSD of time series working and show heatmap
- Clean attractor cycles height calculation and viz and show that
- Clean up evolutionary analysis
- Clean up analysis notebooks


## Citation

If you use any component of this repository, please cite the following:

```
Goldman, Samuel, Maximino Aldana, and Philippe Cluzel. "Resonant Learning in Scale-free Networks." bioRxiv (2021).
```
