# Intent-Aware Control Facilitated by Polynomial Chaos Expansion

**Author:** Zengjie Zhang (z.zhang3@tue.nl)

## Dependencies

Recommended Python version: 3.10

- `conda install -c conda-forge matplotlib`
- `conda install -c anaconda scipy`
- `conda install -c anaconda numpy`
- `conda install -c conda-forge treelib`
- `conda install -c gurobi gurobi`
- `pip install importlib-metadata`
- `conda install -c conda-forge importlib_metadata`
- `conda install -c conda-forge numpoly`

## Run the overtaking case

- In `case_1_main.py`, choose the scenario variable `scene` among `0 (OV switching_lane)`, `1 (OV slowing down)`, and `2 (OV speeding up)`;
- Run `case_1_main.py` to solve the overtaking problem;
- Run `case_1_plot.py` to visualize the result; Remember to switch the scenario variable `scene` to see results in different scenarios.

## Run the intersection case

- In `case_2_main.py`, choose the scenario variable `scene` among `0 (OV switching_lane)`, `1 (OV slowing down)`, and `2 (OV speeding up)`;
- Run `case_2_main.py` to solve the overtaking problem;
- Run `case_2_plot.py` to visualize the result; Remember to switch the scenario variable `scene` to see results in different scenarios.