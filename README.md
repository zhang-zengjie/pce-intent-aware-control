## Dependencies

Python=3.10

- `conda install -c conda-forge matplotlib`
- `conda install -c anaconda scipy`
- `conda install -c anaconda numpy`
- `conda install -c conda-forge treelib`
- `conda install -c gurobi gurobi`
- `pip install importlib-metadata`
- `conda install -c conda-forge importlib_metadata`
- `conda install -c conda-forge numpoly`

The script `gen_basis.py` must be run first to configure random variables. Then:

- Run `compare_statistics.py` to compare PCE and Monte Carlo
- Go go `param.py` to change simulation variables
- Run `main.py` to solve the optimizer
- Run `plot_results.py` for visualization
