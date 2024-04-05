# Intention-Aware Control Facilitated by Polynomial Chaos Expansion

The benchmark of using polynomial chaos expansion (PCE) to solve intention-aware control of autonomous vehicles.

## Requirements

**Operating system**
 - *Windows* (compatible in general, succeed on 11)
 - *Linux* (compatible in general, succeed on 20.04)
 - *MacOS* (compatible in general, succeed on 13.4.1)

 **Python Environment**

 - Python `3.11`
 - Required Packages: `numpy`, `numpoly`, `scipy`, `treelib`, `matplotlib`, `importlib-metadata`. 

**Required Libraries**
 - `gurobipy` solver (**license** required, see [How to Get a Gurobi License](https://www.gurobi.com/solutions/licensing/))
 - `stlpy` toolbox (see [Documentation](https://stlpy.readthedocs.io/en/latest/) or [GitHub repository](https://github.com/vincekurtz/stlpy))
 - `chaospy` toolbox (see [Documentation](https://chaospy.readthedocs.io/en/master/) or [GitHub repository](https://github.com/jonathf/chaospy))

### Quick Installation
 
1. Install conda following this [instruction](https://conda.io/projects/conda/en/latest/user-guide/install/index.html);

2. Open the conda shell, and create an independent project environment;
```
conda create --name intentaware python=3.11
```

3. In the same shell, activate the created environment
```
conda activate intentaware
```

4. In the same shell, within the `intentaware` environment, install the dependencies one by one
 ```
conda install -c anaconda numpy
conda install -c conda-forge numpoly
conda install -c anaconda scipy
conda install -c conda-forge treelib
conda install -c conda-forge matplotlib
conda install -c conda-forge importlib_metadata
```

5. In the same shell, within the `intentaware` environment, install the libraries `gurobipy`, `stlpy`, and `chaospy`:
```
python -m pip install gurobipy
pip install stlpy
pip install chaospy
```

6. Last but not least, activate the `gurobi` license (See [How To](https://www.gurobi.com/documentation/current/remoteservices/licensing.html)). Note that this project is compatible with `gurobi` Released version `11.0.1`. Keep your `gurobi` updated in case of incompatibility. 

## Run Examples

- Lead to the `example` directory;
- Lead to the `overtaking` or the `intersection` folder;
- Run the main script `main.py`;
- Rendering constraints may take up to 1 to 2 minutes, depending on the computational performance;
- Plotted figures automatically saved in the `data` subfolder.

## License

This project is with a BSD-3 license, refer to `LICENSE` for details.