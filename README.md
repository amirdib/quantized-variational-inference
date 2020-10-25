![Python version](https://img.shields.io/pypi/pyversions/Django.svg?style=for-the-badge)

**This repository is the official implementation of Quantized Variational Inference (NeurIPS 2020).**

## Install
Install python requirements in your virtual environnement:

```bash
pip install -U setuptools #important to have the last version
pip install -r requirements.txt
pip install -e .
```


To use the jupyter notebook, it is necessary to install the kernel as follow
```bash
# For instance, with conda.
conda create -n virtual_env
source activate virtual_env    # On Windows, remove the word 'source'

python -m ipykernel install --user --name virtual_env_name
```

Download the complete OQ grids with the following command
```bash
cd data
bash ingest.sh
```

## Usage
Using Tensorflow Probability [Variational Inference Interface](https://www.tensorflow.org/probability/api_docs/python/tfp/vi), given a taget_log_prob function and a surrogate_posterior func (the variational family), a [Tensorflow Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
```python
from qvi.core.vi import QuantizedVariationalInference
qvi = QuantizedVariationalInference(target_log_prob_fn=conditioned_log_prob,
                                            surrogate_posterior=surrogate_posterior,
                                            optimizer=tf.optimizer.Adam(),
                                            num_steps=num_steps,
                                            sample_size=sample_size,
                                            D=D)
qvi.run()
qvi.plot()
```
## Pre-trained Models
Pre-trained models with results can be produced with the scripts in the [notebooks' directory](notebooks/qvi.ipynb) . All article's figures are reproductible by running it.
Also, [HTML](notebooks/qvi.html) and [pdf](notebooks/qvi.pdf) versions of the notebook are available.

## Data and Grid Generation
Data used for modelisation is available in the [data directory](data/).  [Grids' directory](data/grids) contains the optimal grid for Normal Distribution in the format N_D_nopti (where N is the number of sample, D the dimension).

Non-optimal grid can be generated with the script [generate_normal_grids.sh](data/generate_normal_grids.sh) (R version 4.0 is needed).
```bash
cd data
bash generate_normal_grids.sh D N n 
```
where D is the dimension, N the level of the optimal quantizer and n the number of samples used to produce it (larger is better).

## Contributing
See [Licence](LICENSE) file.
