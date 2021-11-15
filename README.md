Overview
--------

This is a scratchbook for experiments with Brax.

Installation
------------

1. Install miniconda by first downloading the bash file from [here](https://docs.conda.io/en/latest/miniconda.html), for instance the python 3.7 bash file can be downloaded with:

`wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh`

then running:

`bash Miniconda3-latest-Linux-x86_64.sh -b -u -p "/scratches/peano_2/miniconda"`

This will install miniconda in the path `"/scratches/peano_2/miniconda"` instead of the default path.

2. Activate miniconda binary via: `export PATH=/scratches/peano_2/miniconda/bin:$PATH` this will make possible to use the command `conda` from the terminal.

3. Create a new environment by: `conda create -n myenv python=3.7`

4. Activate the env by first running `source /scratches/peano_2/miniconda/etc/profile.d/conda.sh` then run `conda activate myenv`

5. The following are the requirements needed to run the demo [here](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training_torch.ipynb#scrollTo=GJhPpM5ZPrpq), which is based on Python 3.7.12 and GCC 7.5.0.

Install a matching version of Jax, Torch, and CUDA drivers. This can be done by:

- `pip install --upgrade jax jaxlib==0.1.72+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html`
- `pip install --upgrade torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
- `pip install --upgrade gym==0.17.3`

Install brax and other packages: `pip install git+https://github.com/google/brax.git@main` (note that Brax 0.0.7 is used)

Note that the version of Brax et al. that works with the above libraries is this: *brax-0.0.7 chex-0.0.8 cloudpickle-2.0.0 dataclasses-0.6 dm-tree-0.1.6 flax-0.3.6 grpcio-1.41.1 gym-0.21.0 importlib-metadata-4.8.2 msgpack-1.0.2 optax-0.0.9 protobuf-3.19.1 pytinyrenderer-0.0.13 tensorboardX-2.4 toolz-0.11.2 zipp-3.6.0*

6. Install optional packages via: `pip install matplotlib gpustat` 

7. **Not used** In alternative pytorch can be installed directly in Conda, but it is necessary to check the version: `conda install -n myenv pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

8. **Note** this is very important for *ant* and *halfcheetah* envs in Brax. It is necessary to add this line or there is an OOM break: `os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'`. This seems to be due to the pre-allocation of memory of JAX, see [this])https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)

From the [salina code](https://github.com/facebookresearch/salina/tree/main/salina_examples/rl/ppo_brax), it seems they use other strings:  `os.environ['OMP_NUM_THREADS']=1` and `os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=false`

9. To create a new SSH key for a new Github repository:

- Type `ssh-keygen` and decide the name and password (can be empty) of your key.
- Copy the content of the file that has been created in `/homes/mp2008/.ssh/id_rsa.pub` (name and path can change based on you choice).
- Go in your GitHub account, select your avatar on top-right corner, then select `Settings` from the menu. Go to `SSH Keys`. Create a new entry and copy the content of your new `/homes/mp2008/.ssh/id_rsa.pub` file.
- To clone the repo, go in your repo main page, and select `Download` then the `SSH` option. You can now clone the repo in your new machine with `github clone repo_name` where `repo_name` is the name of your repo.


