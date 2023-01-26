# RaFL

Resource-Aware Heterogeneous Federated Learning using Neural Architecture Search

## Dependencies

To reproduce our experiments, please follow the instructions to set up environments:

First, creating a python environment via conda:

```
conda creat -n rafl python=3

#activate the environment
source activate rafl
```

Then, installing requirement python packages:

[Pytorch](https://pytorch.org/get-started/locally/) installation (for Linux):

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

Next, installing other python package dependencies:

```
pip install -r requirements.text
```

We are all set about software dependencies, to make sure you can running our codes, you may further need a GPU.
If you are familiar with Docker, please follow the instructions in the `Docker/`, and build a docker image to run our experiments.
## Reproduce experiments

We provided user-friendly interface to run our experiments and extend our method to customized applications.

All our parameters are stored in a `*.yaml` files under the folder `src/config`

To run an experiments, for instance, RaFL-(a) on MobileNet-V3 and CIFAR-100 with 100 clients and 10% random client participation rate in each round, simply exceute the following command:

```
cd /src

python eval.py --cf config/cifar100/mbv3/config.yaml

```

to equip RaFL with ensemble distillation:
```
cd /src

python eval.py --cf config/cifar100/mbv3/ensemble/config.yaml

```
To reproduce more our experiments, can find more configuration files under the folder `src/config`

Detailed argument explaination would be released soon after paper review.

### Run experiments on other dataset

We defaultly provide experiments running option on CIFAR-10/100. If you want to repreduce our experiments on Tiny ImageNet or CINIC-10, please download it first, then change the argument `datadir` in `*.yaml` file.

If running large scaled experiments on FEMNIST with 3000 clients, to avoid GPU memory explore, use the `eval_large_scale.py` to running experiments:

```
cd /src

python eval.py --cf config/femnist/3k_clients.yaml

```

## Acknowledgment

Under review

## License

Under review
