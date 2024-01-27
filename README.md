# FX_Plus
FX+: A Reusable and Extensible Compiler Infrastructure for the Deep Learning

## Install the package
To install the package, under root directory, run
```shell
pip install .
```
To install the package under the develope mode, run
```
pip install -e .
```

## Canonize models for easier profiling and verification
### Rational
To evaluate the compiler quality, profiling and verification are two major metrics used by developers. There are some common helper functions needed in the profiling and verification procedure, such as getting the sample input and configuring the models with different parameter settings. However, not all model implementations offer these helper functions with high quality. On the other hand, manually modifying the implementation for each model is tidious, error-prone, and leading to duplications. 

To address the above issues, Fx+ offers a handy canonize interface. It takes the table definition for models that gather only the necessary information and automatically generate high quality information with a sigle command.

### Register a custom model
To register a custom model, user need to provide two files under the `[model directory]`:
- `[model_name]_impl.py`: the file contains the implementation of nn.Module model
- `[model_name].td`: the file contains the table definition of model information

The detailed requirements of the `*_impl.py` file and syntax of the `*.td` file can be found in [table_definition.md](document/table_definition.md).

The model can be canonized through the following command line instruction:
```
canonize_models -d [model directory]
```
The above instruction will generate two additional files in the same directory:
- `[model_name].py`: a wrapper over the original model implementation that offer all the helper functions for easy profiling and verification. The wrapper class will have the same name of the original implementation, but can be initiated with string containing the path to the the json file.
- `[model_name].json`: A template for users to fill out the config in json format.
