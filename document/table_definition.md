# FX+ Table Definition

This document contains the requirement of the `*_impl.py` and syntax of the `*.td` file used by the FX+ canonize system.

## Requirement of Model Implementation

The `*_impl.py` should contains a derived class of `torch.nn.Module` that has two member functions: `__init__()`, and `forward()`.

The `__init__()` can take a list of arguments used to initiate the neural network model, such as dimension sizes. The arguments show fall in the following types:
* Scalar types (int, float)
* Attribute Class: a class whose values can all be accessed through the attribute names.
* List of above types

The `forward()` takes a list of `torch.Tensor`.

## Table Definition Syntax

The table definition (`.td`) is an LLVM table-gen inspired mechanism that collects all the necessary information required by constructing and running the model. An example of the `.td` file is as follows:
```
[MODEL_NAME]<Model> {

    init_args<list> {...}

    runtime_args<list> {...}

    inputs <list> {...}

    description: some description;
}
```
The syntax has two building blocks: objects and attributes.

**Objects**: The objects take the form `[name]<[type]>{[body]}`. The `name` should follow the Python variable naming constrains. The `body` can contain multiple objects and attributes, which offers the flexibility to adapt to different model implementations. The `type` can be one of the following:
* `Model`: The outer-most layer of the table definition. Its body contains three member objects:
    * `init_args`: the argument list matching the arguments of the model's `__init__` function.
    * `runtime_args`: additional args used at runtime, such as batch sizes.
    * `inputs`: a list of `Tensor` types that match the arguments of the model's `forward` function.
* `attr_cls`: the attribute class.
* `int/float`: the scalar types
* `list`: container type, or list of unknown number of scalar types.
* `tensor`: special type exists in the `inputs`. Which provides the essential information used to construct tensors, and example of a tensor under normal distribution is as follows:
    ```
    e_emb<tensor> {
        distrib: randn;
        size: (batch_size, sequence_length, embedding_dim);
        dtype: float;
        description: input embedding;
    }
    ```
    Notably, the `size` should be a tuple containing the name of objects from either the `init_args` or the `runtime_args`.

Besides, all the objects can have an optional attribute `discription`, which will be used to generate the comments in the generated code.

**Attributes**: The attribute takes the form `[name]:[value];`. It is used to record informations or comments. The `name` should follow the Python variable naming constrains, while the value can be any charactors except "`;`".
