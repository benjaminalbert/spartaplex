# SpartaPlex

SpartaPlex is a deterministic algorithm with linear scalability for massively parallel global optimization of very large-scale problems.

See [the paper](https://doi.org/10.1016/j.advengsoft.2022.103090) for more information.

Currently, there are two implementations: one in Python and one in MATLAB. Both implementations support execution on a CPU or GPU.

A third implementation, written in C++/CUDA, will be published soon for multi-node multi-GPU execution.


## Installation

#### Get the SpartaPlex source
```
git clone https://github.com/benjaminalbert/spartaplex.git
```

#### MATLAB

The MATLAB implementation Does not require toolboxes for running on the CPU, but requires the [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html) for exexecution on the GPU.

#### Python

The Python implementation requires [PyTorch](https://pytorch.org) for both CPU and GPU execution.


## Example

An example 1024-dimensional objective function is provided for all implementations. The function is the sum of squares (spheroid) centered at a random point in [-1,1]. Both the Python and MATLAB folders contain the random center, from which their respective test executables (`test.py` and `test.m`) read to initialize the example objective function. Executing the commands below solves the problem `f(x)=∑((x-c)^2)` using `1024^2` function evaluations. 

To run the MATLAB example, either run the `test.m` script from within MATLAB or execute the following from within the `matlab` dir:
```
matlab -nodisplay -nodesktop -nosplash -r test
```

To run the Python example, execute the following from within the `python` dir:
```
python test.py
```

## Contact (Gmail)

benjialbert2

## Citation
```
@article{ALBERT2022103090,
	title = {SpartaPlex: A deterministic algorithm with linear scalability for massively parallel global optimization of very large-scale problems},
	journal = {Advances in Engineering Software},
	volume = {166},
	pages = {103090},
	year = {2022},
	issn = {0965-9978},
	doi = {https://doi.org/10.1016/j.advengsoft.2022.103090},
	url = {https://www.sciencedirect.com/science/article/pii/S0965997822000035},
	author = {Benjamin Alexander Albert and Arden Qiyu Zhang},
}
```

## License

See the [LICENSE](LICENSE) file
