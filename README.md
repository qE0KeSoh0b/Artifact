

## Enviroment
- [ ] need test on other machines

### 1. environment requirement

- python                     3.8
- pybind11                   2.10.4
- ogb                        1.3.5
- torch                      1.10.1+cu111
- torch-cluster              1.6.0
- torch-geometric            2.2.0
- torch-scatter              2.0.9
- torch-sparse               0.6.13
- torch-spline-conv          1.2.1

```bash

conda create -n ROF python=3.8
...

```

- 

### 2. reuse_search_ext installation

```bash
conda activate ROF
cd $(repo)/ROF/reuse_search_ext
mkdir build
make
```


### 3. reuse_conv installation

```bash

alias add-gcc11 'setenv PATH /usr/local/GNU/gcc11/bin/\:$PATH ; setenv LD_LIBRARY_PATH /usr/local/GNU/gcc11/lib/:/usr/local/GNU/gcc11/lib64/\:$LD_LIBRARY_PATH '

alias add-cu111 'setenv PATH /usr/local/cuda-11.1/bin/\:$PATH ; setenv LD_LIBRARY_PATH /usr/local/cuda-11.1/lib64/\:$LD_LIBRARY_PATH '
```


```bash

conda activate ROF
add-cu111 ; add-gcc11;
cd $(repo)/ROF/reuse_conv
python setup.py install
```

## Evaluation

### reuse ratio

```bash

cd $(repo)/ROF_evaluation/reuse
...
```

### cuda kernel 
```bash

cd $(repo)/ROF_evaluation/cuda
...
```

### end-to-end

```bash
cd $(repo)/ROF_evaluation
# ROF_main.py
```
