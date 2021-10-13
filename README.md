# Best partition
This repo is the author's implementation of the algorithms described in the paper "[Subnetwork Constraints for Tighter Upper Bounds and Exact Solution of the Clique Partitioning Problem](https://arxiv.org/abs/2110.05627)." We apply the branch-and-bound method to solve the clique partitioning problem. It could also be used to maximize the modularity function to reveal a network's community structure. If you find this work useful, please, consider citing:
```
A. Belyi, S. Sobolevsky, A. Kurbatski, C. Ratti, "Subnetwork Constraints for Tighter Upper Bounds and Exact Solution of the Clique Partitioning Problem," arXiv preprint arXiv:2110.05627
```

## Building and running
We tested this code only on Linux and Mac. In theory, it should also work on Windows, but please feel free to contribute if you find any problems.

The project is configured using [CMake](https://cmake.org/), so you need to install it first. Furthermore, to solve linear programming problems, we use COIN_OR CLP and CBC, so in order to build the project, you also need to install the COIN_OR CBC solver.
To do that, follow the instructions from the [official page](https://github.com/coin-or/Cbc).
On Mac we recommend using [Homebrew](https://brew.sh/), on ubuntu/debian use ```sudo apt install coinor-libcbc-dev```.
If you installed to custom location, provide `CBC_ROOT="path/to/cbc"` variable to CMake.
You can also use CPLEX if you have it.
CMake should find it if installed in the default location, otherwise provide the path in `CPLEX_ROOT`. The steps are:
1. Install [CMake](https://cmake.org/download/);
2. Install [CBC](https://github.com/coin-or/Cbc);
3. Run CMake:
```bash
mkdir build
cd build
cmake ..
```
4. Run ```make```;
5. Run built program: ```./BestPartition```.

It should reproduce the results presented in the paper.
To run the algorithm on a custom network, modify main.cpp accordingly, rebuild, and run.
