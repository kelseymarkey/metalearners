# CATE Metalearners

Inspired by the [ML Reproducibility Challenge](https://paperswithcode.com/rc2020), this project aims to reproduce synthetic data experiments conducted in the paper ["Metalearners for estimating heterogeneous treatment effects using machine learning"](https://www.pnas.org/content/pnas/116/10/4156.full.pdf) by Soren R. Kunzel, Jasjeet S. Sekhona, Peter J. Bickela, and Bin Yu. 

To replicate these results we make adjustments to code put forth by the original authors in their repository, [causalToolbox](https://github.com/soerenkuenzel/causalToolbox). We make no claims of ownership towards this code and any futher uses should make reference to the original paper.


### Data Generation
To generate synethetic data, clone this repository into your home directory on NYU's Greene HPC cluster. Create a file named "netid.txt" within the configurations/ directory, and save your NYU netID in it. Then run the following from the command line within the repository root directory:
```bash
sbatch --array=1-30 run_sims.sh
```
This will generate 30 samples of each simulation (A, B, C, D, E, and F), each with 300,000 training observations and 100,000 test observations. Each train and test frame will be saved as a Parquet file in /scratch storage, and any necessary directories will be created.
