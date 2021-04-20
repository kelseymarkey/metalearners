# CATE Metalearners

Inspired by the [ML Reproducibility Challenge](https://paperswithcode.com/rc2020), this project aims to reproduce synthetic data experiments conducted in the paper ["Metalearners for estimating heterogeneous treatment effects using machine learning"](https://www.pnas.org/content/pnas/116/10/4156.full.pdf) by Soren R. Kunzel, Jasjeet S. Sekhona, Peter J. Bickela, and Bin Yu. 

To replicate these results we make adjustments to code put forth by the original authors in their repository, [causalToolbox](https://github.com/soerenkuenzel/causalToolbox). We make no claims of ownership towards this code and any futher uses should make reference to the original paper.


### Data Generation
To generate synethetic data, run the following from the command line:
```bash
run_sims.sh # within repo root directory
```
This will generate 30 samples of each simulation (A, B, C, D, E, and F) for each training set size in `[5000, 10000, 20000, 100000, 300000]`. Each sample will also be accompanied by a 100,000 row test dataset. Each train and test frame will be saved as a Parquet file, and any necessary directories will be created.
