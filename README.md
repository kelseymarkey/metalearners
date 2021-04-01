# CATE Metalearners

Inspired by the [ML Reproducibility Challenge](https://paperswithcode.com/rc2020), this project aims to reproduce synthetic data experiments conducted in the paper ["Metalearners for estimating heterogeneous treatment effects using machine learning"](https://www.pnas.org/content/pnas/116/10/4156.full.pdf) by Soren R. Kunzel, Jasjeet S. Sekhona, Peter J. Bickela, and Bin Yu. 

To replicate these results we make adjustments to code put forth by the original authors in their repository, [causalToolbox](https://github.com/soerenkuenzel/causalToolbox). We make no claims of ownership towards this code and any futher uses should make reference to the original paper.


### Data Generation
To generate synethetic data, run the following from the command line:
```bash
run_sims.sh # within repo root directory
```
This will generate 30 samples of each simulation (A, C, and D). Each sample will contain 3000 rows in train, and 1000 rows in test. Each train and test frame will be saved as a CSV, and any necessary directories will be created.
