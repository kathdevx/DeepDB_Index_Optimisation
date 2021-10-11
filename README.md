# DeepDB with Index Optimisation

Reproduction of the DeepDB experiments and setups and further optimisation of the model by using indexes

**DeepDB** is the work of Benjamin Hilprecht, Andreas Schmidt, Moritz Kulessa, Alejandro Molina, Kristian Kersting, Carsten Binnig: "DeepDB: Learn from Data, not from Queries!", VLDB'2020\. [[PDF]](https://arxiv.org/abs/1909.00607)

We further explore this work by utilising indexes and measuring their effect in the overall performance of DeepDB.

To reproduce the experiments, you will have to first install Postgres Database and import the data. Make sure the file paths are correct when running the code. There are many files used in this code, very easy to get lost. For my experiments, I named the folder for the IMDB dataset fixed_imdb_data since some refining had to be made for them to be inserted in the database, and for the SSB dataset, I named the folder ssb_data.

**Use demo_script.py for user friendly experiment reproduction** <br>
``
python3 demo_script.py --run_experiment n
`` <br>
where n is the number of the experiment you want to run.

The experiments were run on a laptop, i7 cpu with 8 GB of RAM, Ubuntu

Datasets that were explored for this project:

1. IMDB Dataset - Download: <http://homepages.cwi.nl/~boncz/job/imdb.tgz>)
2. SSB dataset (Scale Factor=100) - Download: <https://github.com/eyalroz/ssb-dbgen>

Experiments:

<li> IMDB Dataset </li>

  1. Experiment 1 - Reproduction

    1. Build Relationship Ensemble (naive way)
    2. Evaluate Cardinalities

  2. Experiment 2 - Reproduction

    1. Build RDC Ensemble
    3. Evaluate Cardinalities

  3. Experiment 3 - RDC Optimisation

    1. Drop other indexes
    2. Build RDC indexes
    3. Generate RDC Ensemble
    4. Evaluate Cardinalities

  4. Experiment 4 - Query-based Optimisation

    1. Drop other indexes
    2. Build query-based indexes
    3. Generate RDC Ensemble
    4. Evaluate Cardinalities

  5. Experiment 5 - Auto Optimisation

    1. Drop other indexes
    2. Build auto indexes
    3. Generate RDC Ensemble
    4. Evaluate Cardinalities

<li> SSB Dataset </li>

  1. Experiment 6 - Reproduction

    1. Build Single Ensemble (naive way)
    2. Calculate AQP ground truth queries
    3. Evaluate AQP queries of model
    4. Calculate Confidence Intervals
    5. Evaluate Confidence Intervals of model

  2. Experiment 7 - Reproduction

    1. Build RDC Ensemble
    2. Calculate ground truth for AQP
    3. Evaluate AQP queries of model
    4. Calculate ground truth for Confidence Intervals
    5. Evaluate Confidence Intervals of model

  3. Experiment 8 - RDC Optimisation

    1. Drop other indexes
    2. Build RDC indexes
    3. Generate RDC Ensemble
    4. Generate Single Ensemble
    5. Evaluate AQP queries of models
    6. Evaluate Confidence Intervals of models

  4. Experiment 9 - Query-based Optimisation

    1. Drop other indexes
    2. Build Query-based indexes
    3. Generate RDC Ensemble
    4. Generate Single Ensemble
    5. Evaluate AQP queries of models
    6. Evaluate Confidence Intervals of models

  5. Experiment 10 - Auto Optimisation

    1. Drop other indexes
    2. Build auto indexes
    3. Generate RDC Ensemble
    4. Generate Single Ensemble
    5. Evaluate AQP queries of models
    6. Evaluate Confidence Intervals of models
