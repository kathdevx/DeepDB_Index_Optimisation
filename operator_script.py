from index_optimisation import IndexOptimiser
import glob


class Operator:
    def __init__(self, dataset, db, args, optimisation_method):
        self.args = args
        self.args.dataset = dataset
        self.args.database_name = db
        self.optimised = ''
        opt_options = ['rdc', 'query_based', 'auto']
        if optimisation_method is not None and optimisation_method.lower() in opt_options:
            self.optimised = '_opt_method_' + optimisation_method.lower()
        if dataset == 'imdb-light':
            self.args.csv_path = '/home/kate/Downloads/imdb'
            self.args.hdf_path = 'fixed_imdb_data/gen_single_light'
            self.args.ensemble_path = 'fixed_imdb_data/'
            self.args.pairwise_rdc_path = 'fixed_imdb_data/'
            self.args.csv_seperator = ','
            self.benchmark_folder = 'benchmarks/job-light'
        else:
            self.args.csv_path = 'ssb_data'
            self.args.hdf_path = 'ssb_data/gen_hdf'
            self.args.csv_seperator = '|'
            self.args.ensemble_path = 'ssb_data/'
            self.args.pairwise_rdc_path = 'ssb_data/'
            self.benchmark_folder = 'benchmarks/ssb'

    def generate_hdf_files(self):
        self.args.generate_hdf = True
        if self.args.dataset == 'ssb-500gb':
            return self.args
        self.args.max_rows_per_hdf_file = 1000000
        return self.args

    def generate_hdf_samples(self):
        self.args.generate_sampled_hdfs = True
        self.args.max_rows_per_hdf_file = 1000000
        self.args.hdf_sample_size = 10000
        return self.args

    def generate_ensembles_single(self):
        self.args.generate_ensemble = True
        self.args.samples_per_spn = [10000000]  # 10M
        self.args.ensemble_strategy = 'single'
        self.args.ensemble_path += 'spn_ensembles_single' + self.optimised
        self.args.rdc_threshold = 0.3
        self.args.post_sampling_factor = [10]
        return self.args

    def generate_ensembles_relationship(self):
        self.args.generate_ensemble = True
        self.args.samples_per_spn = [1000000, 1000000, 1000000, 1000000, 1000000]
        self.args.ensemble_strategy = 'relationship'
        self.args.ensemble_path += 'spn_ensembles_relationship' + self.optimised
        self.args.max_rows_per_hdf_file = 1000000
        self.args.post_sampling_factor = [10, 10, 5, 1, 1]
        return self.args

    def generate_ensembles_rdc(self):
        self.args.generate_ensemble = True
        self.args.samples_per_spn = [10000000, 10000000, 10000000, 10000000, 10000000]
        self.args.ensemble_strategy = 'rdc_based'
        self.args.ensemble_path += 'spn_ensembles_rdc' + self.optimised
        self.args.max_rows_per_hdf_file = 1000000
        self.args.samples_rdc_ensemble_tests = 10000
        self.args.post_sampling_factor = [10, 10, 5, 1, 1]
        if self.args.dataset == 'imdb-light':
            self.args.ensemble_budget_factor = 5
            self.args.query_file_location = self.benchmark_folder + '/sql/job_light_queries.sql'
        else:
            self.args.ensemble_budget_factor = 0
        self.args.ensemble_max_no_joins = 3
        self.args.pairwise_rdc_path = self.args.ensemble_path + '/pairwise_rdc.pkl'
        return self.args

    def aqp_ground_truth_queries(self, explain, query_joins=False):
        self.args.aqp_ground_truth = True
        if explain:
            self.args.query_file_location = self.benchmark_folder + '/sql/explain_queries.sql'
            self.args.target_path = self.benchmark_folder + '/ground_truth_explain' + self.optimised + '.pkl'
        else:
            self.args.query_file_location = self.benchmark_folder + '/sql/aqp_queries' + '.sql'
            self.args.target_path = self.benchmark_folder + '/ground_truth_katerina' + self.optimised + '.pkl'
        if query_joins:
            self.args.query_file_location = self.benchmark_folder + '/sql/join_queries.sql'
            self.args.target_path = self.benchmark_folder + '/ground_truth_katerina_joins' + self.optimised + '.pkl'
        return self.args

    def aqp_ground_truth_confidence_interval(self):
        self.args.aqp_ground_truth = True
        self.args.query_file_location = self.benchmark_folder + '/sql/confidence_queries.sql'
        self.args.target_path = self.benchmark_folder + '/confidence_intervals/confidence_intervals_katerina' + \
                                self.optimised + '.pkl'
        return self.args

    def evaluate_cardinalities(self, ensemble_type):
        self.args.evaluate_cardinalities = True
        self.args.max_variants = 1
        if ensemble_type == 'rdc':
            self.args.rdc_spn_selection = True
            self.args.pairwise_rdc_path = 'fixed_imdb_data/spn_ensembles_' + ensemble_type + self.optimised \
                                          + '/pairwise_rdc.pkl'
        self.args.target_path = 'baselines/cardinality_estimation/results/deepDB/imdb_light_model_based_budget_5_katerina_' \
                                + ensemble_type + self.optimised + '.csv'
        for fname in glob.glob('fixed_imdb_data/spn_ensembles_' + ensemble_type + self.optimised + '/ensemble*'):
            self.args.ensemble_location = fname
        self.args.query_file_location = self.benchmark_folder + '/sql/job_light_queries.sql'
        self.args.ground_truth_file_location = self.benchmark_folder + '/sql/job_light_true_cardinalities.csv'
        return self.args

    def evaluate_aqp_queries(self, model_name, ensemble_type):
        self.args.evaluate_aqp_queries = True
        self.args.target_path = 'baselines/aqp/results/deepDB/' + model_name + self.optimised + '_aqp.csv'
        self.args.ground_truth_file_location = self.benchmark_folder + '/ground_truth_katerina.pkl'
        self.args.query_file_location = self.benchmark_folder + '/sql/aqp_queries.sql'

        if ensemble_type.find('ssb') != -1:
            self.args.ensemble_location = self.args.csv_path + '/spn_ensembles_' + ensemble_type + self.optimised + \
                                          '/ensemble_single_ssb-500gb_10000000.pkl'
        else:
            if ensemble_type == 'single':
                self.args.ensemble_location = self.args.csv_path + '/spn_ensembles_' + ensemble_type + self.optimised \
                                              + '/ensemble_single_ssb-500gb_10000000.pkl'
            elif ensemble_type == 'rdc':
                self.args.ensemble_location = self.args.csv_path + '/spn_ensembles_' + ensemble_type + self.optimised \
                                              + '/ensemble_join_3_budget_0_10000000.pkl'
        return self.args

    def evaluate_confidence_intervals(self, model_name, ensemble_type):
        self.args.evaluate_confidence_intervals = True
        self.args.target_path = 'baselines/aqp/results/deepDB/' + model_name + '_' + ensemble_type + self.optimised +\
                                '_intervals.csv'
        self.args.ground_truth_file_location = self.benchmark_folder + \
                                               '/confidence_intervals/confidence_intervals_katerina.pkl'
        if ensemble_type == 'rdc':
            self.args.ensemble_location = self.args.csv_path + '/spn_ensembles_' + ensemble_type + self.optimised + \
                                          '/ensemble_join_3_budget_0_10000000.pkl'
        else:
            self.args.ensemble_location = self.args.csv_path + '/spn_ensembles_' + ensemble_type + self.optimised + \
                                          '/ensemble_single_ssb-500gb_10000000.pkl'
        self.args.query_file_location = self.benchmark_folder + '/sql/aqp_queries.sql'
        self.args.confidence_upsampling_factor = 300
        self.args.confidence_sample_size = 10000000
        return self.args

    def drop_indexes(self):
        idx_opt = IndexOptimiser(db_name=self.args.database_name)
        idx_opt.drop_indexes()
        print(f'Indexes Droped from {self.args.database_name} database')

    def optimise_indexes(self, method, rdc_path='path', n=2):
        idx_opt = IndexOptimiser(db_name=self.args.database_name)

        # Utilise column correlations to find most frequent columns in dataset
        if method.lower() == 'rdc':
            idx_opt.get_most_frequent_columns_from_correlations(rdc_path, n)
            idx_opt.create_rdc_indexes()
        # Build indexes based on queries and what they use most
        elif method.lower() == 'query_based':
            # self.args.query_file_location = self.benchmark_folder + '/sql/aqp_queries.sql'
            idx_opt.create_indexes_from_candidates(query_file=self.args.query_file_location)
        # Locate similar fields in tables and index them to speed up joins
        elif method.lower() == 'auto':
            idx_opt.create_indexes_from_candidates(auto=True)

        # Show indexes that were created
        self.get_db_indexes()

    def get_db_indexes(self):
        idx_opt = IndexOptimiser(db_name=self.args.database_name)
        db_idxs = idx_opt.scan_indexes()
