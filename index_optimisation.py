import pickle
from collections import Counter
import sqlparse
from more_itertools import flatten
from ensemble_compilation.physical_db import DBConnection
import numpy as np


class IndexOptimiser:

    def __init__(self, db_name):
        self.db_name = db_name
        self.db_connection = DBConnection(db=db_name)
        self.db_indexes = set()
        self.db_tables = []
        self.indexes = set()
        self.table_join_indexes = None

    def get_most_frequent_columns_from_correlations(self, rdc_path, n=5):
        with open(rdc_path, 'rb') as f:
            column_correlations = pickle.load(f)

        column_correlations_without_nan = {k: column_correlations[k] for k in column_correlations
                                           if not np.isnan(column_correlations[k])}
        top_n_correlated_columns = sorted(column_correlations_without_nan, key=column_correlations_without_nan.get,
                                          reverse=True)[:n]

        all_keys = list(flatten(column_correlations_without_nan.keys()))
        c = Counter(all_keys)
        top_relationships = c.most_common(n)

        for idx, _ in top_relationships:
            self.indexes.add(idx)
        print(self.indexes)

    def get_db_tables(self):
        # scan database to get all available tables
        get_tables_query = r"select tablename " \
                           r"from pg_catalog.pg_tables " \
                           r"where schemaname != 'pg_catalog' " \
                           r"and schemaname != 'information_schema';"
        result_tables = self.db_connection.get_result_set(get_tables_query)

        for table in result_tables:
            self.db_tables.append(table[0])
        return self.db_tables

    def scan_indexes(self):
        self.db_tables = self.get_db_tables()
        tmp = set()
        for table in self.db_tables:
            get_index_query = r"select indexname " \
                              r"from pg_indexes " \
                              r"where tablename ='" + table + \
                              "' and indexname not like '%pkey';"
            res = self.db_connection.get_result_set(get_index_query)
            if res:
                for item in res:
                    # tmp.add(item)
                    self.db_indexes.add(item[0])

        # for item in tmp:
        #     self.db_indexes.add(item)
        print(f'Available indexes in {self.db_connection.db} : {self.db_indexes}')
        return self.db_indexes

    def drop_indexes(self):
        print('Scanning database for indexes..')
        self.db_indexes = self.scan_indexes()

        for index_name in self.db_indexes:
            drop_index_query = r"drop index " + index_name + ";"
            print(f'Running {drop_index_query}')
            self.db_connection.submit_query(drop_index_query)

        print(r"DROP INDEX operation finished. Scanning again..")
        self.db_indexes = self.scan_indexes()

    def create_rdc_indexes(self):
        # continuously reloading db_indexes to be sure we're up to date
        primary_keys = self.get_primary_keys()
        self.db_indexes = self.scan_indexes()
        while len(self.db_indexes) != len(self.indexes):  # not all indexes were created properly
            for idx in self.indexes:
                table, column = idx.split('.')
                db_like_idx = 'idx_' + column + '_' + table
                self.db_indexes = self.scan_indexes()
                if db_like_idx not in self.db_indexes:
                    create_idx_query = r"create index idx_" + column + '_' + table + \
                                       " on " + table + "(" + column + ");"
                    self.db_connection.submit_query(create_idx_query)
            self.db_indexes = self.scan_indexes()

    def create_indexes_from_candidates(self, auto=False, n=2, query_file=None):
        if auto:
            indexes = self.locate_similar_fields()
            self.db_indexes = self.scan_indexes()
            for table, columns in indexes.items():
                for column in columns:
                    new_idx = 'idx_' + column + "_" + table
                    if new_idx not in self.db_indexes:
                        create_index_query = r"create index " + new_idx + \
                                             " on " + table + " (" + column + ");"
                        self.db_connection.submit_query(create_index_query)
                        print(f'Auto: Done creating {new_idx} index')
        else:
            primary_keys = self.get_primary_keys()
            relationships = self.candidate_index_fields(query_file, n)
            candidate_fields = set()
            for r in relationships:
                r1, r2 = r.split(' = ')
                candidate_fields.add(r1)
                candidate_fields.add(r2)

            for table in self.db_tables:
                get_table_columns_query = r"select column_name " \
                                          r"from information_schema.columns " \
                                          r"where table_name = '" + table + "';"
                table_columns = self.db_connection.get_result_set(get_table_columns_query)

                for candidate in candidate_fields:
                    try:
                        pkey = primary_keys[table]
                    except KeyError:
                        pkey = 'random_name_woohooo'
                    if candidate not in self.db_indexes and candidate != pkey and candidate in table_columns:
                        new_idx = 'idx_' + candidate + '_' + table
                        create_index_query = r"create index " + new_idx + \
                                             " on " + table + " (" + candidate + ");"
                        self.db_connection.submit_query(create_index_query)
                        print(f'Manual: Done creating {new_idx} index')

    def get_primary_keys(self):
        self.db_tables = self.get_db_tables()
        primary_keys = {}
        for table_name in self.db_tables:
            get_primary_key_query = r"select a.attname " \
                                    r"from pg_index i " \
                                    r"join pg_attribute a on a.attrelid = i.indrelid " \
                                    r"and a.attnum = any(i.indkey) " \
                                    r"where i.indrelid='" + table_name + "'::regclass " \
                                    r"and i.indisprimary;"
            try:
                pkey = self.db_connection.get_result_set(get_primary_key_query)[0][0]
                primary_keys[table_name] = pkey
            except IndexError:
                pass
        return primary_keys

    def locate_similar_fields(self):
        primary_keys = self.get_primary_keys()
        similar_key = {}
        if self.db_name == 'ssb':
            for table_name in self.db_tables:
                temp_keys = set()
                for pkey in primary_keys.values():
                    locate_field_query = r"select column_name " \
                                         r"from information_schema.columns " \
                                         r"where table_name='" + table_name + "' and " \
                                         r"column_name like '%" + pkey.split('_')[1] + "';"
                    try:
                        result = self.db_connection.get_result_set(locate_field_query)[0][0]
                        temp_keys.add(result)
                    except IndexError:
                        pass
        elif self.db_name == 'imdb_light':
            for table_name in self.db_tables:
                temp_keys = set()
                for pkey in primary_keys.values():
                    locate_field_query = r"select column_name " \
                                         r"from information_schema.columns " \
                                         r"where table_name='" + table_name + "' and " \
                                         r"column_name like '%" + pkey + "';"
                    try:
                        retrieved_fields = self.db_connection.get_result_set(locate_field_query)
                        for field in retrieved_fields:
                            if field[0] not in primary_keys.values():
                                temp_keys.add(field[0])
                    except IndexError:
                        pass

        for tkey in temp_keys:
            try:
                pkey_of_table = primary_keys[table_name]
            except KeyError:
                pkey_of_table = 'random_name_wooohoo'
            if tkey != pkey_of_table:
                try:
                    similar_key[table_name].add(tkey)
                except:
                    similar_key[table_name] = set()
                    similar_key[table_name].add(tkey)
        self.table_join_indexes = similar_key
        return self.table_join_indexes

    def candidate_index_fields(self, query_file, n=2):
        parsed = []
        with open(query_file, 'r') as f:
            queries = f.read().splitlines()
        for query in queries:
            parsed.append(sqlparse.parse(query))

        blacklist = ['', ' ', 'WHERE', 'AND', 'IN', ';']
        relationship = {}
        for i, _ in enumerate(parsed):
            for j, _ in enumerate(parsed[i][0].tokens):
                try:  # go through sql structure of query to locate WHERE clauses
                    if parsed[i][0].tokens[j].M_OPEN[1] == 'WHERE':
                        for comparison in parsed[i][0].tokens[j]:
                            if comparison.normalized not in blacklist and not comparison.normalized.startswith("("):
                                try:
                                    relationship[comparison.normalized] += 1
                                except KeyError:
                                    relationship[comparison.normalized] = 1
                except AttributeError:
                    continue
        #  store filters, e.g. t1.column1 = t2.column2 to use for indexing to speed up query
        candidates = sorted(relationship, key=relationship.get, reverse=True)[:n]
        return candidates
