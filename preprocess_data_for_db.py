import math

import pandas as pd

df = None
# table = {
#     0: {'name': 'aka_name', 'integers': [0, 1], 'nulls': [0, 1]},
#     1: {'name': 'aka_title', 'integers': [0, 1, 4, 5, 7, 8, 9], 'nulls': [0, 1, 4]},
#     2: {'name': 'cast_info', 'integers': [0, 1, 2, 3, 5, 6], 'nulls': [0, 1, 2, 6]},
#     3: {'name': 'char_name', 'integers': [0, 3], 'nulls': [0, 1]},
#     4: {'name': 'comp_cast_type', 'integers': [0], 'nulls': [0, 1]},
#     5: {'name': 'company_name', 'integers': [0, 3], 'nulls': [0, 1]},
#     6: {'name': 'company_type', 'integers': [0], 'nulls': [0]},
#     7: {'name': 'complete_cast', 'integers': [0, 1, 2, 3], 'nulls': [0, 2, 3]},
#     8: {'name': 'info_type', 'integers': [0], 'nulls': [0, 1]},
#     9: {'name': 'keyword', 'integers': [0], 'nulls': [0, 1]},
#     10: {'name': 'kind_type', 'integers': [0], 'nulls': [0]},
#     11: {'name': 'link_type', 'integers': [0], 'nulls': [0, 1]},
#     12: {'name': 'movie_companies', 'integers': [0, 1, 2, 3], 'nulls': [0, 1, 2, 3]},
#     13: {'name': 'movie_info_idx', 'integers': [0, 1, 2], 'nulls': [0, 1, 2, 3]},
#     14: {'name': 'movie_keyword', 'integers': [0, 1, 2], 'nulls': [0, 1, 2]},
#     15: {'name': 'movie_link', 'integers': [0, 1, 2, 3], 'nulls': [0, 1, 2, 3]},
#     16: {'name': 'name', 'integers': [0, 3], 'nulls': [0, 1]},
#     17: {'name': 'role_type', 'integers': [0], 'nulls': [0, 1]},
#     18: {'name': 'movie_info', 'integers': [0, 1, 2], 'nulls': [0, 1, 2, 3]},
#     19: {'name': 'person_info', 'integers': [0, 1, 2], 'nulls': [0, 1, 2, 3]}
# }
#
# for n, _ in enumerate(table):
#     df = pd.read_csv('/home/kate/Downloads/imdb/' + table[n]['name'] + '.csv', header=None, escapechar='\\',
#                      encoding='utf-8', quotechar='"', sep=',')
#
#     # ---------- TRANSFORM VALUES TO INTEGERS --------------
#     for i in table[n]['integers']:
#         df[i] = df[i].apply(lambda x: int(x) if x == x else "")
#
#     # --------- DROP NULL VALUES WHERE THEY ARE NOT PERMITTED BY THE SCHEMA -----------
#     for i in table[n]['nulls']:
#         df = df[df[i].notna()]
#
#     df.to_csv('fixed_imdb_data/' + table[n]['name'] + '.csv', index=None, header=None, escapechar='\\',
#               encoding='utf-8', quotechar='"', sep=',')
#     print(f'{n} done!')

# ----------------------- SPECIAL CASE: TITLE -----------------------
dtypes = {
    'id': int,
    'kind_id': int,
    'production_year': int,
    'imdb_id': int,
    'episode_of_id': int,
    'season_nr': int,
    'episode_nr': int
}

df = pd.read_csv('/home/kate/Downloads/imdb/title.csv', dtype=object, header=None, escapechar='\\', encoding='utf-8', quotechar='"',
                 sep=',')

# ---------- TRANSFORM VALUES TO INTEGERS --------------
for i in [0, 3, 4, 5, 7, 8, 9]:
    temp = []
    for value in df[i]:
        str_value = str(value)
        if str_value.find('.') != -1:
            int_like = str_value.find('.')
            temp.append(int(str_value[:int_like]))
        # elif math.isnan(value) or value == 'F624':
        #     temp.append(int(float(str_value)))
        elif str_value == 'F624':
            temp.append('')
        else:
            temp.append(value)
    print(df[i][0], temp[0])
    # df.loc[:, i] = [0] * len(df.loc[:, i])
    # df = df.astype({i: int})
    df.loc[:, i] = temp
    print(df[i][0])
    print(f'{i} done!')

# --------- DROP NULL VALUES WHERE THEY ARE NOT PERMITTED BY THE SCHEMA -----------
for i in [0, 1, 3]:
    df = df[df[i].notna()]

df.to_csv('fixed_imdb_data/title.csv', index=None, header=None, escapechar='\\', encoding='utf-8', quotechar='"',
          sep=',')
