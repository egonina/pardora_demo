import MySQLdb as mdb
import numpy as np
from whoosh.query import *
import time
import binascii
import array
import sqlite3

conn_str = '169.229.49.36', 'dbuser', 'p41msongs', 'milsongs'
SV_SIZE = 768

#=====================================
#          DB MANIPULATION 
#=====================================
# ===== SUPERVECTORS =====
def drop_sv_table():
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    c = conn.cursor()
    q = 'DROP TABLE songs1m_sv'
    c.execute(q)
    conn.commit()
    c.close()
    conn.close()

def create_sv_table():
    """
    Creates the file and an empty table.
    """
    # creates file
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    # add stuff
    c = conn.cursor()
    q = 'CREATE TABLE IF NOT EXISTS '
    q += 'songs1m_sv (song_id CHAR(18), '
    q += 'timbre_sv MEDIUMBLOB, '
    q += 'timbre_sv_shape_0 INT, '
    q += 'rhythm_sv MEDIUMBLOB, '
    q += 'rhythm_sv_shape_0 INT, '
    q += 'p_mean_t REAL, '
    q += 'p_sigma_t REAL, '
    q += 'p_mean_r REAL, '
    q += 'p_sigma_r REAL, '
    q += 'PRIMARY KEY (song_id)) ENGINE=NDBCLUSTER DEFAULT CHARSET=utf8;'
    c.execute(q)
    # commit and close
    conn.commit()
    c.close()
    conn.close()

def add_sv_and_p_vals_to_db(conn, c, song_id, t_sv, r_sv, \
                            p_mean_t, p_sigma_t, p_mean_r, p_sigma_r):
    # build query
    q = "INSERT INTO songs1m_sv VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

    t_sv_bin = sqlite3.Binary(t_sv)
    t_sv0 = int(t_sv.shape[0])
    
    r_sv_bin = sqlite3.Binary(r_sv)
    r_sv0 = int(r_sv.shape[0])

    insert_values = (song_id, t_sv_bin, t_sv0, r_sv_bin, r_sv0, \
                     p_mean_t, p_sigma_t, p_mean_r, p_sigma_r)
    
    c.execute(q, insert_values)

# ===== RHYTHM FEATURES =====
def drop_rhythm_table():
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    c = conn.cursor()
    q = 'DROP TABLE songs1m_rhythm'
    c.execute(q)
    conn.commit()
    c.close()
    conn.close()

def create_rhythm_table():
    """
    Creates the file and an empty table.
    """
    # creates file
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    # add stuff
    c = conn.cursor()
    q = 'CREATE TABLE IF NOT EXISTS '
    q += 'songs1m_rhythm (song_id CHAR(18), '
    q += 'rhythm_feats MEDIUMBLOB, '
    q += 'rhythm_shape_0 INT, '
    q += 'rhythm_shape_1 INT, '
    q += 'PRIMARY KEY (song_id)) ENGINE=NDBCLUSTER DEFAULT CHARSET=utf8;'
    c.execute(q)
    # commit and close
    conn.commit()
    c.close()
    conn.close()

def add_rhythm_feats_to_db(conn, c, song_id, r_feats):
    # build query
    q = "INSERT INTO songs1m_rhythm VALUES (%s, %s, %s, %s);"

    r_f_bin = sqlite3.Binary(r_feats)
    r_f0 = int(r_feats.shape[0])
    r_f1 = int(r_feats.shape[1])

    insert_values = (song_id, r_f_bin, r_f0, r_f1)
    
    c.execute(q, insert_values)

#=====================================
#          DB QUERYING 
#=====================================
def get_song_mta_data(artist=None, title=None):
    if title is None or artist is None:
        print "Need title and artist to get song MTA data"
        sys.exit()
    else:

        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()

        sql_query = 'SELECT song_id \
                     FROM songs1m WHERE title = "' + title.lower() + \
                     '" AND artist_name = "' + artist.lower() + '"'
        c.execute(sql_query)
        song_id = c.fetchall()

        song_id_q = str(song_id).strip('[]').replace("u", "").\
                     replace(",)", "").replace("(", "").replace(")", "")

        sql_query = "SELECT mode, tempo, artist_hottness, song_id \
                     FROM songs1m_mta WHERE song_id IN (" + song_id_q + ")"
        c.execute(sql_query)
        song_mta= c.fetchall()
        c.close()
        conn.close()

        for s in song_mta:
            mode = s[0]
            tempo = s[1]
            artist_hottness = s[2]

        return mode, tempo, artist_hottness 

def get_song_features_from_query(song_id_list):
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    c = conn.cursor()

    song_ids_str = str(song_id_list).strip('[]').replace("u", "").\
                 replace(",)", "").replace("(", "").replace(")", "")

    st = time.time()
    sql_query = "SELECT timbre_shape_0, timbre_shape_1, timbre_feats, artist_name, \
                title FROM songs1m WHERE song_id IN (" + song_ids_str + ")"
    c.execute(sql_query)
    timbre_result = c.fetchall()
    print "  TIME: SV GF, timbre:\t", time.time() - st

    st = time.time()
    sql_query = "SELECT rhythm_shape_0, rhythm_shape_1, rhythm_feats \
                FROM songs1m_rhythm WHERE song_id IN (" + song_ids_str + ")"
    c.execute(sql_query)
    rhythm_result = c.fetchall()
    print "  TIME: SV GF: rhythm:\t", time.time() - st

    c.close()
    
    print "  INFO: Num timbre results: ", len(timbre_result)
    print "  INFO: Num rhythm results: ", len(rhythm_result)
             
    conn.close()

    return timbre_result, rhythm_result

def get_song_svs_multi_query(song_id_list):
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    c = conn.cursor()

    ids = str(song_id_list).replace("u", "").replace("[", "").replace("]", "")

    sql_query = "SELECT timbre_sv, rhythm_sv, \
                 p_mean_t, p_mean_r, p_sigma_t, p_sigma_r, song_id \
                 FROM songs1m_sv WHERE song_id IN (" + ids + ")"
    c.execute(sql_query)
    song_data = c.fetchall()

    c.close()
    conn.close()

    total_dict = {}
    for s in song_data:
        song_id = s[6]
        total_dict[song_id] = {}
        total_dict[song_id]['q_t_sv'] = np.ndarray((SV_SIZE,),  buffer=s[0], dtype=np.float32)
        total_dict[song_id]['q_r_sv'] = np.ndarray((SV_SIZE,),  buffer=s[1], dtype=np.float32)
        total_dict[song_id]['p_mean_t'] = s[2]
        total_dict[song_id]['p_mean_r'] = s[3]
        total_dict[song_id]['p_sigma_t'] = s[4]
        total_dict[song_id]['p_sigma_r'] = s[5]

    return total_dict

def get_cf_songs_data(collab_song_info):
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    c = conn.cursor()

    ids = str(collab_song_info.keys()).replace("u", "").replace("[", "").replace("]", "")

    # get all song_ids
    sql_query = "SELECT title, artist_name, song_id \
                 FROM songs1m WHERE song_id IN (" + ids + ")"
    c.execute(sql_query)
    song_titles = c.fetchall()

    sql_query = "SELECT timbre_sv, rhythm_sv, \
                 p_mean_t, p_mean_r, p_sigma_t, p_sigma_r, song_id \
                 FROM songs1m_sv WHERE song_id IN (" + ids + ")"
    c.execute(sql_query)
    song_data = c.fetchall()

    sql_query = "SELECT mode, tempo, artist_hottness, song_id \
                 FROM songs1m_mta WHERE song_id IN (" + ids + ")"
    c.execute(sql_query)
    song_mta = c.fetchall()
    c.close()
    conn.close()

    total_dict = {}
    for s in song_data:
        song_id = s[6]
        total_dict[song_id] = {}
        total_dict[song_id]['t_sv'] = np.ndarray((SV_SIZE,),  buffer=s[0], dtype=np.float32)
        total_dict[song_id]['r_sv'] = np.ndarray((SV_SIZE,),  buffer=s[1], dtype=np.float32)
        total_dict[song_id]['p_mean_t'] = s[2]
        total_dict[song_id]['p_mean_r'] = s[3]
        total_dict[song_id]['p_sigma_t'] = s[4]
        total_dict[song_id]['p_sigma_r'] = s[5]
        total_dict[song_id]['cf_score'] = collab_song_info[song_id]

    for s in song_titles:
        song_id = s[2]
        if song_id in total_dict.keys():
            total_dict[song_id]['title'] = s[0]
            total_dict[song_id]['artist_name'] = s[1]

    for s in song_mta:
        song_id = s[3]
        if song_id in total_dict.keys():
            total_dict[song_id]['mode'] = s[0]
            total_dict[song_id]['tempo'] = s[1]
            total_dict[song_id]['artist_hottness'] = s[2]

    return total_dict

def get_song_ids_from_title_artist_pairs(song_list):
    conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
    c = conn.cursor()

    # construct artist title query strings
    title_artist_string = '('
    for pair in song_list[:-1]:
        title_artist_string += ' title = "' + str(pair[1]) + '" AND artist_name = "' + str(pair[0]) + '") OR ('

    pair = song_list[-1]
    title_artist_string += ' title = "' + str(pair[1]) + '" AND artist_name = "' + str(pair[0]) + '")'

    sql_query = 'SELECT song_id \
                 FROM songs1m WHERE ' + title_artist_string 

    c.execute(sql_query)
    song_ids = c.fetchall()

    song_id_list = []
    for s in song_ids:
        song_id_list.append(s[0])

    if len(song_id_list) > 0:
        return song_id_list
    else:
        return None
