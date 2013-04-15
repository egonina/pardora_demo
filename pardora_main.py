from gmm_specializer.gmm import *
import MySQLdb as mdb
import pickle
import time
import binascii
import array
import sqlite3
import msdtools
import unicodedata
import collab

from whoosh.query import *
 
ubm_t_feats_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_features_1M_008.pkl"
ubm_r_feats_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_rhythm_features_1M_008.pkl"
ubm_t_params_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_params.pkl"
ubm_r_params_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_rhythm_params.pkl"
song_id_pkl = "/disk1/home_user/egonina/msd_database/pickles/song_ids_1M.pkl"
#song_id_pkl = "/disk1/home_user/egonina/msd_database/pickles/song_ids_short_1M.pkl"
norm_param_pkl = "/disk1/home_user/egonina/msd_database/pickles/norm_param_pkl_1M.pkl"
song_ids_with_rhythm = "/disk1/home_user/egonina/msd_database/pickles/rhythm_ids_1M.pkl"

conn_str = '169.229.49.36', 'dbuser', 'p41msongs', 'milsongs'
CHUNK_SIZE = 50 
NORM_CHUNK_SIZE = 1000
M = 64
SV_SIZE = 768

class Pardora:
    #=====================================
    #         HELPERS 
    #=====================================
    def chunks(self, l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]

    #=====================================
    #          DB MANIPULATION 
    #=====================================
    # ===== SUPERVECTORS =====
    def drop_sv_table(self):
        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()
        q = 'DROP TABLE songs1m_sv'
        c.execute(q)
        conn.commit()
        c.close()
        conn.close()

    def create_sv_table(self):
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

    def add_sv_and_p_vals_to_db(self, conn, c, song_id, t_sv, r_sv, \
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
    def drop_rhythm_table(self):
        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()
        q = 'DROP TABLE songs1m_rhythm'
        c.execute(q)
        conn.commit()
        c.close()
        conn.close()

    def create_rhythm_table(self):
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

    def add_rhythm_feats_to_db(self, conn, c, song_id, r_feats):
        # build query
        q = "INSERT INTO songs1m_rhythm VALUES (%s, %s, %s, %s);"

        r_f_bin = sqlite3.Binary(r_feats)
        r_f0 = int(r_feats.shape[0])
        r_f1 = int(r_feats.shape[1])

        insert_values = (song_id, r_f_bin, r_f0, r_f1)
        
        c.execute(q, insert_values)
        
    #=====================================
    #         RHYTHM COMPUTATION 
    #=====================================

    def compute_and_add_rhythm_feats(self):
        print "............... Computing and Adding Rhythm Features To DB ...................."
        p = open(song_id_pkl, "rb")    
        song_ids = pickle.load(p)
        p.close()

        song_id_chunks = self.chunks(song_ids, CHUNK_SIZE)
        chunk_count = 0

        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()

        # get all song_ids
        sql_query = "SELECT song_id FROM songs1m_rhythm;"
        c.execute(sql_query)
        all_songs = c.fetchall()
        all_songs_list = []
        for s in all_songs:
            s = s[0]
            all_songs_list.append(s)

        total_time = time.time()
        for chunk in song_id_chunks:
            if chunk_count > 2908:
                print "==== CHUNK: ", chunk_count, " ===="
                st = time.time()

                song_ids = str(chunk).strip('[]').replace("u", "").replace(",)", "").replace("(", "")
                sql_query = "SELECT timbre_feats, segments_start, timbre_shape_0, \
                     timbre_shape_1, sstart_shape_0, song_id FROM songs1m WHERE song_id IN (" \
                     + song_ids + ")"
                c.execute(sql_query)
                songs = c.fetchall()
                
                for s in songs:
                    t_feats =  np.ndarray((s[2],s[3]), buffer=s[0])
                    segments =  np.ndarray((s[4],), buffer=s[1])
                    song_id = s[5]

                    #compute rhythm features
                    onset_coefs, onset_pattern = msdtools.rhythm_features(t_feats, segments)

                    if onset_coefs is not None:
                        if song_id not in all_songs_list:
                            self.add_rhythm_feats_to_db(conn, c, song_id, onset_coefs)
                            all_songs_list.append(song_id)

                print "INFO: chunk rhythm compute time:", time.time() - st
                conn.commit()

            chunk_count += 1 

        print "================================"
        print "INFO: TOTAL TIME FOR RHYTHM COMP TIME: ", time.time() - total_time 
        print "================================"

        c.close()
        conn.close()
        return

    #=====================================
    #       SUPERVECTOR COMPUTATION 
    #=====================================

    def compute_and_add_song_svs(self):
        print "............... Computing and Adding Supervectors To DB ...................."
        p = open(song_id_pkl, "rb")    
        song_ids = pickle.load(p)
        p.close()

        song_id_chunks = self.chunks(song_ids, CHUNK_SIZE)
        chunk_count = 0

        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()

        t_mean_to_use = np.zeros(1)
        t_sv = np.zeros(1)
        r_mean_to_use = np.zeros(1)
        r_sv = np.zeros(1)

        all_songs_list = []

        total_time = time.time()
        for chunk in song_id_chunks:
              song_id_list = []
              timbre_sv_arr = []
              rhythm_sv_arr = []
              print "==== CHUNK: ", chunk_count, "===="
              chunk_count+=1

              chunk_time = time.time()

              song_ids = str(chunk).strip('[]').replace("u", "").replace(",)", "").replace("(", "")
              
              st = time.time()
              sql_query = "SELECT rhythm_feats, rhythm_shape_0, rhythm_shape_1, song_id \
                           FROM songs1m_rhythm \
                           WHERE song_id IN ("+song_ids+")"
              c.execute(sql_query)
              songs = c.fetchall()
              
              for s in songs:
                  if s[1] is not None and s[2] is not None:
                      feats =  np.array(np.ndarray((s[1],s[2]), buffer=s[0]), dtype=np.float32)
                      feats_t = feats.T
                      rhythm_sv = self.adapt_model(feats_t, self.rhythm_ubm_params, M)
                      rhythm_sv_arr.append(rhythm_sv)
                      song_id_list.append(s[3])

              print "INFO: Rhythm SV comp time: ", time.time() - st

              st = time.time()
              sql_query = "SELECT timbre_feats, timbre_shape_0, timbre_shape_1, song_id FROM songs1m \
                          WHERE song_id IN ("+song_ids+")"
              c.execute(sql_query)
              songs = c.fetchall()

              for s in songs:
                  if s[3] in song_id_list:
                      feats =  np.array(np.ndarray((s[1],s[2]), buffer=s[0]), dtype=np.float32)
                      timbre_sv = self.adapt_model(feats, self.timbre_ubm_params, M)
                      timbre_sv_arr.append(timbre_sv)

              print "INFO: Timbre SV comp time: ", time.time() - st
              
              st = time.time()
              t_sv = np.vstack(timbre_sv_arr)
              del timbre_sv_arr
              r_sv = np.vstack(rhythm_sv_arr)
              del rhythm_sv_arr

              if chunk_count == 1:
                  t_mean_to_use = np.mean(t_sv, axis=0)
                  r_mean_to_use = np.mean(r_sv, axis=0)

              t_sv = msdtools.mcs_norm(t_sv.T, t_mean_to_use).T
              r_sv = msdtools.mcs_norm(r_sv.T, r_mean_to_use).T
              
              print "INFO: MCS norm computation time: ", time.time() - st

              st = time.time()
              p_means_t = np.zeros(len(song_id_list))
              p_sigmas_t = p_means_t.copy()
              p_means_t = p_means_t.copy() 
              p_sigmas_t = p_means_t.copy()

              st = time.time()

              p_means_t, p_sigmas_t = msdtools.p_norm_params_chunk(t_sv.T, t_sv.T, NORM_CHUNK_SIZE)
              p_means_r, p_sigmas_r = msdtools.p_norm_params_chunk(r_sv.T, r_sv.T, NORM_CHUNK_SIZE)

              print "INFO: P-means computation time: ", time.time() - st


              st = time.time()
              idx = 0
              for s_id in song_id_list:
                  t = np.array(t_sv[idx])
                  r = np.array(r_sv[idx])
                  if s_id not in all_songs_list:
                      self.add_sv_and_p_vals_to_db(conn, c, s_id, t, r, \
                                               p_means_t[idx], p_sigmas_t[idx], 
                                               p_means_r[idx], p_sigmas_r[idx])
                      all_songs_list.append(s_id)
                  idx += 1

              conn.commit()

              print "INFO: Database update time: ", time.time() - st

              print "INFO: TOTAL CHUNK TIME:", time.time() - chunk_time

        print "=============================================="
        print "INFO: TOTAL TIME FOR SV COMP TIME: ", time.time() - total_time
        print "=============================================="
        d = {}   
        d['t_sv_mean'] = t_mean_to_use
        d['t_sv_sample'] = t_sv
        d['r_sv_mean'] = r_mean_to_use
        d['r_sv_sample'] = r_sv

        p = open(norm_param_pkl, "wb")
        pickle.dump(d, p, True)
        p.close()

        c.close()
        conn.close()

        return

    #=====================================
    #          UBM ADAPTATION 
    #=====================================
    def adapt_means(self, ubm_means, ubm_covars, ubm_weights, new_means, new_weights, T):
        n_i = new_weights*T
        alpha_i = n_i/(n_i+10)
        new_means[np.isnan(new_means)] = 0.0
        return_means = (alpha_i*new_means.T+(1-alpha_i)*ubm_means.T).T
        diag_covars = np.diagonal(ubm_covars, axis1=1, axis2=2)
        
        return_means = (np.sqrt(ubm_weights)*(1/np.sqrt(diag_covars.T))*return_means.T).T
        return return_means

    def adapt_model(self, feats, ubm_params, M):
        # train GMM on features
        D = feats.shape[1]
        updated_means = np.array(ubm_params['means'], dtype=np.float32)

        for it in range(1): # adaptation loop
            gmm = GMM(M, D, means=updated_means, covars=np.array(ubm_params['covars']), \
                  weights=np.array(ubm_params['weights']), cvtype='diag')
            gmm.train(feats, max_em_iters=1)
        
            new_means = gmm.components.means
            new_weights = gmm.components.weights
            T = feats.shape[0]
            updated_means = self.adapt_means(ubm_params['means'], \
                            ubm_params['covars'], ubm_params['weights'], \
                            new_means, new_weights, T).flatten('C')

        return updated_means
    
    #=====================================
    #         QUERY COMPUTATIONS 
    #=====================================
    def get_song_features_from_query(self, attribute, query):
        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()
        sql_query = "SELECT song_id \
                    title FROM songs1m WHERE " \
                    + attribute + " LIKE '" + query +"'"
        c.execute(sql_query)
        song_ids = c.fetchall()
        
        song_ids_q = str(song_ids).strip('[]').replace("u", "").\
                     replace(",)", "").replace("(", "").replace(")", "")

        sql_query = "SELECT timbre_shape_0, timbre_shape_1, timbre_feats, artist_name, \
                    title FROM songs1m WHERE song_id IN (" + song_ids_q + ")"
        c.execute(sql_query)
        timbre_result = c.fetchall()

        sql_query = "SELECT rhythm_shape_0, rhythm_shape_1, rhythm_feats \
                    FROM songs1m_rhythm WHERE song_id IN (" + song_ids_q + ")"
        c.execute(sql_query)
        rhythm_result = c.fetchall()

        c.close()
        
        print '*********************************************************'
        print 'Num timbre results: ', len(timbre_result)
        print 'Num rhythm results: ', len(rhythm_result)
        print '*********************************************************'
                 
        conn.close()

        artist_title_pairs = []
        for row in timbre_result:
            artist_title_pairs.append((row[3], row[4]))

        return timbre_result, rhythm_result, artist_title_pairs

    def get_query_svs_and_artist_title_pairs(self, attribute, query):
        p = open(norm_param_pkl, "rb")
        song_sv_dict = pickle.load(p)
        print song_sv_dict.keys()
        p.close()

        timbre_result, rhythm_result, artist_title_pairs = self.get_song_features_from_query(attribute, query)
        t_feature_list = []
        r_feature_list = []
    
        for row in timbre_result:
           feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
           t_feature_list.append(feats)

        timbre_features = np.array(np.concatenate(t_feature_list))

        for row in rhythm_result:
           feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
           feats = feats.T
           r_feature_list.append(feats)

        rhythm_features = np.array(np.concatenate(r_feature_list))

        print timbre_features.shape
        print rhythm_features.shape
        query_timbre_sv = self.adapt_model(timbre_features, self.timbre_ubm_params, M)
        query_rhythm_sv = self.adapt_model(rhythm_features, self.rhythm_ubm_params, M)

        query_timbre_sv = msdtools.mcs_norm(query_timbre_sv, song_sv_dict['t_sv_mean'])
        query_rhythm_sv = msdtools.mcs_norm(query_rhythm_sv, song_sv_dict['r_sv_mean'])

        p_mean_t, p_sigma_t = msdtools.p_norm_params_single(query_timbre_sv, song_sv_dict['t_sv_sample'].T)
        p_mean_r, p_sigma_r = msdtools.p_norm_params_single(query_rhythm_sv, song_sv_dict['r_sv_sample'].T)

        query_dict = {}
        query_dict['q_t_sv'] = query_timbre_sv
        query_dict['q_r_sv'] = query_rhythm_sv
        query_dict['p_mean_t'] = p_mean_t
        query_dict['p_mean_r'] = p_mean_r
        query_dict['p_sigma_t'] = p_sigma_t
        query_dict['p_sigma_r'] = p_sigma_r

        return query_dict, artist_title_pairs

    def get_collab_f_info(self, artist_title_pairs):
        query_song_nums = []
        for qp in artist_title_pairs:
            artist,title = qp
            collab.print_matching_queries(artist=artist,title=title)
            query_song_nums += collab.get_song_nums(artist=artist,title=title)
        
        output_song_ids, output_similarity = \
                collab.collaborative_filter(query_song_nums,weights=None,num_neighbors=20)
        
        return output_song_ids, output_similarity

    def get_close_songs_data(self, collab_f_song_ids):
        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()

        ids = str(collab_f_song_ids).replace("u", "").replace("[", "").replace("]", "")

        # get all song_ids
        # TODO: optimize this...
        sql_query = "SELECT title, artist_name, song_id \
                     FROM songs1m WHERE song_id IN (" + ids + ")"
        c.execute(sql_query)
        song_titles = c.fetchall()

        sql_query = "SELECT timbre_sv, rhythm_sv, \
                     p_mean_t, p_mean_r, p_sigma_t, p_sigma_r, song_id \
                     FROM songs1m_sv WHERE song_id IN (" + ids + ")"
        c.execute(sql_query)
        song_data = c.fetchall()
        c.close()
        conn.close()

        total_dict = {}
        for s in song_titles:
            song_id = s[2]
            total_dict[song_id] = {}
            total_dict[song_id]['title'] = s[0]
            total_dict[song_id]['artist_name'] = s[1]
            
        for s in song_data:
            song_id = s[6]
            total_dict[song_id]['t_sv'] = np.ndarray((SV_SIZE,),  buffer=s[0], dtype=np.float32)
            total_dict[song_id]['r_sv'] = np.ndarray((SV_SIZE,),  buffer=s[1], dtype=np.float32)
            total_dict[song_id]['p_mean_t'] = s[2]
            total_dict[song_id]['p_mean_r'] = s[3]
            total_dict[song_id]['p_sigma_t'] = s[4]
            total_dict[song_id]['p_sigma_r'] = s[5]


        return total_dict

    def get_final_list_of_closest_songs(self, qd, NN):
        
        song_ids = []
        title_artist = []
        t_supervectors = []
        t_p_means = []
        t_p_sigmas = []
        r_supervectors = []
        r_p_means = []
        r_p_sigmas = []
        for song in NN.keys():
            song_ids.append(song)        
            t_supervectors.append(NN[song]['t_sv'])
            r_supervectors.append(NN[song]['r_sv'])
            t_p_means.append(NN[song]['p_mean_t'])
            r_p_means.append(NN[song]['p_mean_r'])
            t_p_sigmas.append(NN[song]['p_sigma_t'])
            r_p_sigmas.append(NN[song]['p_sigma_r'])
            #song_ids = str(chunk).strip('[]').replace("u", "")
            title = NN[song]['title']#.encode('utf-8')
            artist = NN[song]['artist_name']#.encode('utf-8')
            title_artist.append((title, artist))
            
        all_t_sv = np.vstack((t_supervectors))
        all_t_p_means = np.array(np.hstack((t_p_means)), dtype=np.float32)
        all_t_p_sigmas = np.array(np.hstack((t_p_sigmas)), dtype=np.float32)
        all_r_sv = np.vstack((r_supervectors))
        all_r_p_means = np.array(np.hstack((r_p_means)), dtype=np.float32)
        all_r_p_sigmas = np.array(np.hstack((r_p_sigmas)), dtype=np.float32)
        st = time.time()
        
        timbre_dist = msdtools.p_norm_distance_single(qd['q_t_sv'], all_t_sv.T, qd['p_mean_t'], all_t_p_means, qd['p_sigma_t'], all_t_p_sigmas)
        rhythm_dist = msdtools.p_norm_distance_single(qd['q_r_sv'], all_r_sv.T, qd['p_mean_r'], all_r_p_means, qd['p_sigma_r'], all_r_p_sigmas)


        total_dist = 0.7*timbre_dist + 0.3*rhythm_dist
        #total_dist = timbre_dist
        sorted_indices = np.argsort(total_dist)
        sorted_distances = np.sort(total_dist)

        close_songs = [(index, title_artist[index]) for index in sorted_indices[:200]]
        return close_songs

    def get_closest_songs(self, attribute, query):
        query_dict, artist_title_pairs = p.get_query_svs_and_artist_title_pairs(attribute, query)

        #print artist_title_pairs
        artist_title_pairs = [("Elton John", "Reverie"), ("Elton John", "Sacrifice"), ("Elton John", "Angeline")]
        collab_f_song_ids, collab_song_similarity = self.get_collab_f_info(artist_title_pairs)

        print len(collab_f_song_ids)
        close_songs_dict = self.get_close_songs_data(collab_f_song_ids)

        print len(close_songs_dict.keys())

        final_list = self.get_final_list_of_closest_songs(query_dict, close_songs_dict)

        print final_list
        
    #=====================================
    #           UBM TRAINING 
    #=====================================
    def get_UBM_features(self):
        '''
        gets the features for timbre and rhythm ubm training 
        from pickle file 
        '''
        p = open(ubm_t_feats_pkl, "rb")
        ubm_timbre_features = np.array(pickle.load(p), dtype=np.float32)
        p.close()
        p = open(ubm_r_feats_pkl, "rb")
        ubm_rhythm_features = np.array(pickle.load(p), dtype=np.float32)
        p.close()

        return ubm_timbre_features, ubm_rhythm_features 

    def train_and_pickle_UBM(self):
        '''
        train the UBM on a subset of features
        for now, get features from pickle file
        '''
        ubm_timbre_features, ubm_rhythm_features  = self.get_UBM_features()

        # Train Timbre UBM
        D = ubm_timbre_features.shape[1]
        num_timbre_feats = ubm_timbre_features.shape[0]
        print "--- total number of timbre ubm features:\t", num_timbre_feats, " -----"
        timbre_ubm = GMM(M,D,cvtype='diag')

        train_st = time.time()
        timbre_ubm.train(ubm_timbre_features)
        train_total = time.time() - train_st
        print "--- timbre ubm training time:\t", train_total, " -----"

        # Train Rhythm UBM
        D = ubm_rhythm_features.shape[1]
        num_rhythm_feats = ubm_rhythm_features.shape[0]
        print "--- total number of rhythm ubm features:\t", num_rhythm_feats, " -----"
        rhythm_ubm = GMM(M,D,cvtype='diag')

        train_st = time.time()
        rhythm_ubm.train(ubm_rhythm_features, max_em_iters=5)
        train_total = time.time() - train_st
        print "--- rhythm ubm training time:\t", train_total, " -----"

        #make dict of ubm parameters
        timbre_ubm_params = {}
        timbre_ubm_params['means'] = timbre_ubm.components.means
        timbre_ubm_params['covars'] = timbre_ubm.components.covars
        timbre_ubm_params['weights'] = timbre_ubm.components.weights

        rhythm_ubm_params = {}
        rhythm_ubm_params['means'] = rhythm_ubm.components.means
        rhythm_ubm_params['covars'] = rhythm_ubm.components.covars
        rhythm_ubm_params['weights'] = rhythm_ubm.components.weights

        #pickle the parameters
        p = open(ubm_t_params_pkl, "wb")
        pickle.dump(timbre_ubm_params, p, True)
        p.close()

        p = open(ubm_r_params_pkl, "wb")
        pickle.dump(rhythm_ubm_params, p, True)
        p.close()

        return timbre_ubm_params, rhythm_ubm_params

    def get_UBM_parameters(self, from_pickle=False):
        if from_pickle:
            p = open(ubm_t_params_pkl, "rb")
            self.timbre_ubm_params = pickle.load(p)
            p.close()
            p = open(ubm_r_params_pkl, "rb")
            self.rhythm_ubm_params = pickle.load(p)
            p.close()
        else:
            self.timbre_ubm_params, self.rhythm_ubm_params = self.train_and_pickle_UBM(M)

        return self.timbre_ubm_params, self.rhythm_ubm_params
    
    def __init__(self):
        UBM_params = self.get_UBM_parameters(from_pickle=True)
        print "------------- DONE INITIALIZING ----------"


p = Pardora()

t = time.time()
#p.create_rhythm_table()
#p.compute_and_add_rhythm_feats()

#p.create_sv_table()
#p.compute_and_add_song_svs()

p.get_closest_songs("artist_name", "elton john")
#p.get_closest_songs("title", "windowpane")

print "----------------------------------------------------------------------------"
print "                           TOTAL TIME: ", time.time() - t
print "----------------------------------------------------------------------------"

