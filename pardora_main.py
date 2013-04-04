from gmm_specializer.gmm import *
import MySQLdb as mdb
import pickle
import time
import binascii
import array
import sqlite3
import msdtools
import unicodedata

from whoosh.query import *
 
ubm_t_feats_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_features_1M_008.pkl"
ubm_t_params_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_params.pkl"
song_id_pkl = "/disk1/home_user/egonina/msd_database/pickles/song_ids_1M.pkl"
norm_param_pkl = "/disk1/home_user/egonina/msd_database/pickles/norm_param_pkl_1M.pkl"

conn_str = '169.229.49.36', 'dbuser', 'p41msongs', 'milsongs'
CHUNK_SIZE = 1000

class Pardora:

    #=====================================
    #          SV DB MANIPULATION 
    #=====================================
    def drop_sv_table(self):
        """
        Creates the file and an empty table.
        """
        # creates file
        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        # add stuff
        c = conn.cursor()
        q = 'DROP TABLE songs1m_sv'
        c.execute(q)
        # commit and close
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
        q += 'p_mean_t REAL, '
        q += 'p_sigma_t REAL, '
        q += 'PRIMARY KEY (song_id)) ENGINE=NDBCLUSTER DEFAULT CHARSET=utf8;'
        c.execute(q)
        # commit and close
        conn.commit()
        c.close()
        conn.close()

    def add_sv_and_p_vals_to_db(self, conn, c, song_id, sv, p_mean, p_sigma):
        # build query
        q = "INSERT INTO songs1m_sv VALUES (%s, %s, %s, %s, %s);"

        sv_bin = sqlite3.Binary(sv)
        sv0 = int(sv.shape[0])
        
        insert_values = (song_id, sv_bin, sv0, p_mean, p_sigma)
        
        c.execute(q, insert_values)

    def chunks(self, l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]

    def compute_and_add_song_svs(self):
        p = open(song_id_pkl, "rb")    
        song_ids = pickle.load(p)
        p.close()

        song_id_chunks = self.chunks(song_ids, CHUNK_SIZE)
        chunk_count = 0

        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()

        t_mean_to_use = np.zeros(1)
        t_sv = np.zeros(1)

        total_time = time.time()
        for chunk in song_id_chunks:
            if chunk_count < 1:
                timbre_sv_arr = []
                print "==="
                print "CHUNK: ", chunk_count
                chunk_count+=1

                song_ids = str(chunk).strip('[]').replace("u", "").replace(",)", "").replace("(", "")
                
                st = time.time()
                sql_query = "SELECT timbre_feats, timbre_shape_0, timbre_shape_1, song_id FROM songs1m \
                            WHERE song_id IN ("+song_ids+")"
                c.execute(sql_query)
                songs = c.fetchall()

                print "INFO: time to query for song id chunk: ", time.time() - st
                
                st = time.time()
                for s in songs:
                    feats =  np.array(np.ndarray((s[1],s[2]), buffer=s[0]), dtype=np.float32)
                    timbre_sv = self.adapt_model(feats, self.timbre_ubm_params, M)
                    timbre_sv_arr.append(timbre_sv)

                print "INFO: SV comp time: ", time.time() - st
                
                st = time.time()
                t_sv = np.vstack(timbre_sv_arr)
                del timbre_sv_arr

                if chunk_count == 1:
                    t_mean_to_use = np.mean(t_sv, axis=0)

                t_sv = msdtools.mcs_norm(t_sv.T, t_mean_to_use).T
                
                print "INFO: MCS norm computation time: ", time.time() - st

                st = time.time()
                p_means_t = np.zeros(len(songs))
                p_sigmas_t = p_means_t.copy()

                st = time.time()

                p_means_t, p_sigmas_t = msdtools.p_norm_params_chunk(t_sv.T, t_sv.T, CHUNK_SIZE)

                print "INFO: P-means computation time: ", time.time() - st

                st = time.time()
                print ".... adding vectors to the database ...."
                for idx in range(len(songs)):
                    if idx % 10 == 0:
                        print idx
                    t = np.array(t_sv[idx])
                    self.add_sv_and_p_vals_to_db(conn, c, song_ids[idx], t, p_means_t[idx], p_sigmas_t[idx])

                conn.commit()

                print "INFO: Database update time: ", time.time() - st

        print "=============================================="
        print "INFO: TOTAL TIME FOR SV COMP TIME: ", time.time() - total_time
        d = {}   
        d['t_sv_mean'] = t_mean_to_use
        d['t_sv_sample'] = t_sv

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
        sql_query = "SELECT timbre_shape_0, timbre_shape_1, timbre_feats, artist_name, \
                    title FROM songs1m WHERE " \
                    + attribute + " LIKE '" + query +"'"
        c.execute(sql_query)
        result = c.fetchall()
        c.close()
        
        print '*********************************************************'
        print 'Num results: ', len(result)
        #for row in result:
        #  print row[1], row[2]
        print '*********************************************************'
                 
        conn.close()

        return result

    def get_query_supervector(self, M, attribute, query):
        query_result = self.get_song_features_from_query(attribute, query)
        feature_list = []
    
        for row in query_result:
           feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
           feature_list.append(feats)

        timbre_features = np.array(np.concatenate(feature_list))

        print timbre_features.shape
        query_timbre_sv = self.adapt_model(timbre_features, self.timbre_ubm_params, M)

        return query_timbre_sv

    #=====================================
    #           UBM TRAINING 
    #=====================================
    def get_UBM_features(self):
        '''
        gets the features for timbre ubm training 
        from pickle file 
        '''
        p = open(ubm_t_feats_pkl, "rb")
        ubm_timbre_features = np.array(pickle.load(p), dtype=np.float32)
        p.close()

        return ubm_timbre_features 

    def train_and_pickle_UBM(self, M):
        '''
        train the UBM on a subset of features
        for now, get features from pickle file
        '''
        ubm_timbre_features = self.get_UBM_features()

        D = ubm_timbre_features.shape[1]
        num_feats = ubm_timbre_features.shape[0]
        print "--- total number of ubm features:\t", num_feats, " -----"
        timbre_ubm = GMM(M,D,cvtype='diag')

        train_st = time.time()
        timbre_ubm.train(ubm_timbre_features)
        train_total = time.time() - train_st
        print "--- timbre ubm training time:\t", train_total, " -----"

        #make dict of ubm parameters
        timbre_ubm_params = {}
        timbre_ubm_params['means'] = timbre_ubm.components.means
        timbre_ubm_params['covars'] = timbre_ubm.components.covars
        timbre_ubm_params['weights'] = timbre_ubm.components.weights

        #pickle the parameters
        p = open(ubm_t_params_pkl, "wb")
        pickle.dump(timbre_ubm_params, p, True)
        p.close()

    def get_UBM_parameters(self, M, from_pickle=False):
        if from_pickle:
            p = open(ubm_t_params_pkl, "rb")
            self.timbre_ubm_params = pickle.load(p)
            p.close()
        else:
            self.timbre_ubm_params = self.train_and_pickle_UBM(M)

        return self.timbre_ubm_params
    
    def __init__(self, M):
        UBM_params = self.get_UBM_parameters(M, from_pickle=True)
        print "------------- DONE INITIALIZING ----------"


M = 64
p = Pardora(M)
p.drop_sv_table()
p.create_sv_table()
p.compute_and_add_song_svs()



#sv = p.get_query_supervector(M, "artist_name", "elton_john")
