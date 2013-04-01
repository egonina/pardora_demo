from gmm_specializer.gmm import *
import MySQLdb as mdb
import pickle
import time
import binascii
import array
 
ubm_t_feats_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_features_1M_008.pkl"
ubm_t_params_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_params.pkl"


class Pardora:

    def get_song_features_from_query(self, attribute, query):
        conn_str = '169.229.49.36', 'dbuser', 'p41msongs', 'milsongs'
         
        conn = mdb.connect(conn_str[0], conn_str[1], conn_str[2], conn_str[3])
        c = conn.cursor()
        sql_query = "SELECT timbre_shape_0, timbre_shape_1, timbre_feats, artist_name, title FROM songs1m WHERE " \
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
            gmm = GMM(M, D, means=updated_means, covars=np.array(ubm_params['covars']), weights=np.array(ubm_params['weights']), cvtype='diag')
            gmm.train(feats, max_em_iters=1)
        
            new_means = gmm.components.means
            new_weights = gmm.components.weights
            T = feats.shape[0]
            updated_means = self.adapt_means(ubm_params['means'], ubm_params['covars'], ubm_params['weights'], new_means, new_weights, T).flatten('C')
            #updated_means = self.adapt_means(updated_means.reshape((64, 12)), ubm_params['covars'], ubm_params['weights'], new_means, new_weights, T).flatten('C')

        return updated_means

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

M = 64
p = Pardora(M)
sv = p.get_query_supervector(M, "artist_name", "elton_john")
