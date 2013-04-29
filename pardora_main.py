from gmm_specializer.gmm import *
import numpy as np
import MySQLdb as mdb
import pickle
import time
import binascii
import array
import sqlite3
import msdtools
import unicodedata
import collab

# Pardora-specific imports
import pardora_db
import pardora_ubm
import pardora_preprocessing

norm_param_pkl = "/disk1/home_user/egonina/msd_database/pickles/norm_param_pkl_1M.pkl"
l2_output_pkl = "/disk1/home_user/egonina/pardora_demo/pardora_2_level_output.pkl"
l3_output_pkl = "/disk1/home_user/egonina/pardora_demo/pardora_3_level_output.pkl"
l4_output_pkl = "/disk1/home_user/egonina/pardora_demo/pardora_4_level_output.pkl"

CF_NEIGHBORS = 100 
M = 64

class Pardora:
    #=================================================
    #        WRAPPERS TO PREPROCESSING FUNCTIONS  
    #=================================================
    def create_rhythm_table(self):
        pardora_db.create_rhythm_table()
        return

    def drop_rhythm_table(self):
        pardora_db.drop_rhythm_table()
        return

    def compute_and_add_rhythm_feats(self):
        pardora_preprocessing.compute_and_add_rhythm_feats()
        return

    def create_sv_table(self):
        pardora_db.create_sv_table()
        return

    def drop_sv_table(self):
        pardora_db.drop_sv_table()
        return

    def compute_and_add_song_svs(self, timbre_ubm_params, rhythm_ubm_params):
        pardora_preprocessing.compute_and_add_song_svs(timbre_ubm_params, rhythm_ubm_params)
        return

    
    #=====================================
    #         QUERY COMPUTATIONS 
    #=====================================

    def get_query_data(self, song_id_list):
        p = open(norm_param_pkl, "rb")
        song_sv_dict = pickle.load(p)
        p.close()

        st = time.time()
        timbre_result, rhythm_result = pardora_db.get_song_features_from_query(song_id_list)
        print "-------------------------"
        print "  TIME: SV, get features time:\t", time.time() - st
        print "-------------------------"

        t_feature_list = []
        r_feature_list = []
    
        st = time.time()
        for row in timbre_result:
           feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
           t_feature_list.append(feats)

        timbre_features = np.array(np.concatenate(t_feature_list))

        for row in rhythm_result:
           feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
           feats = feats.T
           r_feature_list.append(feats)

        rhythm_features = np.array(np.concatenate(r_feature_list))

        print "-------------------------"
        print "  TIME: SV, feature concat time:\t", time.time() - st
        print "-------------------------"

        print timbre_features.shape
        print rhythm_features.shape

        st = time.time()
        query_timbre_sv = pardora_ubm.adapt_model(timbre_features, self.timbre_ubm_params, M)
        query_rhythm_sv = pardora_ubm.adapt_model(rhythm_features, self.rhythm_ubm_params, M)

        query_timbre_sv = msdtools.mcs_norm(query_timbre_sv, song_sv_dict['t_sv_mean'])
        query_rhythm_sv = msdtools.mcs_norm(query_rhythm_sv, song_sv_dict['r_sv_mean'])

        p_mean_t, p_sigma_t = msdtools.p_norm_params_single(query_timbre_sv, song_sv_dict['t_sv_sample'].T)
        p_mean_r, p_sigma_r = msdtools.p_norm_params_single(query_rhythm_sv, song_sv_dict['r_sv_sample'].T)
        print "-------------------------"
        print "  TIME: SV, mcs and p-norm time:\t", time.time() - st
        print "-------------------------"

        query_dict = {}
        query_dict['q_t_sv'] = query_timbre_sv
        query_dict['q_r_sv'] = query_rhythm_sv
        query_dict['p_mean_t'] = p_mean_t
        query_dict['p_mean_r'] = p_mean_r
        query_dict['p_sigma_t'] = p_sigma_t
        query_dict['p_sigma_r'] = p_sigma_r

        return query_dict 

    def get_query_data_multi_query(self, song_id_list):
        p = open(norm_param_pkl, "rb")
        song_sv_dict = pickle.load(p)
        p.close()

        st = time.time()
        query_dicts = pardora_db.get_song_svs_multi_query(song_id_list)
        print "-------------------------"
        print "  TIME: SV, multiquery get feature time:\t", time.time() - st
        print "-------------------------"

        return query_dicts

    def get_collab_info(self, song_id_list):
        st = time.time()
        output_song_ids, output_similarity = \
                collab.filter_compound_query(song_id_list, num_neighbors = CF_NEIGHBORS)
        print "-------------------------"
        print "  TIME: CF Close Song Compute Time:\t", time.time() - st
        print "-------------------------"

        collab_song_data = {}
        if output_song_ids is not None:
            idx = 0
            for s in output_song_ids:
                collab_song_data[s] = output_similarity[idx]
                idx += 1

        return collab_song_data 

    def get_collab_info_multi_query(self, song_id_list):
        st = time.time()
        output_song_ids, output_similarity = \
                collab.filter_multiple_queries(song_id_list, num_neighbors = CF_NEIGHBORS)
        
        print "-------------------------"
        print "  TIME: CF Close Song Compute Time, Multiquery:\t", time.time() - st
        print "-------------------------"
        # collab_song_data[input_song_id] -> dictionary neighbor_song_id -> cf_score
        collab_song_data = {}
        for idx in range(len(song_id_list)):
            song_id = song_id_list[idx]
            cf_nn = output_song_ids[idx] #should be a list..
            cf_scores = output_similarity[idx] #should be a list..
            collab_song_data[song_id] = {}
            for nn_idx in range(len(cf_nn)):
                collab_song_data[song_id][cf_nn[nn_idx]] = cf_scores[nn_idx]

        return collab_song_data 


    def get_nn_dict(self, qd, NN, fanout):
        song_ids = []
        title_artist = []
        mta = []
        t_supervectors = []
        t_p_means = []
        t_p_sigmas = []
        r_supervectors = []
        r_p_means = []
        r_p_sigmas = []
        cf_distances = []
        for song in NN.keys():
            song_ids.append(song)        
            t_supervectors.append(NN[song]['t_sv'])
            r_supervectors.append(NN[song]['r_sv'])
            t_p_means.append(NN[song]['p_mean_t'])
            r_p_means.append(NN[song]['p_mean_r'])
            t_p_sigmas.append(NN[song]['p_sigma_t'])
            r_p_sigmas.append(NN[song]['p_sigma_r'])
            title = NN[song]['title']
            artist = NN[song]['artist_name']
            mode = NN[song]['mode']
            tempo = NN[song]['tempo']
            artist_hottness = NN[song]['artist_hottness']
            cf_distances.append(NN[song]['cf_score'])
            title_artist.append((title, artist))
            mta.append((mode, tempo, artist_hottness))
            
        all_t_sv = np.vstack((t_supervectors))
        all_t_p_means = np.array(np.hstack((t_p_means)), dtype=np.float32)
        all_t_p_sigmas = np.array(np.hstack((t_p_sigmas)), dtype=np.float32)
        all_r_sv = np.vstack((r_supervectors))
        all_r_p_means = np.array(np.hstack((r_p_means)), dtype=np.float32)
        all_r_p_sigmas = np.array(np.hstack((r_p_sigmas)), dtype=np.float32)
        
        timbre_dist = msdtools.p_norm_distance_single(qd['q_t_sv'], all_t_sv.T, qd['p_mean_t'], all_t_p_means, qd['p_sigma_t'], all_t_p_sigmas)
        rhythm_dist = msdtools.p_norm_distance_single(qd['q_r_sv'], all_r_sv.T, qd['p_mean_r'], all_r_p_means, qd['p_sigma_r'], all_r_p_sigmas)

        cf_dist = np.array(cf_distances, dtype=np.float32)

        total_dist = 0.7*timbre_dist + 0.3*rhythm_dist # + some_weight * cf_dist
        sorted_indices = np.argsort(total_dist)
        sorted_distances = np.sort(total_dist)

        close_songs = {} 
        count = 0
        for index in sorted_indices[:fanout]:
            song_id = song_ids[index]
            close_songs[song_id] = {}
            close_songs[song_id]['song_id'] = song_id 
            close_songs[song_id]['artist_name'] = title_artist[index][1]
            close_songs[song_id]['title'] = title_artist[index][0]
            close_songs[song_id]['dist_to_parent'] = sorted_distances[count] 
            close_songs[song_id]['mode'] = mta[index][0] 
            close_songs[song_id]['tempo'] = mta[index][1] 
            close_songs[song_id]['artist_hottness'] = mta[index][2] 
            count += 1

        return close_songs

    def get_nn_one_query(self, song_id_list, fanout):
        print "**************************************************"
        print "..... Getting query data and supervectors"
        print "**************************************************\n"
        st = time.time()
        query_dict = p.get_query_data(song_id_list)
        print "  ===== S1 TIME: Query SV & data:", time.time() - st, " ====="

        print "**************************************************"
        print "..... Getting collaborative filtering results"
        print "**************************************************\n"
        st = time.time()
        collab_song_info = self.get_collab_info(song_id_list)
        print "  ===== S2 TIME: Get collab filtering info: ", time.time() - st, " ====="

        if len(collab_song_info.keys()) > 0:
            print "**************************************************"
            print "..... Getting close song data to compute distances on"
            print "**************************************************\n"
            st = time.time()
            close_songs_dict = pardora_db.get_cf_songs_data(collab_song_info)
            print " ===== S3 TIME: Get cf songs data:", time.time() - st, " ====="

            print "**************************************************"
            print "..... Getting final list of close songs"
            print "**************************************************\n"
            st = time.time()
            nn_dict = self.get_nn_dict(query_dict, close_songs_dict, fanout)
            print " ===== S4 TIME: Get final list:", time.time() - st, " =====\n"
        else:
            print "No collaborative filtering neighbors found."
            nn_dict = None 

        return nn_dict 

    def get_nn_multi_query(self, song_id_list, fanout):
        print "**************************************************"
        print "..... Getting query data and supervectors"
        print "**************************************************\n"
        st = time.time()
        query_dicts = p.get_query_data_multi_query(song_id_list)
        print "  ===== S1 TIME: Query SV & data:", time.time() - st, " ====="

        print "**************************************************"
        print "..... Getting collaborative filtering results"
        print "**************************************************\n"
        st = time.time()
        collab_song_infos = self.get_collab_info_multi_query(song_id_list)
        print "  ===== S2 TIME: Get collab filtering info: ", time.time() - st, " ====="

        total_nn_dict = {}
        total_id_list = []
        for input_song in song_id_list:

            if len(collab_song_infos[input_song].keys()) > 0:
                print "**************************************************"
                print "..... Getting close song data to compute distances on"
                print "**************************************************\n"
                st = time.time()
                close_songs_dict = pardora_db.get_cf_songs_data(collab_song_infos[input_song])
                print " ===== S3 TIME: Get cf songs data:", time.time() - st, " ====="

                print "**************************************************"
                print "..... Getting final list of close songs"
                print "**************************************************\n"
                st = time.time()
                nn_dict = self.get_nn_dict(query_dicts[input_song], close_songs_dict, fanout)
                print " ===== S4 TIME: Get final list:", time.time() - st, " =====\n"

            else:
                print "No collaborative filtering neighbors found."
                nn_dict = None 

            total_nn_dict[input_song] = nn_dict
            for k in nn_dict.keys(): total_id_list.append(k)

        return total_nn_dict , total_id_list

    def get_near_neighbors(self, song_list, num_levels=1, fanout=20):
        print "**************************************************"
        print "QUERY: ", song_list 
        print "**************************************************\n"

        song_id_list = pardora_db.get_song_ids_from_title_artist_pairs(song_list)

        final_dict = {}
        final_dict[0] = {}
        final_dict[0]['artist_name'] = 'Root'

        # Make sure the query returned some results
        if song_id_list is not None:
            if num_levels == 1:
                nn = self.get_nn_one_query(song_id_list, fanout)
                final_dict[0]['nn'] = nn

            else:
                queue = {}
                queue[0] = []
                nn = self.get_nn_one_query(song_id_list, fanout)
                final_dict[0]['nn'] = nn

                for n in nn.keys():
                    queue[0].append(nn[n]) 

                id_list = nn.keys()
                for level in range(num_levels-1):
                    m_nn, id_list = self.get_nn_multi_query(id_list, fanout)

                    for elem in queue[level]:
                        elem['nn'] = m_nn[elem['song_id']] 

                    queue[level+1] = []
                    for m in m_nn.keys():
                        for k in m_nn[m].keys():
                            queue[level+1].append(m_nn[m][k]) 
        else:
            print "No songs matched the query: ", song_list
            sys.exit()

        print ":::::::::::::::::::::::::::::::::::::::::::::::::::"
        print "                FINAL TREE                         "

        print "ROOT: ", final_dict[0]['artist_name'] 

        print "Neighbors:" 
        for nn in final_dict[0]['nn'].keys():
            print "\t", final_dict[0]['nn'][nn]['artist_name'], final_dict[0]['nn'][nn]['title'] 
            print "\tNeighbors:" 
            childs = final_dict[0]['nn'][nn]['nn']
            for c in childs.keys():
                print "\t\t", childs[c]['artist_name'], childs[c]['title']
                print "\t\tNeighbors:" 
                childs2 = childs[c]['nn']
                for c2 in childs2.keys():
                    print "\t\t\t", childs2[c2]['artist_name'], childs2[c2]['title']


        print ":::::::::::::::::::::::::::::::::::::::::::::::::::"
    
    def __init__(self):
        self.timbre_ubm_params, self.rhythm_ubm_params = pardora_ubm.get_UBM_parameters(M, from_pickle=True)
        print "------------- DONE INITIALIZING ----------"

p = Pardora()

t = time.time()
#p.create_rhythm_table()
#p.compute_and_add_rhythm_feats()

#p.create_sv_table()
#p.compute_and_add_song_svs()

song_list = []
song_list.append(("radiohead", "karma police"))
song_list.append(("elton john", "angeline"))
#song_list.append(("radiohead", "paranoid android"))
#song_list.append(("elton john", "candle in the wind"))

nn = p.get_near_neighbors(song_list, 3, 5)


print "----------------------------------------------------------------------------"
print "                           TOTAL TIME: ", time.time() - t
print "----------------------------------------------------------------------------"
