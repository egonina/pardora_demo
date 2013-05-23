import operator
import sqlite3 as sql
import numpy as np
import cPickle as pickle
import pylab as pl
import pdb
import cProfile
from scipy import sparse
import time

'''
This module implements Million Song Dataset collaborative filtering as described in the paper:
[1] F. Aiolli, "A Preliminary Study on a Recommender System for the Million Songs Dataset Challenge," PREFERENCE LEARNING: PROBLEMS AND APPLICATIONS IN AI, p. 1, 2012.
'''

np.set_printoptions(precision=2)

DATA_PATH = '/disk1/home_user/egonina/msd_database/cf_data/'
#ALPHA = 0.15 # Aiolli
ALPHA = 0.5 # Aiolli


try: init_stuff
except:
    print 'collab.py: Loading dictionaries/matrices'

    # read user/song lookup dictionaries into memory
    with open(DATA_PATH + 'user_song_lookup.pkl', 'rb') as fp:
        lookup_table = pickle.load(fp)

    #with open(DATA_PATH + 'play_count_matrix.pkl', 'r') as fp:
    #    play_count_matrix,col_norms_lin = pickle.load(fp)

    with open(DATA_PATH + 'play_count_log.pkl', 'rb') as fp:
        play_count_log, col_norms_log = pickle.load(fp)

    #with open(DATA_PATH + 'collab_p_norm_stats.pkl', 'rb') as fp:
    #    pnorm_dict = pickle.load(fp)
    #    pnorm_num_nonzero = pnorm_dict['num_nonzero']
    #    pnorm_means = pnorm_dict['means']
    #    pnorm_stddevs = pnorm_dict['stddevs']

    with open(DATA_PATH + 'artist_title_lookup.pkl', 'rb') as fp:
        artist_title_lookup = pickle.load(fp)

    init_stuff = True

def construct_artist_title_lookup():
    '''
    dictionary to lookup artist and title by song id
    requires the presence of the tracks.db database
    '''
    with sql.connect(DATA_PATH + 'tracks.db') as con:
        cur = con.cursor()

        cur.execute("SELECT * FROM Tracks")
        rows = cur.fetchall()

    artist_title_lookup = {}
    for song_id,artist,title in rows:
        artist_title_lookup[song_id] = (artist, title)

    with open(DATA_PATH + 'artist_title_lookup.pkl','wb') as fp:
        pickle.dump(artist_title_lookup,fp,protocol=-1)



def construct_testing_user_song_lookup():
    '''
    Build a dictionary for user/song lookups on the kaggle test set.

    Dictionary is saved to DATA_PATH as 'test_user_song_lookup.pkl'.

    Dictionary structure:
    lookup['user']['num'][num] = user_id
    lookup['user']['id'][user_id] = user_num
    lookup['song']['num'][num] = song_id
    lookup['song']['id'][song_id] = song_num
    '''

    with open(DATA_PATH + 'kaggle_users.txt','r') as fp:
        user_num_lookup = dict()
        user_id_lookup = dict()

        ind = 0
        for line in fp:
            user_id = line.strip()
            user_num_lookup[user_id] = ind
            user_id_lookup[ind] = user_id
            print ind, user_id
            ind += 1

        user_lookup = {'num':user_num_lookup,'id':user_id_lookup}

    with open(DATA_PATH + 'kaggle_songs.txt','r') as fp:
        song_num_lookup = dict()
        song_id_lookup = dict()

        for line in fp:
            song_id, song_num = line.strip().split(' ')
            song_num = int(song_num) - 1
            song_num_lookup[song_id] = song_num
            song_id_lookup[song_num] = song_id
            print song_num, song_id
            ind += 1

        song_lookup = {'num':song_num_lookup,'id':song_id_lookup}

    with open(DATA_PATH + 'testing_user_song_lookup.pkl', 'wb') as fp:
        lookup = {'user':user_lookup,'song':song_lookup}
        pickle.dump(lookup, fp, protocol=-1)

def construct_user_song_lookup():
    '''
    Build a dictionary for user/song lookups on the entire MSD + eval half histories

    Dictionary is saved to DATA_PATH as 'user_song_lookup.pkl'.

    Dictionary structure:
    lookup['user']['num'][num] = user_id
    lookup['user']['id'][user_id] = user_num
    lookup['song']['num'][num] = song_id
    lookup['song']['id'][song_id] = song_num
    '''



    user_num_lookup = dict()
    song_num_lookup = dict()

    song_num = 0
    user_num = 0

    for filename in ['kaggle_visible_evaluation_triplets.txt', 'train_triplets.txt']:
        with open(DATA_PATH + filename,'r') as fp:

            for line in fp:
                user_id, song_id, count = line.strip().split('\t')
                if not user_num_lookup.has_key(user_id):
                    user_num_lookup[user_id] = user_num
                    if user_num % 1000 == 0:
                        print 'user', user_num, user_id
                    user_num += 1
                if not song_num_lookup.has_key(song_id):
                    song_num_lookup[song_id] = song_num
                    if song_num % 1000 == 0:
                        print '\tsong', song_num, song_id
                    song_num += 1


    user_id_lookup = dict((v,k) for k,v in user_num_lookup.iteritems())
    song_id_lookup = dict((v,k) for k,v in song_num_lookup.iteritems())

    lookup = {  'user': {'num':user_num_lookup,'id':user_id_lookup},
                'song': {'num':song_num_lookup,'id':song_id_lookup}}

    with open(DATA_PATH + 'user_song_lookup.pkl', 'wb') as fp:
        pickle.dump(lookup, fp, protocol=-1)

def create_play_count_matrix():
    '''
    Create sparse playcount matrix from MSD data.

    Saved to DATA_PATH as 'play_count_matrix.pkl'.
    '''

    num_users = max(lookup_table['user']['num'].values())+1
    num_songs = max(lookup_table['song']['num'].values())+1

    play_count_matrix = sparse.lil_matrix((num_users,num_songs),dtype='uint16')

    #with open(DATA_PATH + 'kaggle_visible_evaluation_triplets.txt','r') as fp:
    for filename in ['kaggle_visible_evaluation_triplets.txt', 'train_triplets.txt']:
        with open(DATA_PATH + filename,'r') as fp:

            ind = 0
            for line in fp:
                user_id, song_id, count = line.strip().split('\t')

                user_num = lookup_table['user']['num'][user_id]
                song_num = lookup_table['song']['num'][song_id]

                play_count_matrix[user_num,song_num] = int(count)

                ind += 1
                if ind % 100000 == 0:
                    print ind

    play_count_matrix = sparse.csc_matrix(play_count_matrix)

    col_norms = play_count_matrix.copy()
    col_norms.data *= col_norms.data
    col_norms = np.array(col_norms.sum(axis=0)).flatten()

    with open(DATA_PATH + 'play_count_matrix.pkl','wb') as fp:
        pickle.dump([play_count_matrix,col_norms],fp,protocol=-1)

def create_log_play_count_matrix(play_count_matrix):
    ''' 
    Create log-scaled play count matrix from sparse play count matrix.

    Saved to DATA_PATH as 'play_count_log.pkl'.
    '''

    play_count_log = play_count_matrix.copy()
    play_count_log.data = np.log(play_count_log.data+1)

    col_norms = play_count_log.copy()
    col_norms.data *= col_norms.data
    col_norms = np.array(col_norms.sum(axis=0)).flatten()


    with open(DATA_PATH + 'play_count_log.pkl', 'wb') as fp:
        pickle.dump([play_count_log, col_norms],fp,protocol=-1)

def get_song_ids(artist=None,title=None):
    '''
    Return list of song ids that match artist/title query.
    '''

    with sql.connect(DATA_PATH + 'tracks.db') as con:
        cur = con.cursor()

        if title is None and artist is not None:
            cur.execute("SELECT Id FROM Tracks WHERE Artist = ?", (artist,))
        elif title is not None and artist is None:
            cur.execute("SELECT Id FROM Tracks WHERE Title = ?", (title,))
        elif title is not None and artist is not None:
            cur.execute("SELECT Id FROM Tracks WHERE Artist = ? AND Title = ?", (artist,title))

        rows = cur.fetchall()
        song_ids = [song_id for song_id, in rows]

    song_ids = list(set(song_ids)) # remove duplicates

    return song_ids

def get_song_nums(artist=None,title=None):
    '''
    Return list of song numbers that match artist/title query.

    Only songs with positive play count are returned.
    '''

    with sql.connect(DATA_PATH + 'tracks.db') as con:
        cur = con.cursor()

        if title is None and artist is not None:
            cur.execute("SELECT Id FROM Tracks WHERE Artist = ?", (artist,))
        elif title is not None and artist is None:
            cur.execute("SELECT Id FROM Tracks WHERE Title = ?", (title,))
        elif title is not None and artist is not None:
            cur.execute("SELECT Id FROM Tracks WHERE Artist = ? AND Title = ?", (artist,title))

        rows = cur.fetchall()
        song_nums = []
        for song_id, in rows:
            song_num = lookup_table['song']['num'].get(str(song_id),None)
            if song_num is not None:
                song_nums.append(song_num)

    song_nums = list(set(song_nums)) # remove duplicates

    return song_nums

def print_matching_queries(artist=None,title=None):
    '''
    print info about matching queries, sorted by number of plays
    '''

    print 'Query:', artist, '-', title

    #query_song_num = lookup_table['song']['num']['SOBRFHG12A81C210FB']
    with sql.connect(DATA_PATH + 'tracks.db') as con:
        cur = con.cursor()

        if title is None and artist is not None:
            cur.execute("SELECT * FROM Tracks WHERE Artist = ?", (artist,))
        elif title is not None and artist is None:
            cur.execute("SELECT * FROM Tracks WHERE Title = ?", (title,))
        elif title is not None and artist is not None:
            cur.execute("SELECT * FROM Tracks WHERE Artist = ? AND Title = ?", (artist,title))

        rows = cur.fetchall()
        songs_by_count = {}
        song_nums = []
        for song_id,artist,title in rows:
            song_num = lookup_table['song']['num'].get(str(song_id),None)
            if song_num is not None:
                col = play_count_log.getcol(song_num)
                count = col.sum()
                user_count = np.sum(col.data > 0)
                meta_data = '(%5u) %s %8u %s - %s' % (user_count,song_id,song_num,artist,title)
            else:
                count = 0
                meta_data = '(%5u) %s %8s %s - %s' % (0,song_id,'-',artist,title)
            songs_by_count[meta_data] = count


                #print song_id, song_num ,artist,title, count
        sorted_by_count = sorted(songs_by_count.iteritems(),key=operator.itemgetter(1))
        sorted_by_count.reverse()
        print '%7s:   (%5s) %18s %8s %s - %s' % ('Plays','Users', 'SongID', 'SongNum', 'Artist', 'Title')
        for c in sorted_by_count:
            print '%7u:   %s' % (c[1], c[0])

def collaborative_filter(query_song_nums,weights=None,num_neighbors=100):
    '''
    Return collaboratively filtered near neighbors given input song ids
    and optional song weights

    Input:
    song_ids - list/array of input song numbers (converted song ids)
    weights - list/array of song-wise weights [optional]
    '''



    # sort song nums and weights
    query_song_nums = np.array(query_song_nums)
    sorted_inds = np.argsort(query_song_nums)
    query_song_nums = query_song_nums[sorted_inds]
    if weights is not None:
        weights = np.array(weights)
        weights = weights[sorted_inds]

    print '\nQuery songs:', query_song_nums
    print 'Weights:', weights

    
    # alpha = 0.5 -> cosine similarity
    # alpha = 0.0 -> conditional probability
    alpha = ALPHA

    # similarity locality
    q = 3

    #history_matrix = play_count_matrix
    #col_norms = col_norms_lin
    history_matrix = play_count_log
    col_norms = col_norms_log

    query_norms = col_norms[query_song_nums]
    query_cols = [history_matrix.getcol(num) for num in query_song_nums]
    query_cols = sparse.hstack(query_cols,format='csc')

    cross_terms = history_matrix.T.dot(query_cols)

    query_scale = 1./ query_norms[None,:]**(1-alpha)
    col_scale = 1./ col_norms[:,None]**alpha
    scale = query_scale * col_scale

    similarity = sparse.csc_matrix(cross_terms.multiply(scale))
    similarity.data **= q

    if weights is not None:
        score = np.asarray(similarity.dot(weights)).flatten()
    else:
        score = np.asarray(similarity.sum(axis=1)).flatten() ** (1./q)

    nonzero_inds = np.nonzero(score)[0]

    #mean_score = np.mean(score)
    #std_score = np.std(score)

    sorted_nonzero_inds = np.argsort(score[nonzero_inds])[::-1]
    #song_nums = nonzero_inds[sorted_nonzero_inds][:num_neighbors]
    song_nums = nonzero_inds[sorted_nonzero_inds]

    #output_song_nums = []
    output_song_ids = []
    output_similarity = []

    num_returned = 0

    print '\nNear Neighbors'
    with sql.connect(DATA_PATH + 'tracks.db') as con:
        cur = con.cursor()

        #query_artists = zip(*query_pairs)[0]

        for i in xrange(len(song_nums)):

            song_num = song_nums[i]
            sim = score[nonzero_inds[sorted_nonzero_inds[i]]]
            #sim  = (sim - mean_score)/std_score


            song_id = lookup_table['song']['id'][song_num]

            cur.execute("SELECT * FROM Tracks WHERE Id = ?", (song_id,))
            #cur.execute("SELECT * FROM Tracks WHERE Artist = ?", ('Opeth',))
            rows = cur.fetchall()
            for song_id,artist,title in rows:
                if song_num not in query_song_nums:
                    print '%10.4f' % sim,
                    print song_id, ': ', '%7u' % song_num, ',', artist, ' - ', title
                    num_returned += 1
                    #output_song_nums.append(song_num)
                    output_song_ids.append(song_id)
                    output_similarity.append(sim)

            if num_returned >= num_neighbors:
                break

    return output_song_ids, output_similarity

def compute_similarity(query_song_nums,alpha=ALPHA):
    '''
    compute song-wise similarities for each query song

    Input:
    query_song_nums - list/array of input song numbers (converted song ids)
    alpha - similarity parameter, 0.5 -> cosine similarity, 0.0 -> conditional probability
    '''



    query_song_nums = np.array(query_song_nums)

    print '\nQuery song numbers:', query_song_nums
    
    #history_matrix = play_count_matrix
    #col_norms = col_norms_lin
    history_matrix = play_count_log
    col_norms = col_norms_log

    query_norms = col_norms[query_song_nums]
    query_cols = [history_matrix.getcol(num) for num in query_song_nums]
    query_cols = sparse.hstack(query_cols,format='csc')

    cross_terms = history_matrix.T.dot(query_cols)
    #cross_terms = query_cols.T.dot(history_matrix).T

    query_scale = 1./ query_norms[None,:]**(1-alpha)
    col_scale = 1./ col_norms[:,None]**alpha
    scale = query_scale * col_scale

    cross_terms.data *= scale[cross_terms.nonzero()]
    similarity = cross_terms.tocsc()

    return similarity

def compute_similarity_range(start,stop,alpha=ALPHA):
    '''
    compute song-wise similarities for each query song

    Input:
    query_song_nums - list/array of input song numbers (converted song ids)
    alpha - similarity parameter, 0.5 -> cosine similarity, 0.0 -> conditional probability
    '''



    print '\nQuery song numbers:', start, 'to', stop
    
    #history_matrix = play_count_matrix
    #col_norms = col_norms_lin
    history_matrix = play_count_log
    col_norms = col_norms_log

    query_norms = col_norms[start:stop]
    #query_cols = [history_matrix.getcol(num) for num in query_song_nums]
    #query_cols = sparse.hstack(query_cols,format='csc')
    query_cols = history_matrix[:,start:stop]

    cross_terms = history_matrix.T.dot(query_cols)
    #cross_terms = query_cols.T.dot(history_matrix).T

    query_scale = 1./ query_norms[None,:]**(1-alpha)
    col_scale = 1./ col_norms[:,None]**alpha
    scale = query_scale * col_scale

    cross_terms.data *= scale[cross_terms.nonzero()]
    similarity = cross_terms.tocsc()


    return similarity

def compute_compound_similarity(query_song_nums,alpha=ALPHA):
    '''
    compute song-wise similarities for each query song

    Input:
    query_song_nums - list/array of input song numbers (converted song ids)
    alpha - similarity parameter, 0.5 -> cosine similarity, 0.0 -> conditional probability
    '''



    query_song_nums = np.array(query_song_nums)

    print '\nQuery song numbers:', query_song_nums
    
    #history_matrix = play_count_matrix
    #col_norms = col_norms_lin
    history_matrix = play_count_log
    col_norms = col_norms_log

    #query_norms = col_norms[query_song_nums]
    query_cols = [history_matrix.getcol(num) for num in query_song_nums]
    query_cols = sparse.hstack(query_cols,format='csc')

    query_col = sparse.csc_matrix(query_cols.sum(axis=1))
    query_norm = np.sum(query_col.data ** 2)

    cross_terms = history_matrix.T.dot(query_col)

    query_scale = 1./ query_norm**(1-alpha)
    col_scale = 1./ col_norms[:,None]**alpha
    scale = query_scale * col_scale

    similarity = sparse.csc_matrix(cross_terms.multiply(scale))

    return similarity

def filter_compound_query(query_song_ids,num_neighbors=100):
    '''
    Return a single list of collaboratively filtered near neighbors and p-normed distance scores
    given a list of song_ids

    Input:
    query_song_ids - list/array of input song ids
    num_neighbors - maximum number of near neighbors to return

    Output:
    output_song_ids - array of output song ids
    output_distances - array of output distances
    '''

    # convert song_ids to song_nums
    query_song_nums = np.array([lookup_table['song']['num'].get(str(song_id),-1) for song_id in query_song_ids])

    # remove songs with zero play count
    zero_play_inds = np.where(query_song_nums == -1)[0]
    query_song_nums = query_song_nums[query_song_nums != -1]

    if len(query_song_nums) == 0:
        return None, None

    alpha = ALPHA
    t = time.time()
    similarity = compute_similarity(query_song_nums,alpha)
    similarity = np.asarray(similarity.mean(axis=1)).flatten()
    print 'sim:', time.time() - t
    #similarity = compute_compound_similarity(query_song_nums,alpha)
    #similarity = similarity.toarray().flatten()



    nonzero_inds = similarity.nonzero()[0]
    sim = similarity[nonzero_inds]
    sorted_nonzero_inds = np.argsort(sim)[::-1]
    sorted_nonzero_inds = sorted_nonzero_inds[:num_neighbors]
    sorted_song_nums = nonzero_inds[sorted_nonzero_inds]
    sorted_song_ids = np.array([lookup_table['song']['id'][song_num] for song_num in sorted_song_nums])
    sorted_similarities= sim[sorted_nonzero_inds]

    #print_query_results(query_song_ids,sorted_song_ids,sorted_similarities)

    return sorted_song_ids, sorted_similarities

def filter_multiple_queries(query_song_ids,num_neighbors=100):
    '''
    Return multiple lists of collaboratively filtered near neighbors and p-normed distance scores
    given a list of song_ids to be used as separate queries

    Input:
    query_song_ids - list/array of input song ids
    num_neighbors - maximum number of near neighbors to return

    Output:
    output_song_ids - list of arrays of output song ids
    output_distances - list of arrays of output distances

    Note: returns empty array in the list location corresponding to query songs without any plays
    '''

    # convert song_ids to song_nums
    
    query_song_nums = np.array([lookup_table['song']['num'].get(str(song_id),-1) for song_id in query_song_ids])


    # remove songs with zero play count, but remember their indices
    nonzero_play_locs = query_song_nums != -1
    query_song_nums = query_song_nums[nonzero_play_locs]


    alpha = ALPHA
    similarity = compute_similarity(query_song_nums,alpha)



    sorted_song_ids = []
    sorted_similarities = []
    col_ind = 0
    for query_ind in xrange(len(query_song_ids)):

        if nonzero_play_locs[query_ind]:


            col = similarity.getcol(col_ind)
            nonzero_inds = col.nonzero()[0]
            sim = col.data.copy()

            sorted_nonzero_inds = np.argsort(sim)[::-1]
            sorted_nonzero_inds = sorted_nonzero_inds[:num_neighbors]
            sorted_song_nums = nonzero_inds[sorted_nonzero_inds]
            song_ids = np.array([lookup_table['song']['id'][song_num] for song_num in sorted_song_nums])
            sorted_song_ids.append(song_ids)
            sorted_similarities.append(sim[sorted_nonzero_inds])
            col_ind += 1
        else:
            sorted_song_ids.append(np.array([]))
            sorted_similarities.append(np.array([]))


        #print_query_results([query_song_ids[query_ind]],sorted_song_ids[query_ind],sorted_similarities[query_ind])

    return sorted_song_ids, sorted_similarities

def compute_p_norm_stats():
    '''
    compute stats used in computing a normalized CF distance


    outputs:
    a pickle file containing:
    number of nonzero similarity elements for each song,
    mean of -np.log(sim) of nonzero similarity elements for each song,
    standard deviation of -np.log(sim) of nonzero similarity elements for each song.

    Output is pickled to disk
    '''

    alpha = ALPHA

    num_songs = play_count_log.shape[1]
    chunk_size = 1000 # songs
    num_chunks = num_songs/chunk_size
    partial_chunk = slice(num_chunks*chunk_size,num_songs)

    num_nonzero = -np.ones(num_songs,dtype='int32')
    means = -np.ones(num_songs,dtype='float32')
    stddevs = -np.ones(num_songs,dtype='float32')
    sorted_song_nums = []
    sorted_distances = []

    t = time.time()


    for c in xrange(num_chunks):
        print 'elapsed time:', time.time() - t
        print 'chunk:', c+1, 'of', num_chunks
        sl = slice(c*chunk_size,(c+1)*chunk_size)
        similarity = compute_similarity_range(sl.start,sl.stop)

        nonzero_els = similarity.copy()
        nonzero_els.data[:] = 1
        num_nonzero[sl] = np.asarray(nonzero_els.sum(axis=0)).flatten()

        distance = similarity.copy()
        distance.data[:] = -np.log(distance.data)

        sum_nonzero = np.asarray(distance.sum(axis=0)).flatten()
        means[sl] = sum_nonzero / num_nonzero[sl]

        if np.any(means[sl] < 0):
            pdb.set_trace()


        distance2 = distance.copy()
        distance2.data[:] **= 2
        sum2_nonzero = np.asarray(distance2.sum(axis=0)).flatten()
        stddevs[sl] = np.sqrt(sum2_nonzero/num_nonzero[sl] - means[sl]**2)

        print 'num nonzero: [%u,%u]' % (num_nonzero[sl].min(), num_nonzero[sl].max())
        print 'means: [%.2f,%.2f]' % (means[sl].min(), means[sl].max())
        print 'std: [%.2f,%.2f]' % (stddevs[sl].min(), stddevs[sl].max())
        print ''


    if partial_chunk.stop > partial_chunk.start:
        sl = partial_chunk

        similarity = compute_similarity_range(sl.start,sl.stop)

        nonzero_els = similarity.copy()
        nonzero_els.data[:] = 1
        num_nonzero[sl] = np.asarray(nonzero_els.sum(axis=0)).flatten()

        distance = similarity.copy()
        distance.data[:] = -np.log(distance.data)

        sum_nonzero = np.asarray(distance.sum(axis=0)).flatten()
        means[sl] = sum_nonzero / num_nonzero[sl]

        distance2 = distance.copy()
        distance2.data[:] **= 2
        sum2_nonzero = np.asarray(distance2.sum(axis=0)).flatten()
        stddevs[sl] = np.sqrt(sum2_nonzero/num_nonzero[sl] - means[sl]**2)

        print 'num nonzero: [%u,%u]' % (num_nonzero[sl].min(), num_nonzero[sl].max())
        print 'means: [%.2f,%.2f]' % (means[sl].min(), means[sl].max())
        print 'std: [%.2f,%.2f]' % (stddevs[sl].min(), stddevs[sl].max())
        print ''


    with open(DATA_PATH + 'collab_p_norm_stats2.pkl','wb') as fp:
        pickle.dump(dict(num_nonzero=num_nonzero,means=means,stddevs=stddevs),fp,protocol=-1)

def cache_collab_neighbors(num_neighbors = 1000):
    '''
    compute stats used in computing a normalized CF distance

    input:
    num_neighbors - maximum number of top results to cache for each song

    outputs:
    dictionary with (key,val) pairs...
    collab_cache['sorted_song_nums'] - list of numpy arrays each containing up to 1000 song numbers sorted from smallest to largest distance.
    collab_cache['sorted_distances'] - list of numpy arrays each containing up to 1000 distances corresponding to the sorted_song_nums array.
    Lists are indexed by the query song number {0,...,~385000}

    Output is pickled to disk
    '''

    with open(DATA_PATH + 'collab_p_norm_stats.pkl','rb') as fp:
        p_norm_stats = pickle.load(fp)
    means = p_norm_stats['means']
    stddevs = p_norm_stats['stddevs']
    num_nonzero = p_norm_stats['num_nonzero']

    num_songs = play_count_log.shape[1]
    chunk_size = 1000 # songs
    num_chunks = num_songs/chunk_size
    partial_chunk = slice(num_chunks*chunk_size,num_songs)

    sorted_song_nums = []
    sorted_distances = []

    t = time.time()


    for c in xrange(num_chunks):
        print 'elapsed time:', time.time() - t
        print 'chunk:', c+1, 'of', num_chunks
        sl = slice(c*chunk_size,(c+1)*chunk_size)
        similarity = compute_similarity_range(sl.start,sl.stop)


        distance = similarity.copy()
        distance.data[:] = -np.log(distance.data)

        print 'num nonzero: [%u,%u]' % (num_nonzero[sl].min(), num_nonzero[sl].max())

        # cache top results for each query
        for i in xrange(chunk_size):

            curr_song_num = c*chunk_size + i

            col = distance.getcol(i)
            nonzero_inds = col.nonzero()[0]
            dist = col.data.copy()

            sorted_nonzero_inds = np.argsort(dist)
            sorted_nonzero_inds = sorted_nonzero_inds[:num_neighbors]

            sorted_dist = dist[sorted_nonzero_inds]
            sorted_nums = nonzero_inds[sorted_nonzero_inds]

            p_normed_distances = 0.5 * ((sorted_dist - means[curr_song_num])/stddevs[curr_song_num] +
                    (sorted_dist - means[sorted_nums])/stddevs[sorted_nums])


            sorted_song_nums.append(sorted_nums)
            #sorted_distances.append(dist[sorted_nonzero_inds])
            sorted_distances.append(p_normed_distances)



    if partial_chunk.stop > partial_chunk.start:
        sl = partial_chunk

        similarity = compute_similarity_range(sl.start,sl.stop)


        distance = similarity.copy()
        distance.data[:] = -np.log(distance.data)

        print 'num nonzero: [%u,%u]' % (num_nonzero[sl].min(), num_nonzero[sl].max())

        # cache top results for each query
        for i in xrange(sl.stop-sl.start):

            col = distance.getcol(i)
            nonzero_inds = col.nonzero()[0]
            dist = col.data.copy()

            sorted_nonzero_inds = np.argsort(dist)
            sorted_nonzero_inds = sorted_nonzero_inds[:num_neighbors]

            sorted_dist = dist[sorted_nonzero_inds]
            sorted_nums = nonzero_inds[sorted_nonzero_inds]

            p_normed_distances = 0.5 * ((sorted_dist - means[curr_song_num])/stddevs[curr_song_num] +
                    (sorted_dist - means[sorted_nums])/stddevs[sorted_nums])


            sorted_song_nums.append(sorted_nums)
            #sorted_distances.append(dist[sorted_nonzero_inds])
            sorted_distances.append(p_normed_distances)


    with open(DATA_PATH + 'collab_cache.pkl','wb') as fp:
        pickle.dump(dict(sorted_song_nums=sorted_song_nums,
            sorted_distances=sorted_distances),fp,protocol=-1)

        



def print_result(song_id,artist,title,value):
    print '%10.4f' % value,
    print song_id, ': ',  artist, ' - ', title

def print_query_results(query_song_ids, output_song_ids, output_values):
    '''
    print a lines that show results of a near neighbors query: 
    song_id/num, artist, title, and a value

    input:
    query_song_ids - iterable 
    output_song_ids - iterable length N
    output_values - iterable length N
    '''

    with sql.connect(DATA_PATH + 'tracks.db') as con:
        cur = con.cursor()

        print '\nQuery:'
        for song_id in query_song_ids:
            cur.execute("SELECT * FROM Tracks WHERE Id = ?", (song_id,))
            rows = cur.fetchall()
            song_id,artist,title = rows[0]
                
            val = 1.0
            print_result(song_id,artist,title,val)

        print ''
        print 'Results:'
        for output_ind in xrange(len(output_song_ids)):

            song_id = output_song_ids[output_ind]
            val = output_values[output_ind]

            cur.execute("SELECT * FROM Tracks WHERE Id = ?", (song_id,))
            rows = cur.fetchall()
            for song_id,artist,title in rows:
                print_result(song_id,artist,title,val)


# construct user/song lookup dictionaries
#construct_testing_user_song_lookup()
#construct_user_song_lookup()
#create_play_count_matrix()
#create_log_play_count_matrix(play_count_matrix)





#def filter_multiple_queries(query_song_ids,num_neighbors=100):
#    '''
#    Return multiple lists of collaboratively filtered near neighbors and p-normed distance scores
#    given a list of song_ids to be used as separate queries
#
#    Input:
#    query_song_ids - list/array of input song ids
#    num_neighbors - maximum number of near neighbors to return
#
#    Output:
#    output_song_ids - list of lists of output song ids
#    output_distances - list of lists of output distances
#    '''
#
#    # convert song_ids to song_nums
#    
#    query_song_nums = np.array([lookup_table['song']['num'].get(str(song_id),None) for song_id in query_song_ids])
#
#    alpha = 0.15
#    similarity = compute_similarity(query_song_nums,alpha)
#
#
#    sorted_song_nums = []
#    sorted_similarities = []
#    for query_ind in xrange(similarity.shape[1]):
#        nonzero_inds = np.nonzero(similarity[:,query_ind])
#        sorted_nonzero_inds = np.argsort(similarity[nonzero_inds,query_ind])[::-1]
#        sorted_nonzero_inds = sorted_nonzero_inds[:num_neighbors]
#        sorted_song_nums.append(nonzero_inds[sorted_nonzero_inds])
#        sorted_similarities.append(similarity[nonzero_inds[sorted_nonzero_inds]])
#
#
#        print '\nNear Neighbors'
#        with sql.connect(DATA_PATH + 'tracks.db') as con:
#            cur = con.cursor()
#
#            cur.execute("SELECT * FROM Tracks WHERE Id = ?", (query_song_ids[query_ind],))
#            rows = cur.fetchall()
#            song_id,artist,title = rows[0]
#                
#            sim = similarity[query_song_nums[query_ind],query_ind]
#            print_query_result(query_song_ids[query_ind],query_song_nums[query_ind],artist,title,sim)
#            print ''
#
#            for result_ind in xrange(len(sorted_song_nums[query_ind])):
#
#                song_num = sorted_song_nums[query_ind][result_ind]
#                sim = sorted_similarities[query_ind][result_ind]
#
#                song_id = lookup_table['song']['id'][song_num]
#
#                cur.execute("SELECT * FROM Tracks WHERE Id = ?", (song_id,))
#                rows = cur.fetchall()
#                for song_id,artist,title in rows:
#                    print_query_result(song_id,song_num,artist,title,sim)
#
#    return sorted_song_ids, sorted_similarities
