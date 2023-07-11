from typing import Literal, List
import numpy as np
import pandas as pd
from difflib import get_close_matches

import tensorflow as tf
# print("tf.version: ", tf.version.VERSION)
# print("tf.keras.version: ", tf.keras.__version__)
# print("tf.config.devices: ", tf.config.list_physical_devices())

# # Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
# assert(tf.test.gpu_device_name())

# tf.keras.backend.clear_session()
# tf.config.optimizer.set_jit(True) # Enable XLA.
# # from tensorflow.python.ops.numpy_ops import np_config
# # np_config.enable_numpy_behavior()
# # tf.enable_eager_execution()

# tf.executing_eagerly()

def predicted_distances(vector1: List[int], vector2: List[List[int]], model: tf.keras.Model, number_targets: int) -> np.float32:
    x = np.tile(vector1, (number_targets, 1))
    return model.predict([x, vector2], verbose=0, use_multiprocessing=True).astype('float32').flatten()

def find_Model_ranks(model: tf.keras.Model, df: pd.DataFrame, targets: pd.DataFrame, equivalence_set: pd.DataFrame, ranks: int = 5, find_related_rank: bool = False, report_distances: bool = False) -> pd.DataFrame:
    """For each row in the input dataframe, the model is used to predict the top matching Unique Product Names 
    in 'x' against the entry matches any of the 'y' entries. This is done at the encoded sequence level for 
    both name all unique drugnames.
       
         Parameters
    ----------
    model : tf.keras.Model
        A Keras model based Siamese Network that takes three inputs. 
        Namely, two input sequeces and a third input binary target specifying wether the two sequeces match.
        
    df: pd.DataFrame
        Pandas DataFrame containing the 'x' and 'Processed_x' series to be predicted by the model.
    
    targets: pd.DataFrame
        Pandas Series containing all unique '', 'dUnique_seq_padded' to be use by the model for prediction.
        
    ranks : int 
        The amount of ranks to report. Defaults to 5.
    
    find_related_rank: bool
        A flag indicating wether to compare the top ranked results against the 'y' and it's potential equivalents or not.
        
    report_distances: bool
        A flag indicating return the distance values of the top ranked results against the 'y'.
      
    Returns
    -------
    df : pd.DataFrame
        Returns the padded 'x', 'y', 'rank1', ..., 'rank#' series.
        And additionally the 'exact_rank' and 'equivalent_rank' series and the 'rank1_distance', ..., 'rank#_distance' if requested.
    
    """
    for i in range(1, ranks +1):
        df["rank{}".format(i)] = ""
    
    if report_distances:
        for i in range(1, ranks +1):
            df["rank{}_distance".format(i)] = 1        

    if find_related_rank:
        df.assign(exact_rank= np.Inf, equivalent_rank = np.Inf)
    
    distances = df["Processed_"+ df.columns[0]].apply(predicted_distances, vector2 = np.stack(targets[targets.columns[1]]), number_targets = targets.index.size, model = model)
    sortings = distances.apply(np.argsort, axis=0).apply(lambda x: x[0:ranks])
    
    for i in df.index:
        predicts = distances.at[i]
        argsort = sortings.at[i]
        
        # Top-5 smalles distances
        for n in range(ranks):
            df.at[i, 'rank{}'.format(n+1)] = targets[targets.columns[0]][argsort[n]]  

            if report_distances:
                 df.at[i,'rank{}_distance'.format(n+1)] = predicts[argsort[n]]   

        if find_related_rank:
            ranks_set = ['rank{}'.format(x) for x in range(1, ranks + 1)]
            # Find the top-5 predicted matches
            lookup_clean = df.at[i , df.columns[1]]
            predicted_rank = df.loc[i, ranks_set].eq(lookup_clean).to_numpy().nonzero()
            
            # Find the top ranking correct match, if not rank is infinity so that 1/inf ~ 0, for the MRR computation.
            lookup_rank = np.Inf    
            if len(predicted_rank[0]) > 0 :
                lookup_rank = predicted_rank[0][0] + 1
            df.loc[i, "exact_rank"] = lookup_rank

            # Find all the equivalent common names and latin binomials relative to the look up value that would be equaly correct.
            equivalent = np.setdiff1d(equivalence_set[(equivalence_set["x"] == lookup_clean) | (equivalence_set["y"] == lookup_clean)].unstack().unique(), lookup_rank)

            # Find the top ranking correct match 
            related_rank = np.Inf
            if len(equivalent) > 0:
                for lookup_result in equivalent:
                    annotated_rank = df.loc[i][ranks_set].eq(lookup_result).to_numpy().nonzero()
                    if len(annotated_rank[0]) > 0: 
                        new_related_rank = annotated_rank[0][0] + 1
                        related_rank = min(related_rank, new_related_rank)
                        if related_rank == 1:
                            break
                    
            #find related mappings to lookup value in predicted values 
            df.loc[i, 'equivalent_rank'] = min(lookup_rank, related_rank)
            
    return df

# @tf.function(reduce_retracing=True)  # The decorator converts `normalized_levenshtein` into a tensolflow `Function`.
def normalized_levenshtein(vector1: List[int], vector2: List[int], number_targets: int) -> np.float32:
    x = np.tile(vector1, (number_targets,1))
    return tf.edit_distance(tf.sparse.from_dense(x), tf.sparse.from_dense(vector2), normalize=True)

def find_Levenshtein_ranks(df: pd.DataFrame, targets: pd.DataFrame, equivalence_set: pd.DataFrame, ranks: int = 5, find_related_rank: bool = False, report_distances: bool = False) -> pd.DataFrame:
    """For each row in the input data frame, this function utilizes the Levenshtein distance to find the 
    top 5 unique natural product names that match the row's 'x' string value.
       
       
    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe with the fist column containing 'x' strings to be matched against natural product names.
        Optionally containing Pandas Series with the clean encoded 'y' column.
        
   ranks : int 
        The amount of ranks to report. Defaults to 5.
   
    find_related_rank: bool
        A flag indicating wether to compare the top ranked results against the 'y' and it's potential equivalents or not.
        
    report_distances: bool
        A flag indicating return the distance values of the top ranked results against the 'y'.
      
    Returns
    -------
    df : pd.DataFrame
        Returns the padded 'x', 'y', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5' series.
        And additionally the 'exact_rank' and 'equivalent_rank' series and the 'rank1_distance', 'rank2_distance', 
        'rank3_distance', 'rank4_distance', 'rank5_distance' if requested.
    
    """
    for i in range(1, ranks +1):
        df["rank{}".format(i)] = ""
    
    if report_distances:
        for i in range(1, ranks +1):
            df["rank{}_distance".format(i)] = 1        

    if find_related_rank:
        df.assign(exact_rank= np.Inf, equivalent_rank = np.Inf)
        
    
    distances = df["Processed_"+ df.columns[0]].apply(normalized_levenshtein, vector2 = np.stack(targets[targets.columns[1]]), number_targets = targets.index.size).transform(lambda x : np.array(x))
    sortings = distances.apply(np.argsort, axis=0).apply(lambda x: x[0:ranks])

    for i in df.index:
        predicts = distances.at[i]
        argsort = sortings.at[i]
        
        # Top-5 smalles distances
        for n in range(ranks):
            df.at[i, 'rank{}'.format(n+1)] = targets[targets.columns[0]][argsort[n]]  

            if report_distances:
                 df.at[i,'rank{}_distance'.format(n+1)] = predicts[argsort[n]]   

        if find_related_rank:
            ranks_set = ['rank{}'.format(x) for x in range(1, ranks + 1)]
            # Find the top-5 predicted matches
            lookup_clean = df.at[i , df.columns[1]]
            predicted_rank = df.loc[i, ranks_set].eq(lookup_clean).to_numpy().nonzero()
            
            # Find the top ranking correct match, if not rank is infinity so that 1/inf ~ 0, for the MRR computation.
            lookup_rank = np.Inf    
            if len(predicted_rank[0]) > 0 :
                lookup_rank = predicted_rank[0][0] + 1
            df.loc[i, "exact_rank"] = lookup_rank

            # Find all the equivalent common names and latin binomials relative to the look up value that would be equaly correct.
            equivalent = np.setdiff1d(equivalence_set[(equivalence_set["x"] == lookup_clean) | (equivalence_set["y"] == lookup_clean)].unstack().unique(), lookup_rank)

            # Find the top ranking correct match 
            related_rank = np.Inf
            if len(equivalent) > 0:
                for lookup_result in equivalent:
                    annotated_rank = df.loc[i][ranks_set].eq(lookup_result).to_numpy().nonzero()
                    if len(annotated_rank[0]) > 0: 
                        new_related_rank = annotated_rank[0][0] + 1
                        related_rank = min(related_rank, new_related_rank)
                        if related_rank == 1:
                            break
                    
            #find related mappings to lookup value in predicted values 
            df.loc[i, 'equivalent_rank'] = min(lookup_rank, related_rank)
            
    return df

def find_Gesalt_ranks(df: pd.DataFrame, targets: pd.DataFrame, equivalence_set: pd.DataFrame, ranks: int = 5, find_related_rank: bool = False) -> pd.DataFrame:
    """For each row in the input data frame, this function utilizes the difflib implementation of fuzzy string match
       to find the top 5 unique natural product names that match the row's 'x' string value.
       
       
    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe with the fist column containing 'x' strings to be matched against natural product names.
        Optionally containing Pandas Series with the clean encoded 'y' column.
        
   ranks : int 
        The amount of ranks to report. Defaults to 5.
    
    find_related_rank: bool
        A flag indicating wether to compare the top ranked results against the 'y' and it's potential equivalents or not.
      
    Returns
    -------
    df : pd.DataFrame
        Returns the padded 'x', 'y', 'rank1', ..., 'rankN' series.
        And additionally the 'exact_rank' and 'equivalent_rank' series if requested.
    
    """   
    ranks_set = ['rank{}'.format(x) for x in range(1, ranks + 1)]
    
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                    df[df.columns[0]].apply(lambda x: get_close_matches(x,targets[targets.columns[0]].to_list(), n = ranks, cutoff=0.0)).to_list(),
                columns = ranks_set,
                index = df.index
            )
        ], 
        axis= 1, 
        join="inner"
    )
    
    if find_related_rank:
        df.assign(exact_rank= np.Inf, equivalent_rank = np.Inf)
        for i in df.index:     
            # Find the top-5 predicted matches
            lookup_clean = df.at[i , df.columns[1]]
            predicted_rank = df.loc[i, ranks_set].eq(lookup_clean).to_numpy().nonzero()
            
            # Find the top ranking correct match, if not rank is infinity so that 1/inf ~ 0, for the MRR computation.
            lookup_rank = np.Inf    
            if len(predicted_rank[0]) > 0 :
                lookup_rank = predicted_rank[0][0] + 1
            df.loc[i, "exact_rank"] = lookup_rank

            # Find all the equivalent common names and latin binomials relative to the look up value that would be equaly correct.
            equivalent = np.setdiff1d(equivalence_set[(equivalence_set["x"] == lookup_clean) | (equivalence_set["y"] == lookup_clean)].unstack().unique(), lookup_rank)

            # Find the top ranking correct match 
            related_rank = np.Inf
            if len(equivalent) > 0:
                for lookup_result in equivalent:
                    annotated_rank = df.loc[i][ranks_set].eq(lookup_result).to_numpy().nonzero()
                    if len(annotated_rank[0]) > 0: 
                        new_related_rank = annotated_rank[0][0] + 1
                        related_rank = min(related_rank, new_related_rank)
                        if related_rank == 1:
                            break
                    
            #find related mappings to lookup value in predicted values 
            df.loc[i, 'equivalent_rank'] = min(lookup_rank, related_rank)
            
    return df
