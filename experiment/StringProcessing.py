import numpy as np
import pandas as pd
import networkx as nx
import random
import pickle
import re
import string
from typing import Literal, List

# random.seed(2022)

global encode_dict 
encode_dict = {l:i for i,l in enumerate(string.ascii_uppercase + " ", 1)}

'''
Parameters
----------
text : str

Returns
-------
text : str
'''

def clean(text: str) -> str:
    '''Removes all the non-ascii and special characters from a string and 
    returns the string's alphabetichal characters with spaces.
    
    Expects a string to be cleaned and removes all the non-ascii and special characters. 
    This is done by applying a substitution to regex matches
    Returns the cleaned string containing uppercased versions of the characters.
    
    Parameters
    ----------
    text : str
        Text to be cleaned
        
    Returns
    -------
    Cleaned and uppercased text. Eg. "Ac-d" -> "ACD"
    '''
    regex = re.compile('[^a-zA-Z ]')
    r = regex.sub('', text)
    result = re.sub(' +', ' ', r)
    result = result.strip()
    return result.upper()

def add_noise(w: str, percent: float, maxlen: int) -> str:
    ''' Adds a specified proportion of noise to a string.
    
    Expects a string and a number stating the percent of noise to add to this string.
    The string is modified by editing, deleting, or adding characters in/to the string.
    The modification to perform is determined randomly by generating a random number 
    from an uniform distribution [0,1].
    If the number is < 1/5 edit one position with new random character or space.
    If the number is < 2/5 delete one position.
    If the number is < 3/5 add one random character or space.
    If the number is < 4/5 transpose one position.
    Finally, if the number is >= 4/5 do not add noise.
    
    In order to retain the length of the sequence compliant with the maximum sequence 
    length, additional processing has been added such that sequences that reach 
    the maximum sequence length can only be modified by removing or swapping characters.
    
    Parameters
    ----------
    w : str
        The string to add noise to.
    
    percent: float
        Percentange representing the proportion of noise to add to the string.
        
    maxlen: int
        Maximun lengnth of a valid sequence.
        
    Returns
    -------
    Modified string with noise added. Eg "ACD" -> "AE D"
    '''
    w_size = len(w)
    positions = random.choices(range(0, w_size), k=round(percent * w_size))
    for p in positions:
        curr_w = len(w)
        r = random.uniform(0,1)
        if p > curr_w - 1:
            p = curr_w - 1
        if w_size < maxlen:
            if   r < 0.2: # edit
                w = w[:p] + random.choice(string.ascii_uppercase + " ") + w[p+1:]
            elif r < 0.4: # delete
                w = w[:p] + w[p+1:]
            elif r < 0.6: # add
                w = w[:p] + random.choice(string.ascii_uppercase + " ") + w[p:]
            elif r < 0.8: # transpose  
                if p > 0:
                    w = "".join([w[:p-1], w[p], w[p-1], w[p+1:]])
                else:
                    w = "".join([w[p+1], w[p], w[p+2:]])    
            # else: # skip adding noise 
            # do nothing

        else:
            if   r < 0.25: # edit
                w = w[:p] + random.choice(string.ascii_uppercase + " ") + w[p+1:]
            elif r < 0.50: # delete
                w = w[:p] + w[p+1:]
            elif r < 0.75: # transpose
                if p > 0:
                    w = "".join([w[:p-1], w[p], w[p-1], w[p+1:]])
                else:
                    w = "".join([w[p+1], w[p], w[p+2:]])
            # else: # skip adding noise 
            # do nothing
            
    final_size = len(w)
    if final_size > maxlen:
        positions = random.choices(range(0, final_size), k = (final_size - maxlen))
        w = ''.join([w[i] for i in range(0, final_size) if i not in positions])
    
    return w

def balance_dataset(df: pd.DataFrame, noise_percent: float, maxlen: int) -> pd.DataFrame:
    '''It balances the data within a given DataFrame by adding positive values.
    
    df: pd.DataFrame
        Expects a pd.DataFrame object with columns ['x', 'y'].
        
    noise_percent: float
        Indicates the noise threshold. Wich is the maximun ammount 
        of noise allowable to be added to the sequence.
    
    maxlen: int
        Maximun lengnth of a valid sequence.
    
    Returns
    -------
    A copy of the input pd.DataFrame balanced with regards to column "y", positive "x, y" pairs 
    are created by adding noise to "x".  Eg. ["ACD", "ACD"] -> ["AE D", "ACD"] 
    '''
    df["y"].value_counts(ascending=True).plot(rot=90)
    counts = df["y"].value_counts(ascending=True)
    max_set = round(counts.mean() + counts.std() * 4) 
    for target in counts.index:
        increase_size = round(max_set - counts[target])
        if increase_size > 0:
            tmp_df = pd.DataFrame(columns=['x', 'y'])
            increase_set = df[df["y"] == target]["x"].sample(n=increase_size, replace= True)
            tmp_df["x"] = increase_set.apply(add_noise, percent = noise_percent, maxlen = maxlen).to_list()
            tmp_df["y"] = [target] * increase_size
            df = pd.concat([df, tmp_df], ignore_index=True)
    df["y"].value_counts(ascending=True).plot(rot=90)
    return df
        
def encode_sequence(string: str) -> List[int]:
    '''Maps a string from text representation to am integer list representation.
    
    Parameters
    ----------
    string : str
        Sting to be encoded into an int sequence.
    
    Returns
    -------
    The string as an Integer sequence (List[Int]). Eg. "ACD" -> [1,3,5]. 
    '''
    return list(map(encode_dict.get, string))
    
def pad_sequence(sequence: List[int], maxlen: int) -> List[int]:
    '''Adds '0' as padding character up to the specified lenght.
    
    Parameters
    ----------
    sequence : List[int]
        Encoded sequence where string characters have been mapped to integers.
        
    maxlen:
        Target size of the padding. 
        
    Returns
    -------
    The padded sequence Eg. [1,3,5] -> [1,3,5, ..., 0, 0, 0]. 
    '''
    return sequence + ([0] * (maxlen -len(sequence)))

def preprocessInput(filename: str, maxlen: int, reflexive: bool, balance: bool, noise: float, **kwargs) -> pd.DataFrame:
    '''Preprocess CSV file into a Pandas DataFrame.
    
    Expects the file name or path of a csv file with named columns containing strings 
    representing product names. It then removes the sequences with length greater than
    the maximun sequence length, cleans the sequences and uppercases them, and it finally
    drops any duplicates that might have arrisen from this processing. Returns a Pandas 
    Dataframe containing unique cleaned and uppercased versions of the strings on each cell.
    
    Parameters
    ----------
    filename : str
    
    maxlen: int
        It's used to determine the ammount of padding to add
        to sequences smaller than maxlen.
    
    reflexive: bool
        For every pair (x,y) ensure (y,x) is also in the set.
    
    balance: bool
        Wether the target y values should be balanced.
        
    noise: float
        The threshhold of the maximum ammount of noise to add to a string.
    
    **kwargs:
        Keyword arguments for pandas read csv function. 
        
    Returns
    -------
    df : Pandas DataFrame
    '''  
    df = pd.read_csv(filename, **kwargs)
    print(df.info())
    print("Processing file: ----------------------------------------")    
    print("Renaming colums:")
    print("\tCurrent names: {}".format(df.columns))
    cols = df.columns.size
    match cols:
        case 1: 
            df.columns = ["x"]
        case 2:
            df.columns = ["x", "y"]
    print("\tNew names: {}".format(df.columns))
    
    print("Dropping row with empty cells:")
    original_count = df.index.size
    df.dropna(subset=df.columns, inplace=True)
    new_count = df.index.size
    print("\tDropped", original_count - new_count, "rows with empty cells.")
    
    print("\tCleaning string sequences.")
    df = df.applymap(clean)
    
    print("\tUppercasing string sequences.")
    df = df.applymap(lambda x: str.upper(x))
    
    if reflexive and len(df.columns) == 2:
        df = pd.concat([df, pd.DataFrame.from_dict({"x": df['y'].to_list(), "y": df['x'].to_list()})], axis = 0)
    
    if balance:
        print("Balancing target sequence representation:")
        original_count = df.index.size
        df = balance_dataset(df, noise_percent = noise, maxlen = maxlen)
        new_count = df.index.size
        print("\tAdded", new_count - original_count, "new sequences of underrepresented targets.")
    
    print("Dropping sequences longer than the maxlen of {}:".format(maxlen))
    original_count = df.index.size
    for column in df.columns:
        df.drop(df[df[column].apply(len).gt(maxlen)].index, inplace = True)
    new_count = df.index.size
    print("\tDropped", original_count - new_count, "that exceeded the maximum sequence length.")
        
    print("Dropping duplicate sequences:")
    original_count = df.index.size
    df.drop_duplicates(ignore_index=True, inplace=True)
    new_count = df.index.size
    print("\tDropped", original_count - new_count, "duplicate sequences.")
    
    print("Done processing: ---------------------------------------")
    print(df.info())
    return df


def encode_pad_tag(df: pd.DataFrame, match: Literal[0,1], distance: Literal[0,1], maxlen: int, verbose: bool = True) -> pd.DataFrame:
    '''It encodes, pads and tags the preprocessed sequences in a Pandas DataFrame.
    
    Expects a pandas dataframe with cleaned and uppercased sequences. It processes the 
    the DataFrame by creating an additional 'Processed_' + current column name for each
    of the columns in the data frame, where each of the sequences in the column get 
    transformed from a string sequence to an encoded sequence and then transformed again 
    by padding the encoded sequences up to the maximun sequence length by 0's as needed. 
    The tag represents wheter the sequences Match (1) or not (0) and the Distance weather the 
    embeddings should be distant (1) or close (0). Finally, this function returns this 
    dataframe with both the original and processed columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sequences.
    
    match: Literal[0 | 1]
        Tag indicating wether sequences match 1 indicates 'yes' and 0 indicates 'no'.
        
    distance: Literal[0 | 1]
        Distance between sequences, 1 indicates 'far' and 0 indicates 'close'. 
        
    maxlen: int
        Dafaults to 65. It's used to determine the ammount of padding to add
        to sequences smaller than maxlen.
        
    Returns
    -------
    df : Pandas DataFrame
        A copy of the origininal DataFrame with the processed sequences added as new columns.
    '''
    if verbose:
        print("Encoding and Padding: ----------------------------------")
    for column in df.columns:
        if verbose:
            print("\tProcessing {}".format(column))
        df["Processed_" + column] = df[column].apply(encode_sequence).transform(pad_sequence, maxlen = maxlen)
    if verbose:
        print("Tagging: -----------------------------------------------")
    df["Match"] = match
    df["Distance"] = distance
    return df

def generate_noisy_positive_pairs(df: pd.DataFrame, scale: float, noise_percent: float, maxlen: int) -> pd.DataFrame:
    '''Creates noisy positive pairs by adding some noise to the sequences in the 'x'
    column while retaining the match to the correct 'y'.

    For each unique name in the 'x' column of the train set, get the product name
    and apply noise to a specified percentage of the sequence. Finally, return a DataFrame with
    the generated synthethic noisy sequences encoded, padded and tagged.
        
    Parameters
    ----------
    df: pd.DataFrame
        A Pandas DataFrame containing the 'x' and the 'y' Series.
        
    scale: float
        The scale of data to be generated relative to the size of the true positives. 
        Eg. scale = 1.0 generates approximately a 1:1 DataFrame with equivalent noisy 
        sequences relative to the input DataFrame's true positives. And scale = 2.50 
        generates a 2.5:1 meaning an output Dataframe of 2 times and a half times the 
        size of the input DataFrame.
        
    noise_percent: float
        Indicates the approximate percentage of noise to add to each sequence. 
    

    Returns
    -------
    df : pd.DataFrame
        Returns a DataFrame containing the 'x', 'y', 'Processed_x', 'Processed_y', 
        'Match' and 'Distance' pd.Series from the synthetic data.
    '''
    noisy = pd.DataFrame(columns=['x', 'y'])
    x = []
    y = []
    
    if scale >= 1.0:
        for i in range(int(scale)):
            x.extend(df['x'].apply(lambda x: add_noise(x, noise_percent, maxlen)).to_list())
            y.extend(df['y'].to_list())
    
    remainder = scale - int(scale)
    if remainder > 0:
        remaining_sample = df.sample(frac=remainder)
        x.extend(remaining_sample['x'].apply(lambda x: add_noise(x, noise_percent, maxlen)).to_list())
        y.extend(remaining_sample['y'].to_list())

    noisy['x'] = x
    noisy['y'] = y
    del x
    del y
    encode_pad_tag(noisy,  match=1, distance = 0, maxlen= maxlen)
    return noisy

def get_target_groups_from_connected_components(df: pd.DataFrame) -> List[set]:
    '''Proceses the 'x' and 'y' pairs to create a graph that connects the associated terms.
    Then extracts the groups of connected componnents from the graph. 
    
    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame containing the 'x' and the 'y' Series.
        
    Returns
    -------
        An array of the sets of unique terms that are paired among themselves. 
        
        Eg. "x","y"      List[set]
            [A, B] \
            [B, C]  => [{A,B,C}, {D}]  
            [D, D] /
    '''
    unique_targets = df['y'].unique()
    subset = df['x'].apply(lambda x: x in unique_targets)
    pairs = df[subset][['x','y']].itertuples(index=False)
    G = nx.Graph()
    G.add_edges_from(pairs)
    return list(nx.connected_components(G))

def get_non_match(row, target_groups: List[set], noise: float, maxlen: int) -> [str, str]:
    '''
    Parameters
    ----------
    row : [x,y]
        A row of the dataset containing and x and y pair.
        
    target_groups: List[set]
        A list of sets, where each set represents an equivalence group of product names.
        
    noise: float
        The maximun ammount of noise to add to the sequence.
        
    maxlen: int
        Maximum size of the sequence.
        
    Returns
    -------
    text : str
    '''
    x, y = row
    group = random.choice(target_groups)
    while y in group: 
        group = random.choice(target_groups)
    new_x = random.choice(list(group))
    r = random.uniform(0,1)
    if r < 0.5:
        new_x = add_noise(new_x, percent = noise, maxlen = maxlen)
    return new_x, y

def generate_synthethic_negative_pairs(df: pd.DataFrame, equivalences: List[set], noise: float,  scale: float, maxlen: int) -> pd.DataFrame:
    '''Create negative pairs where 'x' does not match the correct 'y'.

    For each unique name in the 'y' column sample of the train set, get the name and
    and then pick four random different product names. For each of those 4 additional product names 
    check if it matches any of the names in the training set if its not then add it to the dataset as 
    a negative pair. The goal of this is to help further distance the embeddings in the vector space.
    Returns a DataFrame containing those negative sequences encoded, padded and tagged.

    
    Parameters
    ----------
    df: pd.DataFrame
         A Pandas DataFrame containing the 'x' and 'y' Series.
         
    scale: float
        The scale of data to be generated relative to the size of the true positives. 
        Eg. scale = 1.0 generates approximately a 1:1 DataFrame with equivalent true 
        negative pairs relative to the input DataFrame's true positives. And scale = 1.5
        generates a 1.5:1 meaning an output Dataframe ~ 1 and a 1/2 times the size of the input DataFrame.
    
    Returns
    -------
     df : pd.DataFrame
        Returns a DataFrame containing the 'x', 'y', 'Processed_x', 'Processed_y', 'Match', and 'Distance' pd.Series from the synthetic data.
    '''
    synthethic = pd.DataFrame(columns=['x', 'y'])
    
    if scale >= 1.0:
        for _ in range(int(scale)):
            tmp = df[["x", "y"]].transform(get_non_match, axis=1, target_groups = equivalences, noise = noise, maxlen = maxlen)
            synthethic = pd.concat([synthethic, tmp], ignore_index=True)
    
    remainder = scale - int(scale)
    if remainder > 0:
        remaining_sample = df.sample(frac=remainder)
        tmp = remaining_sample[["x", "y"]].transform(get_non_match, axis=1, target_groups = equivalences, noise = noise, maxlen = maxlen)
        synthethic = pd.concat([synthethic, tmp], ignore_index=True)

    encode_pad_tag(synthethic, match = 0, distance = 1, maxlen= maxlen)
    return synthethic

def balance_complete_set(df: pd.DataFrame, target_groups: List[set], noise_percent: float, maxlen: int):
    '''Attempts to balance the input DataFrame by adding 50% positve and 50% negative synthethic pairs up to the size of 
    the maximum represented target "y". The goal is to achieve a somewhat uniform representation of all "y" targets.
    
    
    Parameters
    ----------
    df : str
         A Pandas DataFrame containing the 'x', 'y', and 'Match' series.
    
    target_groups: List[set]
        A list of sets, where each set represents an equivalence group of product names.
        
    noise_percent: float
        The maximun ammount of noise to add to the sequence.
    
    maxlen: int
        Maximum size of the sequence.
    
    Returns
    -------
    A copy of the input dataframe with the additional records.
    '''
    counts = df["y"].value_counts(ascending=True)
    counts.plot(rot=90)
    for target in counts.index:
        increase_size = round(counts.mean() + counts.std() * 4) - counts[target]
        if (increase_size//2) > 0:
            increase_set = df[(df["y"] == target) & (df["Match"] == 1)]["x"].sample(n= (increase_size//2), replace= True)
            tmp_df_pos = pd.DataFrame(columns=['x', 'y'])
            tmp_df_pos["x"] = increase_set.transform(add_noise, percent = noise_percent, maxlen = maxlen).to_list()
            tmp_df_pos["y"] = [target] * (increase_size//2)
            tmp_df_pos = encode_pad_tag(tmp_df_pos, match=1, distance=0, maxlen=maxlen, verbose = False)
            
            increase_set = df[df["y"] == target][["x", "y"]].sample(n = (increase_size//2), replace= True)
            tmp_df_neg = increase_set.transform(get_non_match, axis=1, target_groups = target_groups, noise = noise_percent, maxlen = maxlen)
            tmp_df_neg = encode_pad_tag(tmp_df_neg, match=0, distance=1, maxlen= maxlen, verbose = False)
            
            df = pd.concat([df, tmp_df_pos, tmp_df_neg], ignore_index=True)
    df.drop_duplicates(subset=['x', 'y'], inplace=True)
    for column in ['x', 'y']:
        df.drop(df[df[column].apply(len).gt(maxlen)].index, inplace = True)
    df["y"].value_counts(ascending=True).plot(rot=90)
    return df