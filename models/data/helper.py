'''Below are helper functions used in the 
    manipulation of data and the 
    pandas data frames'''

#operates on data frame to tell you where the zero entries are
def is_zero(df):
    a = df
    b = []
    for i in df.columns:
        lit = i + " == 0.0"
        b.append((i, a.query(lit)))
    #print out list  
    for j in b:
        print (j[0], (j[1].shape)[0])

#### [pandas <----> numpy] functions 

''' For the following functions, "df" is pandas data frame.
    Function "return_data_mat" turns the data frame into 
        numpy array with player name column removed.
    Function "return_names" returns the names column of 
        the players.'''

def return_data_mat(df):
    vals = df.values
    trimvals = delete_nonquant_cols(vals)
    data_mat = trimvals.astype(float)
    return data_mat

def return_names(df):
    names = np.array(df['name']).astype(str)
    return names

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

#### pandas functions

def sort_columns_type(df):
   
    #get idx for string columns
    vals = df.values
    idx = get_idx_str_cols(vals)
    
    #get column list and sort it    
    cols = (df.columns).tolist()
    str_cols = [cols[i] for i in idx]
 
    for x in str_cols:
        cols.remove(x)
    cols = str_cols + cols
    
    # sort new data frame based on new column ordering
    newdf = df[cols]
    return newdf

def get_idx_str_cols(arr):
    
    #get indicies of non-quantitative columns
    idxlist = []
    m, n = arr.shape
    for i in range(m):
        for j in range(n):
            if (type(arr[i,j]) == str):
                idxlist.append(j)
    idxset = set(idxlist)
    idx = list(idxset)
    idx.sort()
    return idx

def delete_nonquant_cols(arr):
    
    m, n = arr.shape
    idx = get_idx_str_cols(arr)
        
    #create a mask with a "False" for each column not desired       
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    
    #index through arr with the mask
    result = arr[:,mask]
    
    #testing that only numbers remain
    a, b = result.shape
    for i in range(a):
        for j in range(b):
            g = type(result[i,j])
            assert (g is float or g is long or g is int or g is complex) 
    
    return result