import pickle

# Read pkl GT file     
def GTlist(pkl_fle):
    resultList = []
    with open(pkl_fle, 'rb') as f:
        data = pickle.load(f)    
    for query in data:
        queryList = []
        for entry in query[1]:
            if(entry >= 0):
                queryList.append('ima_{:06d}.jpg'.format(entry))
            else:
                queryList.append(-1)
        resultList.append(queryList)
    return resultList
    