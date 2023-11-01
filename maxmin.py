import numpy as np
import random
  
    
def maxmin_from_class(data, pool_cl, n, p=np.inf):
    
    if n <= 0:
        raise ValueError("Error: n must be a positive number")
    else:
        lendata = len(pool_cl)
        if n >= lendata:
            return pool_cl
        else:
            pool=np.arange(lendata)
            r=random.randrange(lendata)
            picks= np.array([r],dtype=int)
            pool=np.delete(pool,r)
            picked=1
            distmat=np.array([[r,pool[0],np.linalg.norm(np.subtract(data[picks[0]],data[pool[0]]),ord=p)]])
            for i in range(1,lendata-1):
                distmat = np.vstack((distmat,np.array([[r,pool[i],np.linalg.norm(np.subtract(data[picks[0]],data[pool[i]]),ord=p)]])))
            while picked < n:
                picks, pool, distmat = pick_one(picks,pool,data,distmat,picked,p)
                picked+=1
            return np.sort(pool_cl[picks])

def pick_one(picks,pool,data,distmat,picked,p):
                    
    n_pool=len(pool)
    current_max=0
    current_index=-1
    for i in range(n_pool):
        current_max, current_index, distmat = min_search(i,picks,pool,data,distmat,picked,current_max,current_index,p)
    picks = np.append(picks, pool[current_index])
    distmat = distmat[distmat[:,1]!=pool[current_index]]
    pool = np.delete(pool, current_index)
    return picks, pool, distmat

def min_search(i,picks,pool,data,distmat,picked,current_max,current_index,p):
                    
    current_min = np.inf
    pooli = pool[i]
    for j in range(picked):
        d=np.inf
        picksj = picks[j]
        filt = distmat[(distmat[:, 0] == picksj) & (distmat[:, 1] == pooli)]
        if np.any(filt):
            d=filt[0][2]
        else:
            d=np.linalg.norm(np.subtract(data[picksj],data[pooli]),ord=p)
            distmat=np.vstack((distmat,np.array([[picksj,pooli,d]])))
        if d < current_max:
            return current_max, current_index, distmat
        elif d < current_min:
            current_min = d
    current_max = current_min
    current_index = i
    return current_max, current_index, distmat
        


        
 