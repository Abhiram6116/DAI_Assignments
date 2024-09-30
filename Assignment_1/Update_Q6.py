from math import sqrt
# k is the new data value.


def UpdateMean(old_mean, k, n, A) :
    new_mean = ((old_mean*n)+k)/(n+1) 
    return new_mean 

def UpdateStd(OldMean, OldStd, NewMean, k, n, A) :
    
    new_var = ((OldStd**2)*n + (k**2) + n*(OldMean**2) - (n+1)*(NewMean**2))/(n+1)
    NewStd = sqrt(new_var) 
    return NewStd  
    
# OldMedian is M.
def UpdateMedian(M, k, n, A):
    B =A # Make a copy
    
    # The idea is that all we need are the TWO ELEMENTS surrounding the median to know the new Median.
    # We just throw in all the elements less than median on 1 side by iterating over once.
    # Then we compute max of the them and min of the others excluding median to get the 2 required values.
    # Then just see where k is going wrt them and we are done.
    # Overall an O(n) algorithm .
    
    # n is odd.
    if (n%2 == 1):
        t = (n-1)//2
        count = 0
        median_index = 0
        while(A[median_index]!=M) : median_index+=1
        B[median_index],B[t] = B[t],B[median_index]
        for i in range(n):
            if (A[i]<=M and i!=t):
                B[count],B[i] = B[i], B[count] 
                count+=1
                if(count == t) : break
        # find max in the first (n-1)/2 and min in last (n-1)/2.
        max, min = B[0],B[t+1] 
        for i in range(t) :
            if B[i]>max : max = B[i]
        for j in range(t+1, n):
            if B[j]<min : min = B[j]
        # Now we have the median and the 2 elements surrounding it .
        # Update the median.
        if (k>min):  
            newMedian = (M+min)/2
        elif(k<max):
            newMedian = (M+max)/2
        else:
            newMedian = (k+M)/2 
    
    # if n is even.
    else:
        med_found = False
        for i in range(n):
            if A[i]==M :
                med_found = True
                break
        
        if (med_found) :
            min = max = M 
        else:
            t = n//2
            count = 0
            for i in range(n):
                if (A[i]<=M):
                    B[count],B[i] = B[i], B[count] 
                    count+=1
                    if(count == t) : break
            # find max in the first n/2 and min in last n/2.
            max, min = B[0],B[t] 
            for i in range(t) :
                if B[i]>max : max = B[i]
            for j in range(t, n):
                if B[j]<min : min = B[j]
            
        # Now we have the two elements whose average was our median M.
        # Update the median.
        if (k>min) : newMedian = min
        elif(k<max) : newMedian = max
        else : newMedian = k
    
    return newMedian



    
    
    
    