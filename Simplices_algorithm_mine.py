import numpy as np
from numpy import random
import sympy
from sympy import Matrix, ZZ, GF
from sympy.matrices.normalforms import smith_normal_form
import itertools
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import re
#import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
from scipy import stats
#from phonemizer import phonemize
#from phonemizer.backend.espeak import EspeakBackend
import os
import jellyfish
#from rapidfuzz.distance import Levenshtein
#from metaphone import doublemetaphone
from ripser import ripser
from persim import plot_diagrams
#from persim import plot_barcodes


#os.environ["PATH"] = r"C:\\Program Files\\eSpeak NG;" + os.environ["PATH"]
#os.add_dll_directory(r"C:\\Program Files\\eSpeak NG")

X = ["A", "B", "C", "D","AB", "AC", "BC", "BD", "CD", "BCD"]

square = ["A", "B", "C", "D","AB", "AC", "BC", "BD", "CD", "ABC", "BCD"]

opposites_identify_disc = ["A", "B", "C", "D", "E", "F", "AB", "AC", "AD", "AE", "AF", "BC", "BD", "BE", "BF", "CD", "CE", "CF", "DF", "EF", "ABC", "ABD", "ABE", "ABF", "ACD", "AEF", "BCE", "BDF", "CDF", "CEF"]

circle = ["A", "B", "C", "AB", "AC", "BC"]

torus = ["A","B","C","D","E","F","G","H","I","AB","AC","AD","AE","AG","AH","BC","BD","BF","BH","BI","CE","CF","CG","CI","DE","DF","DG","DI","EF","EH","EI","FG","FH","GH","GI","HI","ABD","ABH","ACE","ACG","ADG","AEH","BCF","BCI","BDF","BHI","CEI","CFG","DEF","DEI","DGI","EFH","FGH","GHI"]

tetra = ["A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD"]

triangle = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
def count_letters(string):
    count = 0
    for letter in string:
        if letter.isalpha():
            count += 1
    return count


def create_mats(enumerated, field):
    max_n = max(len(n) for n in enumerated)
    #vertices = len([simplex for simplex in enumerated if count_letters(simplex)==1])
    vertices = len([simplex for simplex in enumerated if len(simplex)==1])
    zero_mat = np.zeros([0,vertices])

    mat_list = [zero_mat]
    j_simps = []
    for k in range(max_n-1):
        i_simp_len = k+1
        j_simp_len = k+2
        i_simps = [simplex for simplex in enumerated if len(simplex)==i_simp_len]
        j_simps = [simplex for simplex in enumerated if len(simplex)==j_simp_len]
        
        del_mat = np.zeros([len(i_simps),len(j_simps)])
        
        if (del_mat.shape[0] != 0) or (del_mat.shape[1] != 0):
            for i in range(len(i_simps)):
                for j in range(len(j_simps)):
                    del_mat[i,j] = get_coeff(i_simps[i],j_simps[j], field)

        mat_list.append(del_mat)
    
    end_zeros = np.zeros([len(j_simps),0])
    mat_list.append(end_zeros)

    return(mat_list)





def get_coeff(small_simp, big_simp, field):
    edges = []
    for i in range(len(big_simp)):
        edge = big_simp[:i]+big_simp[i+1:]
        edges.append(edge)
    #print(edges)
    for i in range(len(edges)):
        if small_simp == edges[i]:
            if field == "Z2":
                return 1
            return (-1)**(i)
    return 0


def get_coeff2(small_simp, big_simp, field):
    """
    Version of above function where vertices are of the form P#,
    so simplices are of the form "P#P#P#", eg "P1P2P21P34"
    """
    parts = re.split(r'(?=[A-Za-z])', big_simp)
    edges = [p for p in parts if p]
    for i in range(len(edges)):
        edge = ""
        for j in range(len(edges)):
            if i != j:
                edge += edges[j]
        if small_simp == edge:
            if field == "Z2":
                return 1
            return (-1)**(i)
    return 0

#print(get_coeff("ABCD", "ABE", "Z_2"))


#print(create_mats(X,"Z2"))




def find_homology_groups(mat_list, field):
    
    betti = []
    for i in range(len(mat_list)-1):
        if field == "Z2":
            ker_mat = Matrix( mat_list[i])
            im_mat = Matrix(mat_list[i+1])
            s = smith_normal_form(ker_mat, domain = GF(2))
            dim_ker = dim_im = 0
            for j in range(min(ker_mat.shape)):
                if int(s[j,j]) == 0:
                    dim_ker += 1
            s2 = smith_normal_form(im_mat, domain = GF(2))
            for j in range(min(im_mat.shape)):
                if int(s2[j,j]) != 0:
                    dim_im += 1
                
            #print(dim_ker - dim_im)
            
        
        else:
            ker_mat = mat_list[i]
            im_mat = mat_list[i+1]
            
            #print("kernel matrix is", (ker_mat), "\n", "image matrix is", im_mat, "\n", "Group", i+1)

            if ker_mat.size == 0:
                null_vect = np.eye(ker_mat.shape[1])
            else:
                u,s,v = np.linalg.svd(ker_mat)
                null_vect = np.hstack((v.T[:, :len(s)][:,s<=1e-5],(v.T[:,len(s):])))
            #print("for", i, "a kernel basis is", null_vect, "with dimension", null_vect.shape[1])

            if im_mat.size == 0:
                image_vect = np.zeros([ker_mat.shape[1],0])     
            else:

                u2,s2,v2 = np.linalg.svd(im_mat)
                image_vect = v2.T[:, :len(s2)][:,s2>1e-5]
                #image_vect = v2.T[:,s2>1e-5]
            #print("for", i, "an image vector is", image_vect, "with dimension", image_vect.shape[1])

            
            #print("So we have betti number", i, "is", null_vect.shape[1]-image_vect.shape[1])
            betti.append(null_vect.shape[1]-image_vect.shape[1])
    #print(betti)
    return(betti)


        

#find_homology_groups(create_mats(circle, "Z2"), "Z2")

#print(create_mats(X, "real"))


test_mat = np.zeros([3,3])

for i in range(3):
    for j in range(3):
        test_mat[i,j] = 3*i + j




transform_mat = np.zeros([5,5])

transform_mat[0,1] = transform_mat[1,0] = 2

for i in range(2,5):
    transform_mat[i,i] = 1

transform_mat[2,3]=transform_mat[2,4]=transform_mat[3,2]=transform_mat[4,2]=transform_mat[3,4] = transform_mat[4,3] = 1


#print(transform_mat@test_mat)

def unit_up_diag(mat):
    #print("original mat is", mat)
    mat = mat.copy()

    if np.allclose(mat, 0, atol=1e-12):
        return mat, np.eye(mat.shape[0])

    if mat.shape[1] == 0:
        # No columns left — identity transform on however many rows remain
        return mat, np.eye(mat.shape[0])
    if np.allclose(mat[:,0], 0, atol=1e-6) and mat.shape[1] == 1:
        return mat, np.eye(mat.shape[0])
    if mat.size==0:
        size = max(mat.shape[0]-1, 0)
        
        return mat, np.eye(size)
    m,n = mat.shape
    if m == 1:
        pivot = mat[0,0]
        if abs(pivot) <= 1e-6:
            return mat, np.eye(m)
        U = mat*(1/pivot)
        P = np.eye(m)*(1/pivot)
        
        return U, P
    perm_mat = np.eye(m)
    col = mat[:,0]
    pivot = max(col, key=abs)
    
    if abs(pivot) <= 1e-6:
        rest, rest_perm = unit_up_diag(mat[:,1:])
        if np.allclose(rest, 0, atol=1e-6):
            return np.zeros_like(mat), np.eye(m)
        #full_perm = np.eye(m+1)
        #full_perm[1:, 1:] = rest_perm
        zeros = np.zeros((m,1))

        if rest.size == 0:
            new_mat = zeros
        else:
            new_mat = np.hstack((zeros, rest))
        
        return new_mat, rest_perm




    # zero_cols = 0
    # while zero_cols < mat.shape[1] and np.allclose(mat[:, zero_cols], 0, atol=1e-6):
    #     zero_cols += 1

    # if zero_cols > 0:
    #     # strip ALL zero columns at once
    #     stripped = mat[:, zero_cols:]

    #     # if nothing left after stripping → nothing more to pivot on
    #     if stripped.size == 0:
    #         # return the zero-matrix as is
    #         return mat, np.eye(mat.shape[0])

    #     # recurse on the stripped part
    #     rest, rest_perm = unit_up_diag(stripped)

    #     # revive the zero columns on the left
        
    #     ### NEED THESE FOR EXACT
    #     zeros = np.zeros((mat.shape[0], zero_cols))
    #     new_mat = np.hstack((zeros, rest))

    #     return new_mat, rest_perm



    pivot_row = list(col).index(pivot)
    
    if pivot_row != 0:
        mat[[0, pivot_row],:] = mat[[pivot_row, 0],:]
        perm_mat[[0,pivot_row],:] = perm_mat[[pivot_row,0],:]
    
    mat[0,:] = mat[0,:]*(1/pivot)
    perm_mat[0,:] = perm_mat[0,:]*(1/pivot)

    for i in range(1,m):
        
        mult = mat[i,0]
        if abs(mult) <= 1e-6:
            continue
        mat[i,:] = mat[i,:] - mult*mat[0,:]
        
        perm_mat[i,:] = perm_mat[i,:] - mult*perm_mat[0,:]
        
    further_mat = mat[1:,1:]
    rest, next_perm = unit_up_diag(further_mat)

    mat[1:,1:] = rest


    full_perm = np.eye(m)
    full_perm = np.eye(m)
    full_perm[1:, 1:] = next_perm
    comb_perm = full_perm @ perm_mat

    return mat, comb_perm



def diag_upper(matrix):
    '''
    The input here is a matrix which is unit upper triangular, so we are not rearranging
    Permute columns for this
    '''
    mat = matrix.copy()
    n,m = mat.shape
    for i in range(min(m,n)):
        if mat[i,i] != 0:
            mat




#pairwise_zero = create_mats(opposites_identify_disc, "real")

#print(pairwise_zero[2])
#print(unit_up_diag(pairwise_zero[1]))
#mat1, perm1 = unit_up_diag(pairwise_zero[1])
#mat2, perm2 = unit_up_diag(pairwise_zero[2])

#print("matrix is: \n", mat1)
#print("and the perm to get it is: \n", perm1)
#print("checking: \n", perm1 @ pairwise_zero[1])
#print("fully diag? \n", perm1 @ pairwise_zero[1] @ np.linalg.inv(perm2))


def homology_mats(mat_list):
    '''
    Takes in an ordered list of boundary matrices. FOR NOW computes the diagonalized version of each mat
    If basis vectors are not needed, simply returns betti numbers (not good for general fields)
    '''
    diags = []
    ranks = []
    for mat in mat_list:
        upper, perm = unit_up_diag(mat)
        #diag,perm2 = unit_up_diag(upper.T)
        #_, eigs, _ = np.linalg.svd(mat)

        rank = sum(not np.allclose(row, 0, atol=1e-6) for row in upper)

        #rank = sum(abs(eigs[i])> 1e-6 for i in range(min(mat.shape)))
        #rank = sum(abs(diag[i, i]) > 1e-6 for i in range(min(upper.shape)))
        null = mat.shape[1] - rank
        ranks.append((rank, null))
        
        #count = sum(abs(diag[i, i] - 1) <= 1e-6 for i in range(min(diag.shape)))
        #null = max(diag.shape[0],diag.shape[1]) - count
        #pair = [count,null]
        #ranks.append(pair)
        #print("count is", count, "nullity is", null, "and the eigenvalues are", eigens)
    betti = []
    #for i in range(len(ranks)-2):
    #    betti.append(ranks[i][1]-ranks[i+1][0])
    #betti.append(ranks[-1][1]-ranks[-2][0])
    for i in range(len(ranks) - 1):
        null_i = ranks[i][1]
        rank_ip1 = ranks[i+1][0]
        betti.append(int(null_i - rank_ip1))
    return betti




#pairwise_zero = create_mats(X, "real")

#print(homology_mats(pairwise_zero))










###################################################
############# SIMPLICES FROM DATA #################
###################################################

def vr(data, eps):
    '''Input data is in the form of a matrix where each row is a data point
    and each column is a feature
    We therefore have n data points and m parameters'''
    #sorted_keys = ["P1"]
    sorted_keys = [[1]]
    n,m = data.shape
    char = "P"
    num = 1
    for i in range(n-1):
        #if char != "Z":
        #    char = chr(ord(char)+1)
        #else:
        #    char = "A"
        num += 1
        #key = char + str(num)
        #sorted_keys.append(key)
        sorted_keys.append([num])
    info_dict = {}
    for i in range(n):
        info_dict[str(sorted_keys[i])] = data[i,:]
    simp_complex = sorted_keys.copy()
    connected = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(i+1,n):
    #         diff = data[i,:] - data[j,:]
    #         d = (sum([abs(x**2) for x in diff]))**(1/2)
    #         if d <= eps:
    #             connected[i,j] = 1
    D = squareform(pdist(data))
    connected = (D <= eps).astype(int)
    np.fill_diagonal(connected, 0)
    add_keys = [sorted_keys[i]+sorted_keys[j] for i in range(n) for j in range(i+1, n) if connected[i,j]]
    for add_key in add_keys:
        simp_complex.append(add_key)

    for i in range(3,n+1):
        plus_one = []
        for face in add_keys:
            #parts = re.split(r'(?=[A-Za-z])', face)
            #vertices = [p for p in parts if p]
            #all_but_first = ""
            #for vertex in vertices[1:]:
                #all_but_first += vertex
            #all_but_first = face[1:]
            for face2 in add_keys:
                #if face2.startswith(all_but_first):
                if face2[:-1] == face[1:]:
                    #parts2 = re.split(r'(?=[A-Za-z])', face2)
                    #vertices2 = [p for p in parts2 if p]
                    #last_edge = vertices[0] + vertices2[-1]
                    last_edge = [face[0], face2[-1]]
                    if last_edge in simp_complex:
                        new_simp = face.copy()
                        new_simp.append(face2[-1])
                        plus_one.append(new_simp)
                        simp_complex.append(new_simp)
        add_keys = plus_one


    return simp_complex



rand_val = np.random.uniform(0,1)
x = np.sin(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
y = np.cos(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
matrix = np.array([x,y])
for i in range(3):
    rand_val = np.random.uniform(0,1)
    x = np.sin(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
    y = np.cos(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
    coord = np.array([x,y])
    matrix = np.vstack((matrix,coord))



def inclusion_vr(data):
    ## First create the dictionary which associates a key to each data point
    sorted_keys = [[1]]
    n,m = data.shape
    
    num = 1
    for i in range(n-1):
        num += 1
        sorted_keys.append([num])
    info_dict = {}
    for i in range(n):
        info_dict[str(sorted_keys[i])] = data[i,:]
    simp_complex = sorted_keys.copy()
    D = squareform(pdist(data))


    i, j = np.triu_indices(D.shape[0], k=1)

    vals = D[i,j]

    order = np.argsort(vals)

    eps_vals = sorted(list(vals))

    sorted_indices = list(zip(i[order]+1, j[order]+1))

    connect_eps = [0]*n

    conn_index = 0
    for add_edge_not in sorted_indices:
        add_edge = [int(add_edge_not[0]),int(add_edge_not[1])]
        simp_complex.append((add_edge))
        connect_eps.append(eps_vals[conn_index])
        
        #print("added edge", add_edge)
        for existing in simp_complex:
            if add_edge[0] in existing:
                others = existing.copy()
                others.remove(add_edge[0])
                if not others:
                    continue
                others.append(add_edge[1])
                # others = set(existing).remove(set([add_edge[0]])).add(set([add_edge[1]]))
                #print(existing)
                if sorted((others)) in simp_complex:
                    new = existing.copy()
                    new.append(add_edge[1])
                    if new not in simp_complex:
                        simp_complex.append(sorted(new))
                        connect_eps.append(eps_vals[conn_index])
                        #print("So complex", simp_complex, "was added")
        conn_index += 1

    temp_dict = {}
    for i in range(n):
        temp_dict[i+1] = []

    for simplex in simp_complex:
        key = len(simplex)
        temp_dict[key].append(simplex)
        


    return temp_dict, simp_complex, connect_eps

    
#print(inclusion_vr(matrix))






#### Computing persistent homology fast

### First we need to compute the boundary matrices for this algorithm:
### simplices are ordered by appearance in the complex, then matrices are constructed with 1s (mod 2) in any row which is a face of the simplex (same as before)
### We will do this in multiple steps


### Step 1: a function which uses the vr function and takes in a data set, returning lists for each simplex length of ordered simplices based on where they appear

# Note: this function contains no explicit information about the position of points in the data set


### Edit: This function needs to also return the full ordered list, with all dimension of simplex ordered

def order_by_vr(data):
    '''
    returns a dictionary of lists of ordered simplices. Keys are the length of the simplices in the list
    '''

    temp_dict = {}

    n = len(data)
    for i in range(n):
        temp_dict[i+1] = []

    dist_mat = squareform(pdist(data))

    min_dist = np.min(dist_mat[np.nonzero(dist_mat)])
    max_dist = dist_mat.max()
    diff = max_dist - min_dist

    all_simps = []
    full_order = []
    for i in range(11):
        eps = min_dist + i * diff/9
        old_and_new = vr(data, eps)
        just_new = [x for x in old_and_new if x not in all_simps]

        all_simps = old_and_new
        full_order += just_new
        for simplex in just_new:
            simp_len = len(simplex)
            temp_dict[simp_len].append(simplex)
    
    
    return temp_dict, full_order


# Step 2: Construct the matrices based on the ordering given. Very similar to mats above


def create_mats_ordered(enumerated, full_order):
    max_n = max(enumerated)

    vertices = len(enumerated[1])
    zero_mat = np.zeros([0,vertices])

    mat_list = [zero_mat]
    j_simps = []
    for k in range(max_n-1):
        i_simp_len = k+1
        j_simp_len = k+2
        i_simps = enumerated[i_simp_len]
        j_simps = enumerated[j_simp_len]
        
        del_mat = np.zeros([len(i_simps),len(j_simps)])
        
        if (del_mat.shape[0] != 0) or (del_mat.shape[1] != 0):
            for i in range(len(i_simps)):
                for j in range(len(j_simps)):
                    del_mat[i,j] = get_coeff_ones(i_simps[i],j_simps[j])

        mat_list.append(del_mat)
    
    end_zeros = np.zeros([len(j_simps),0])
    mat_list.append(end_zeros)

    return mat_list, enumerated, full_order





def get_coeff_ones(small_simp, big_simp):
    edges = []
    for i in range(len(big_simp)):
        edge = big_simp[:i]+big_simp[i+1:]
        edges.append(edge)
    #print(edges)
    for i in range(len(edges)):
        if small_simp == edges[i]:
            
            return 1
            
    return 0


### Step 3: Once we have the matrices, we implement algorithm from CTDA on this list of matrices


def low(col):
    nonzero_ind = np.nonzero(col)[0]
    if len(nonzero_ind) == 0:
        low = -1
    else:
        low = nonzero_ind[-1]
    return low
    

def mat_persistence(mat, simplices_big, simplices_small, full_order):
    global_id = {str(simp): idx for idx, simp in enumerate(full_order)}
    mat = mat.copy()
    n, m = mat.shape
    pair_list = []
    for j in range(m):
        index_list = list(range(m))
        low_fixed = low(mat[:,j])
        while low_fixed != -1:
            done = False
            for jdash in range(j):
                low_dash = low(mat[:,jdash])
                if low_fixed == low_dash:
                    mat[:,j] = np.logical_xor(mat[:,j], mat[:,jdash]).astype(int)
                    low_fixed = low(mat[:,j])
                    done = True
                    break
            if not done:
                break
            
        low_fixed = low(mat[:,j])
        if low_fixed != -1:
            sigma_i = simplices_small[low_fixed]
            overall_index = global_id[str(sigma_i)]
            sigma_j = simplices_big[j]
            overall_big_index = global_id[str(sigma_j)]
            pair = [overall_index, overall_big_index]
            pair_list.append(pair)
    return mat, pair_list


def full_persistence(mat_list, dictionary, full_order):
    '''
    The mat_list CONTAINS the zero matrices, we want to ignore the top one
    The dictionary has  1: list of all 1 simplices
                        2: list of all 2 simplices
                        ...
                        n: list of the n simplex
    full_order has all simplices in the complete relative order
    '''

    global_id = {str(simp): idx for idx, simp in enumerate(full_order)}
    mat_list.pop()
    #mat_list.pop(0)
    d = len(mat_list)
    reduced_mats = []
    D_d = mat_list[-1]
    reduced_d, pairs_d = mat_persistence(D_d, dictionary[d], dictionary[d-1], full_order)
    paired = set([x[0] for x in pairs_d])
    reduced_mats.append(reduced_d)
    
    for i in range(d-2,0,-1):
        pair_list = pairs_d
        D_i = mat_list[i].copy()
        n, m = D_i.shape
        big_simp_list = dictionary[i+1]   #### CHECK FOR INDEXING HERE
        small_simp_list = dictionary[i]
        #print("Rows in D_i:", D_i.shape[1])
        #print("Number of small simplices:", len(big_simp_list))
        for j in range(m):
            simp = big_simp_list[j]
            sigma_j = global_id[str(simp)]
            if sigma_j in paired:
                D_i[:,j] = 0
                continue
            if sigma_j not in paired:
                #print(sigma_j, "has not been paired yet")
                low_fixed = low(D_i[:,j])
                while low_fixed != -1:
                    done = False
                    for jdash in range(j):
                        low_dash = low(D_i[:,jdash])
                        if low_fixed == low_dash:
                            D_i[:,j] = np.logical_xor(D_i[:,j], D_i[:,jdash]).astype(int)
                            low_fixed = low(D_i[:,j])
                            done = True
                            break
                    if not done:
                        break
                low_fixed = low(D_i[:,j])
                if low_fixed != -1:
                    simp2 = small_simp_list[low_fixed]
                    sigma_i = global_id[str(simp2)]
                    pair = [sigma_i, sigma_j]
                    pair_list.append(pair)
                    paired.add(sigma_i)
        #paired = set([x[0] for x in pair_list])
        reduced_mats.append(D_i)

    D1 = reduced_mats[0]
    small_0 = dictionary[1]

    all_small_globals = set()
    for s in small_0:
        all_small_globals.add(global_id[str(s)])
    killed_small = set()
    for simp in paired:
        
        if len(full_order[simp]) == 1:
            killed_small.add(simp)
    
    unpaired = all_small_globals-killed_small

    for v in unpaired:
        pair_list.append([v, None])

    
    return reduced_mats, pair_list
























rand_val = np.random.uniform(0.25,0.75)
x = np.sin(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
y = np.cos(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
u_shape = np.array([x,y])
for i in range(50):
    rand_val = np.random.uniform(0.25,0.75)
    x = np.sin(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.02)
    y = np.cos(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.02)
    coord = np.array([x,y])
    u_shape = np.vstack((u_shape,coord)) 
# for i in range(50):
#     rand_val = np.random.uniform(-1,1)
#     y = 0#np.random.normal(loc=0,scale=0.005)
#     x = rand_val #+ np.random.normal(loc=0,scale=0.005)
#     coord = np.array([x,y])
#     u_shape = np.vstack((u_shape,coord))  




#vr(np.eye(4),1.5)
rand_val = np.random.uniform(0,1)
x = np.sin(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
y = np.cos(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
matrix = np.array([x,y])
for i in range(100):
    rand_val = np.random.uniform(0,1)
    x = np.sin(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
    y = np.cos(2*np.pi*rand_val) + np.random.normal(loc=0,scale=0.05)
    coord = np.array([x,y])
    matrix = np.vstack((matrix,coord))


#print(full_persistence(create_mats_ordered(order_by_vr(matrix))))
# temp_dict, full_order = order_by_vr(matrix)
# mat_list, dictionary, full_order = create_mats_ordered(temp_dict, full_order)
# mats, pairs = full_persistence(mat_list, dictionary, full_order)

# print(pairs)
# print(full_order)
# n = len(mats)
# all_homs = {}
# for i in range(n+1):
#     all_homs[i] = []
# print(all_homs)
# for pair in pairs:
#     #print(pair[1])
#     if not pair[1]:
#         key = len(full_order[pair[0]])-1
#     else:
#         key = len(full_order[pair[1]])-1
#     all_homs[key].append(pair)

# print(all_homs)


def persistent_homology_from_data(matrix, max_hom):
    temp_dict, full_order, connection_lengths = inclusion_vr(matrix)
    mat_list, dictionary, full_order = create_mats_ordered(temp_dict, full_order)
    mats, pairs = full_persistence(mat_list, dictionary, full_order)

    n = len(mats)
    all_homs = {}
    for i in range(n+1):
        all_homs[i] = []
    for pair in pairs:
        if not pair[1]:
            key = len(full_order[pair[0]])-1
        else:
            key = len(full_order[pair[1]])-1
        all_homs[key].append(pair)
    
    
    for key in all_homs:
        if key <= max_hom:
        
            intervals = all_homs[key]
            n = len(intervals)
            for i in range(n):
                start = connection_lengths[intervals[i][0]]
                end_index = intervals[i][1]
                #print(end_index)
                if not end_index:
                    max_end = connection_lengths[-1]#intervals[-1][1]

                    if not max_end:
                        max_end = len(full_order)
                    end = max_end+1
                else:
                    end = connection_lengths[end_index]
                x = [start, end]
                y = [i, i]
                plt.plot(x,y,color = "red")
            
            plt.title(f"Homology group {key}")
        
            plt.show()
    

    return all_homs


#persistent_homology_from_data(matrix, 4)









rand_val = np.random.uniform(-1,1)
x = rand_val
y = abs(x)
v_shape = np.array([x,y])
for i in range(50):
    rand_val = np.random.uniform(-1,1)
    x = rand_val + np.random.normal(loc=0,scale=0.05)
    y = abs(x) + np.random.normal(loc=0,scale=0.05)
    coord = np.array([x,y])
    v_shape = np.vstack((v_shape,coord))



x_vals = matrix[:,0]
y_vals = matrix[:,1]

x_u = u_shape[:,0]
y_u = u_shape[:,1]
plt.scatter(x_u,y_u)
plt.show()

x_v = v_shape[:,0]
y_v = v_shape[:,1]
plt.scatter(x_v,y_v)
plt.show()

def plot_vr(data, complex):
    """
    Takes the matrix of data, and reconstructs the dictionary EXACTLY LIKE FOR CONSTRUCTION of the vr complex
    Also takes the actual complex to plot it
    
    """
    # sorted_keys = ["P1"]
    # n,m = data.shape
    # char = "P"
    # num = 1
    # for i in range(n-1):
    #     #if char != "Z":
    #     #    char = chr(ord(char)+1)
    #     #else:
    #     #    char = "A"
    #     num += 1
    #     key = char + str(num)
    #     sorted_keys.append(key)
    # info_dict = {}
    # for i in range(n):
    #     info_dict[sorted_keys[i]] = data[i,:]
    sorted_keys = [[1]]
    n,m = data.shape
    char = "P"
    num = 1
    for i in range(n-1):
        #if char != "Z":
        #    char = chr(ord(char)+1)
        #else:
        #    char = "A"
        num += 1
        #key = char + str(num)
        #sorted_keys.append(key)
        sorted_keys.append([num])
    info_dict = {}
    for i in range(n):
        info_dict[str(sorted_keys[i])] = data[i,:]
    
    plt.scatter(data[:, 0], data[:, 1], color='black')

    for simp in complex:
        if len(simp) == 2:
            #parts = re.split(r'(?=[A-Za-z])', simp)
            x_points = [info_dict[str([p])][0] for p in simp]
            y_points = [info_dict[str([p])][1] for p in simp]
            plt.plot(x_points,y_points, color="red")

    plt.show()



#circle_cloud = vr(matrix,0.1)
#print(circle_cloud)
#print("done with 1")
#print(create_mats(circle_cloud, "real"))
#plot_vr(matrix, circle_cloud)
#print(homology_mats(create_mats(circle_cloud, "real")))



#circle_cloud = vr(matrix,0.2)
#print(circle_cloud)
#print("done with 1")
#print(create_mats(circle_cloud, "real"))
#plot_vr(matrix, circle_cloud)
#print(homology_mats(create_mats(circle_cloud, "real")))


#circle_cloud = vr(matrix,0.3)
#print(circle_cloud)
#print("done with 1")
#print(create_mats(circle_cloud, "real"))
#plot_vr(matrix, circle_cloud)
#print(homology_mats(create_mats(circle_cloud, "real")))


#circle_cloud = vr(matrix,0.5)
#print(circle_cloud)
#print("done with 1")
#print(create_mats(circle_cloud, "real"))
#plot_vr(matrix, circle_cloud)
#print(homology_mats(create_mats(circle_cloud, "real")))


#pairwise_zero = create_mats(circle_cloud, "real")
#print(pairwise_zero)
#print(homology_mats(pairwise_zero))

#circle_cloud = vr(matrix,0.4)
#print((circle_cloud))
#print("done with 2")
#pairwise_zero = create_mats(circle_cloud, "real")
#print(homology_mats(pairwise_zero))
#plot_vr(matrix, circle_cloud)

# circle_cloud = vr(matrix,0.4)
# #print(len(circle_cloud))

# pairwise_zero = create_mats(circle_cloud, "real")
# print(homology_mats(pairwise_zero))
# plot_vr(matrix, circle_cloud)

# circle_cloud = vr(matrix,0.5)
# #print(len(circle_cloud))

# pairwise_zero = create_mats(circle_cloud, "real")
# print(homology_mats(pairwise_zero))
# plot_vr(matrix, circle_cloud)

# circle_cloud = vr(matrix,0.6)
# #print(len(circle_cloud))

# pairwise_zero = create_mats(circle_cloud, "real")
# print(homology_mats(pairwise_zero))
# plot_vr(matrix, circle_cloud)

# circle_cloud = vr(matrix,0.8)
# #print(len(circle_cloud))

# pairwise_zero = create_mats(circle_cloud, "real")
# print(homology_mats(pairwise_zero))
# plot_vr(matrix, circle_cloud)

# circle_cloud = vr(matrix,1)
# #print((circle_cloud))

# pairwise_zero = create_mats(circle_cloud, "real")
# print(homology_mats(pairwise_zero))
# plot_vr(matrix, circle_cloud)

# pairwise_zero = create_mats(circle_cloud, "real")
# print(homology_mats(pairwise_zero))

#pairwise_zero = create_mats(triangle, "real")
#print(pairwise_zero)
#print(homology_mats(pairwise_zero))

#plt.plot(x_vals,y_vals,'o')
#plt.show()






####################################################
############# CURVATURE CALCULATIONS ###############
####################################################
pca = PCA(n_components=1)

def assoc_tangents(points, thresh):

    sample_size = len(points)

    D = squareform(pdist(points))

    graph = [[] for _ in range(sample_size)]

    for i in range(sample_size):
        for j in range(sample_size):
            if i != j and D[i, j] <= thresh:
                graph[i].append(j)
    #connected = (D <= thresh).astype(int)
    direction_dict = {}
    for i in range(sample_size):
        p = points[i]
        #dists = np.linalg.norm(points - points[i], axis=1)
        within_tol = points[D[i] <= thresh]

        shifted = within_tol - within_tol.mean(axis=0)
        pca.fit(shifted)

        v = pca.components_[0]   # unit vector

        dists = D[i].copy()
        dists[i] = np.inf                  # don't choose itself
        j = np.argmin(dists)               # nearest neighbour
        displacement = points[j] - p       # vector toward NN

        if np.dot(v, displacement) < 0:
            v = -v

        direction_dict[tuple(p)] = v

        #### Now we globally order tangents based on some starting point, and using the nearest threshold region in the same direction,
        #### continuing this forward until we have all points oriented for that connected component, and 

    visited = []
    oriented = {}

    for k in range(sample_size):
        if k not in visited: # Start a new chain for each connected component from the graph
            finished = False
            tangent = direction_dict[tuple(points[k])].copy() # Orienting tangent direction
            oriented[tuple(points[k])] = tangent # Add to the dictionary of oriented tangents
            visited.append(k) # Don't wanna add it twice - keep track of where we've been
            to_orient = graph[k] # adjacent vertices
            paired = []
            for j in to_orient:
                paired.append([j,k]) # create pairs where first is to be oriented compared to second
            
            
            while not finished:
                new_pairs = [] # Set of pairs for the next iteration (depth)
                check = False # When we are at max "depth" i.e. no more connected elements to be oriented, this will remain false
                for pair in paired:
                    if pair[0] not in visited: # again, if we have already oriented, no orientation needed
                        check = True
                        comp_tangent = direction_dict[tuple(points[pair[0]])].copy()
                        if np.dot(oriented[tuple(points[pair[1]])], comp_tangent) < 0:
                            comp_tangent = -comp_tangent
                        oriented[tuple(points[pair[0]])] = comp_tangent
                        visited.append(pair[0])
                    for conns in graph[pair[0]]:
                        if conns not in visited:
                            new_pairs.append([conns,pair[0]])
                paired = new_pairs
                if not check:
                    finished = True
    
    return oriented#direction_dict

data = np.zeros((5,2))
data[0,0] = 1
data[1,0] = 0.2
data[2,1] = 0.3
data[3,0] = 0.5



def plot_tangents(dict):
    '''
    dict is a dictionary where keys are points (as coordinates) as strings and they point to an array corresponding to the tangent vector
    '''
    plt.figure(figsize=(6,6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for entry in dict:
        vector = dict[entry]
        point = np.array(entry)
        ax1.scatter(point[0], point[1], s=20, color='black')
        ax1.quiver(
            point[0], point[1],     # starting point
            vector[0], vector[1],     # direction vector
            #angles='xy',
            #scale_units='xy',
            #scale=1/scale,  # controls arrow length
            color='red',
            width=0.005
        )
        ax2.scatter(vector[0], vector[1], s=20, color='black')

    plt.show()

#plot_tangents(assoc_tangents(u_shape, 0.3))
#pca = PCA(n_components=2)
#pca.fit(matrix)

def Most_Common(lst):
    considered = []
    max_occurances = 0
    for item in lst:
        if item not in considered:
            counter = 0
            for item2 in lst:
                if item2 == item:
                    counter += 1
            if counter > max_occurances:
                max_occurances = counter
                max_item = item
            considered.append(item)
    return max_item



def corner_spotter(matrix):
    """"
    - We start by finding an estimate for the topology, using a VR complex (FOR NOW) at various values of epsilon on the data matrix
    - Next, we construct the tangent field. Note that this currently works in two dimensions
    - The tangent is added as a feature to each data point, and the homology is recomputed
    - Changes in betti number indicate that a corner (in some dimension) is present
    - We currently use epsilon from twice the minimum distance to 1/5 the maximum distance between points
    """
    min_dist = 2*np.min(pdist(matrix))
    max_dist = 0.3*np.max(pdist(matrix))

    inc = (max_dist - min_dist)/5


    betti_seq = []
    for i in range(6):
        eps = min_dist+i*inc
        print("Epsilon value is", eps)
        vr_complex = vr(matrix, eps)
        hom_groups = homology_mats(create_mats(vr_complex, "real"))
        while hom_groups and hom_groups[-1]==0:
            hom_groups.pop()
        betti_seq.append(list(hom_groups))
        print("betti groups are", hom_groups)
    most_common = Most_Common(list(betti_seq))
    print("most common occurance is", most_common)

    
#corner_spotter(u_shape)





# circle_cloud = vr(u_shape,0.1)
# print(homology_mats(create_mats(circle_cloud, "real")))
# plot_vr(u_shape, circle_cloud)
# #plot_tangents(assoc_tangents(v_shape,0.1))

# circle_cloud = vr(u_shape,0.2)
# print("done here")
# print(homology_mats(create_mats(circle_cloud, "real")))
# print("made the mats")
# plot_vr(u_shape, circle_cloud)
# #plot_tangents(assoc_tangents(v_shape,0.2))

# circle_cloud = vr(u_shape,0.3)
# print(homology_mats(create_mats(circle_cloud, "real")))
# plot_vr(u_shape, circle_cloud)
plot_tangents(assoc_tangents(u_shape,0.3))
plot_tangents(assoc_tangents(v_shape,0.3))


# circle_cloud = vr(u_shape,0.4)
# print(homology_mats(create_mats(circle_cloud, "real")))
# plot_vr(u_shape, circle_cloud)
# plot_tangents(assoc_tangents(u_shape,0.4))


# circle_cloud = vr(u_shape,0.5)
# print(homology_mats(create_mats(circle_cloud, "real")))
# plot_vr(u_shape, circle_cloud)
# plot_tangents(assoc_tangents(u_shape,0.5))







#########################################################################
####################### Estimating dimension ############################
#########################################################################

helix = []
x = np.cos(0)
y = np.sin(0)
z = 2*np.cos(0)
q = 2*np.sin(0)
helix = np.array([x,y,z,q])
for i in range(1,400):
    x = np.cos(0.02*i) + np.random.normal(0,0.005)
    y = np.sin(0.02*i) + np.random.normal(0,0.005)
    z = 2*np.cos(0.01*i) + np.random.normal(0,0.0025)
    q = 2*np.sin(0.01*i) + np.random.normal(0,0.0025)

    coord = np.array([x,y,z,q])
    helix = np.vstack((helix,coord))


sphere = []
x = np.sin(0)*np.cos(0)
y = np.sin(0)*np.sin(0)
z = np.cos(0)
sphere = np.array([x,y,z])
for i in range(1,70):
    for j in range(1,70):
        x = np.sin(0.1*i)*np.cos(0.1*j)
        y = np.sin(0.1*i)*np.sin(0.1*j)
        z = np.cos(0.1*i)

        coord = np.array([x,y,z])
        sphere = np.vstack((sphere,coord))

def est_dim(data, min_scale = 0, max_scale = np.inf, min_dim = 0, tol = 0):
    n,m = data.shape
    max_dim = m

    pca = PCA(n_components=m)

    D = squareform(pdist(data))

    min_sep = min([min([sep for sep in D[i] if sep != 0]) for i in range(n)])
    r_min = max(min_scale, min_sep)
    av_sep = np.mean(D)
    r_max = min(max_scale,av_sep+r_min)

    inc = (r_max-r_min)/5

    r_vals = []
    for k in range(6):
        r_vals.append(r_min + k*inc)

    point_dim_dict = {}
    all_dims = []

    for i in range(n):
        point_info = {}
        point = data[i]
        dists = D[i].copy()
        dists[i] = np.inf

        xi_vals_prog = []
        


        for j in range(6):
            r_val = r_min + j*inc
            cons = data[dists <= r_val]
            #print(cons[0])

            vectors = cons - point

            if len(vectors) >= 2:
                pca.fit(vectors)
                #n_vects = len(vectors)
                xi = pca.explained_variance_
                xi_vals_prog.append(xi)

        slopes = []

        for j in range(m):
            y_vals = [xi_val[j] for xi_val in xi_vals_prog]
            n_r = len(xi_vals_prog)
            r_vals = []
            for k in range(n_r):
                r_vals.insert(0,r_max-k*inc)
            slope = estimate_rate(y_vals, r_vals)
            #if slope>=2:
            #print("for eigenvalue", j, ",slope is", slope)
            slopes.append(float(slope))
        #print(slopes)

        j = 0
        prev = 0
        count = 0
        #print(slopes)
        while j<m:

            #print(j)
            slope = slopes[j]
            if slope<2.5:
                count += 1
                prev = slope
                j += 1
            else:    
                split = slope - prev
                j = m+1
        point_info["dim"] = count
        point_info["split"] = split
        point_info["point"] = point

        point_dim_dict[i] = point_info

        all_dims.append(count)

    av_dim = np.mean(all_dims)

    return av_dim, point_dim_dict
        #print(xi_vals_prog)


def estimate_rate(y,r):
    '''
    We assume y=c*r^alpha and try to estimate alpha

    '''

    X = np.log(r)
    Y = np.log(y)

    slope, _, _, _, _ = stats.linregress(X, Y)

    return slope







def dim_graph(data):
    n,m = data.shape
    max_dim = m

    pca = PCA(n_components=m)

    D = squareform(pdist(data))

    min_sep = min([min([sep for sep in D[i] if sep != 0]) for i in range(n)])
    
    min_row = min(D, key=sum)
    min_index = D.sum(axis=1).argmin()

    non_zero = np.where(min_row != 0)[0]
    ordered_indices = non_zero[np.argsort(min_row[non_zero])]

    no = len(ordered_indices)

    points_to_pca = [data[min_index]]

    point_dim_dict = {}
    all_dims = []

    for i in range(n):
        point_info = {}
        point = data[i]
        dists = D[i].copy()
        dists[i] = np.inf

        xi_vals_prog = []
        


    #     for j in range(6):
    #         r_val = r_min + j*inc
    #         cons = data[dists <= r_val]
    #         #print(cons[0])

    #         vectors = cons - point

    #         if len(vectors) >= 2:
    #             pca.fit(vectors)
    #             #n_vects = len(vectors)
    #             xi = pca.explained_variance_
    #             xi_vals_prog.append(xi)

    #     slopes = []

    #     for j in range(m):
    #         y_vals = [xi_val[j] for xi_val in xi_vals_prog]
    #         n_r = len(xi_vals_prog)
    #         r_vals = []
    #         for k in range(n_r):
    #             r_vals.insert(0,r_max-k*inc)
    #         slope = estimate_rate(y_vals, r_vals)
    #         #if slope>=2:
    #         #print("for eigenvalue", j, ",slope is", slope)
    #         slopes.append(float(slope))
    #     #print(slopes)

    #     j = 0
    #     prev = 0
    #     count = 0
    #     #print(slopes)
    #     while j<m:

    #         #print(j)
    #         slope = slopes[j]
    #         if slope<2.5:
    #             count += 1
    #             prev = slope
    #             j += 1
    #         else:    
    #             split = slope - prev
    #             j = m+1
    #     point_info["dim"] = count
    #     point_info["split"] = split
    #     point_info["point"] = point

    #     point_dim_dict[i] = point_info

    #     all_dims.append(count)

    # av_dim = np.mean(all_dims)

    # return av_dim, point_dim_dict

#dim_graph(sphere)


#av, _ = est_dim(sphere)

#print(av)







####################################################################################
############################## Name Topology #######################################
####################################################################################


name_origin = []
with open("all_germanic.txt", encoding="utf-8") as f:
    for x in f:
        x = x.strip()
        if not x or " form of " in x or "Saint" in x or "Variant" in x:
            continue
        

        if " m & f " in x:
            pair_list = x.split(" m & f ")
            name_origin.append(pair_list)
        elif (" m " in x) and (" m and " not in x):
            pair_list = x.split(" m ")
            name_origin.append(pair_list)

phonetic = []

name_got = []
with open("all_got.txt", encoding="utf-8") as g:
    for line in g:
        if type(line) == str:
            #g = g.strip()
            names = line.split()
            if names:
                name = names[0]
                if len(name) > 1:
                    name_got.append(name)


name_got = set(name_got)



bad_parts = ["'", "-", "Butter", ",", "Two", "Lord", "The", "Piss", "Grunt", "Kindly", "Pudding", "Grey", "Shrouded", "Loyal", "House", "First", "Moon", "Jar", "Onyx", "Warg", "Muttering", "Sorcerer", "Likely", "Hairy", "Tyroshi", "Orphan", "Blushing", "Blind", "Drunken", "Squint", "Brute", "Queen", "Violet", "Beard", "Blood", "Black", "Pimply", "Knight", "Smiling", "Turnip", "Bastard", "Demon", "Shepherd", "Nine", "Little", "Hammer", "Great", "Plague", "Silver", "Hunchbacked"]


name_got = [
    name for name in name_got
    if not any(bad in name for bad in bad_parts)
]

#print(name_got)




for pair in name_origin:
    # if "Germanic" in pair[1]:
    #     continue
    # elif "German" in pair[1]:
    #     phon = g2p_de(pair[0])#phonemize(pair[0], language='de', backend="espeak")
    #     #phon = backend.phonemize([pair[0]])[0]
    #     #phon = 0
    #     phonetic.append(phon)
    phon = jellyfish.metaphone(pair[0])
    phonetic.append(phon)

germ_phon = []
germ_names = []

for pair in name_origin:
    if "German" in pair[1]:
        germ_phon.append(jellyfish.metaphone(pair[0]))
        germ_names.append(pair[0])

eng_phon = []
eng_names = []

for pair in name_origin:
    if "English" in pair[1]:
        eng_phon.append(jellyfish.metaphone(pair[0]))
        eng_names.append(pair[0])

got_phon = []
for name in name_got:
    got_phon.append(jellyfish.metaphone(name))

fr_phon = []
fr_names = []

for pair in name_origin:
    if "French" in pair[1]:
        fr_phon.append(jellyfish.metaphone(pair[0]))
        fr_names.append(pair[0])


import gruut

names = ["Alfie", "Aelfwyn", "Arwyn", "Marwyn", "Marvin", "Martin", "Martina", "Marina", "Maria", "Marie", "Maline", "Aline", "Allie"]


def name_dist_mat(names, metaphones):
    n = len(names)
    if len(metaphones) != n:
        return "lists did not match"
    
    dist_mat = np.zeros([n,n])

    for i in range(n):
        for j in range(i,n):
            names_dist = jellyfish.levenshtein_distance(names[i],names[j])
            sound_dist = jellyfish.levenshtein_distance(metaphones[i],metaphones[j])
            dist_mat[i,j] = dist_mat[j,i] = 0.1*names_dist + 0.5*sound_dist

    return dist_mat


just_names = []

for pair in name_origin:
    just_names.append(pair[0])

dist_mat = name_dist_mat(just_names, phonetic)
dist_germ = name_dist_mat(germ_names, germ_phon)
dist_eng = name_dist_mat(eng_names, eng_phon)
dist_fr = name_dist_mat(fr_names, fr_phon)
dist_got = name_dist_mat(name_got, got_phon)


#result = ripser(dist_got, distance_matrix=True)
#result = ripser(matrix, maxdim = 2)

#diagrams = result['dgms']


#plot_diagrams(diagrams, show=True)


#plot_barcodes(diagrams[0], title="H0 Barcodes", show=True)
#plot_barcodes(diagrams[1], title="H1 Barcodes", show=True)
#plot_barcodes(diagrams[2], title="H2 Barcodes", show=True)
#import subprocess
#print(subprocess.getoutput("espeak --version"))
#print(phonetic)
#import os
#print(os.environ["PATH"])





#### TRYING TO DO CIRCLE OF NAMES FOR PRESENTATION #####

name_circular_1 = [
    "Alfie",
    "Aelfwyn",
    "Arwyn",
    #"Marwyn",
    "Marvin",
    "Martin",
    "Martina",
    "Marina",
    "Maria",
    "Marie",
    "Maline",
    "Aline",
    "Allie",
]

name_circular_2 = [
    "Alfie",
    "Aelfwyn",
    "Arwyn",
    #"Marwyn",
    "Marvin",
    "Martin",
    "Martina",
    "Marina",
    "Maria",
    "Marie",
    #"Maline",
    #"Aline",
    "Allie",
]

name_circular_2 = [
    "Isabella",
    "Isabelle",
    "John"
]

def create_dist_mat(word_list):
    n = len(word_list)
    phon_list = []
    for text in word_list:
        for sentence in gruut.sentences(text, lang="en"):
            for word in sentence:
                phon = "".join(word.phonemes)
                phon_list.append(phon)
                #print(word.text, word.phonemes)
    
    dist_mat = np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            names_dist = jellyfish.levenshtein_distance(word_list[i],word_list[j])/max(len(word_list[i]),len(word_list[j]))
            sound_dist = jellyfish.levenshtein_distance(phon_list[i],phon_list[j])/max(len(phon_list[i]),len(phon_list[j]))
            dist_mat[i,j] = dist_mat[j,i] = 0.2*names_dist + 0.8*sound_dist
    return dist_mat


np.set_printoptions(
    threshold=np.inf,   # don't truncate
    linewidth=200,      # allow wider rows
    precision=2,        # decimal precision
    suppress=True       # suppress scientific notation for floats
)

phon_circular_1 = []
n = len(name_circular_2)
i = 0
coords = []
xs = []
ys = []
for name in name_circular_1:
    phon_circular_1.append(jellyfish.metaphone(name))
    corr_x = np.cos(2*np.pi*(i/n))
    corr_y = np.sin(2*np.pi*(i/n))
    coords.append([corr_x,corr_y,name])
    xs.append(corr_x)
    ys.append(corr_y)

    i += 1

circle_try_mat = name_dist_mat(name_circular_2, phon_circular_1)
circle_try_mat = create_dist_mat(name_circular_2)


print(circle_try_mat)
fig, ax = plt.subplots()

ax.scatter(xs,ys)

for i in range(n):
    ax.annotate(name_circular_2[i], (xs[i], ys[i]))
    for j in range(i):
        if circle_try_mat[i,j]<0.28:
            ax.plot([xs[i],xs[j]],[ys[i],ys[j]], color="red", linewidth=15*(0.38-circle_try_mat[i,j]))
ax.axis("off")
plt.show()

result = ripser(circle_try_mat, distance_matrix=True)
result = ripser(circle_try_mat, maxdim = 2, distance_matrix=True)

diagrams = result['dgms']


plot_diagrams(diagrams, show=True)


#plot_barcodes(diagrams[0], title="H0 Barcodes", show=True)
#plot_barcodes(diagrams[1], title="H1 Barcodes", show=True)
#plot_barcodes(diagrams[2], title="H2 Barcodes", show=True)