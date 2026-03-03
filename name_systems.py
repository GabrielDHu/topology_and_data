import gruut
import jellyfish
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import pandas as pd
#import gudhi as gd
import dionysus as d

#from openpyxl import load_workbook

from phonemizer import phonemize
import panphon
import panphon.distance as pfd






def ipa(word, lang="en-gb-x-rp"):
    return phonemize(
        word,
        language=lang,
        backend="espeak",
        strip=True
    )



dst = pfd.Distance()

def feature_distance(w1, w2, lang="en-gb-x-rp"):
    p1 = ipa(w1, lang)
    p2 = ipa(w2, lang)
    return dst.weighted_feature_edit_distance(p1, p2)




print(feature_distance("Olivia", "Ahmed"))
print(feature_distance("Amy", "Amy"))
print(feature_distance("Amy", "Bob"))
print(feature_distance("Olivia", "Oliver"))


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



def create_dist_mat_2(word_list):
    n = len(word_list)
    #phon_list = []
    #for text in word_list:
    #    for sentence in gruut.sentences(text, lang="en"):
    #        for word in sentence:
    #            phon = "".join(word.phonemes)
    #            phon_list.append(phon)
    #            #print(word.text, word.phonemes)
    ipa_list = [ipa(name) for name in word_list]
    dist_mat = np.zeros([n,n])

    for i in range(n):
        for j in range(i,n):
            names_dist = jellyfish.levenshtein_distance(word_list[i],word_list[j])/max(len(word_list[i]),len(word_list[j]))
            #sound_dist = jellyfish.levenshtein_distance(phon_list[i],phon_list[j])/max(len(phon_list[i]),len(phon_list[j]))
            #dist_mat[i,j] = dist_mat[j,i] = feature_distance(word_list[i], word_list[j])
            d = dst.weighted_feature_edit_distance(
                ipa_list[i],
                ipa_list[j]
            )
            dist_mat[i, j] = dist_mat[j, i] = d + 5 * names_dist
    return dist_mat





### Access the names and the counts into a data frame
df_girls_unprocessed = pd.read_excel("girlsnames2024.xlsx", sheet_name="Table_6")
df_boys_unprocessed = pd.read_excel("boysnames2024.xlsx", sheet_name="Table_6")

# Get rid of empty columns (they just have descriptive text)
df_girls = df_girls_unprocessed.dropna()
df_boys = df_boys_unprocessed.dropna()

# Reformat tables to have the correct headers and only names and counts
df_girls.columns = df_girls.iloc[0]
df_girls = df_girls.iloc[1:]
df_girls = df_girls.drop(df_girls.columns[0], axis=1)
df_boys.columns = df_boys.iloc[0]
df_boys = df_boys.iloc[1:]
df_boys = df_boys.drop(df_boys.columns[0], axis=1)



df_combined = pd.concat([df_girls, df_boys], ignore_index=True)
df_combined = df_combined.sort_values(by="Count", ascending=False)


def name_mat_weighted_count(df):
    # Get matrix as before
    names = list(df["Name"])
    counts = list(df["Count"])
    n = len(names)
    unweighted_mat = create_dist_mat_2(names)
    weighted_mat = np.zeros([n,n])
    # Apply weighting function.
    for i in range(n):
        for j in range(n):
            min_count = min(counts[i], counts[j])
            distance = unweighted_mat[i,j]

            scaled = distance + (1-distance) * np.exp(-min_count/20)
            weighted_mat[i,j] = scaled#unweighted_mat[i,j] * scale_factor
    
    return names, weighted_mat, unweighted_mat
    


#print(name_mat_weighted_count(df_combined.head()))
#print(df_combined.head(n=500).tail(n=1))

names, weighted_mat, unweighted_mat = name_mat_weighted_count(df_combined.head(n=10))


print(unweighted_mat[0,1])

'''
rips = gd.RipsComplex(
    distance_matrix=unweighted_mat,
    max_edge_length=1
)

simplex_tree = rips.create_simplex_tree(max_dimension=2)

simplex_tree.compute_persistence()

generators = simplex_tree.flag_persistence_generators()

H1_gen = generators[1]

diag = simplex_tree.persistence()

print(H1_gen[1])
'''




'''
result = ripser(unweighted_mat, maxdim = 2, distance_matrix=True, do_cocycles=True)

diagrams = result['dgms']


plot_diagrams(diagrams, show=True)

cocycles = result["cocycles"]

diagrams_1 = diagrams[1]

cocycles_1 = cocycles[1]

cycle_dict = {}

for i in range(len(diagrams_1)):
    birth, death = diagrams_1[i]
    cycle_lifespan = death-birth



'''



#result = ripser(weighted_mat, maxdim = 2, distance_matrix=True)

#diagrams = result['dgms']


#plot_diagrams(diagrams, show=True)




# These methods do not give representative cycles, so we will use... dionysus!!


# We need to build the VR complex explicitly from the distance matrix.


def build_vr_from_dist_mat(dist, max_dist=np.inf):
    dist = unweighted_mat
    n = dist.shape[0]

    filt = d.Filtration()

    max_dim = 2
    

    for i in range(n):
        filt.append(d.Simplex([i], 0.0))

    for i in range(n):
        for j in range(i+1, n):
            if dist[i, j] <= max_dist:
                filt.append(d.Simplex([i, j], dist[i, j]))

    if max_dim >= 2:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    dmax = max(dist[i,j], dist[i,k], dist[j,k])
                    if dmax <= max_dist:
                        filt.append(d.Simplex([i, j, k], dmax))

    filt.sort()

    return filt

filt = build_vr_from_dist_mat(unweighted_mat)

homology = d.homology_persistence(filt)
dgms = d.init_diagrams(homology, filt)


H1_cycles = []

for pt in dgms[1]:
    birth = pt.birth
    death = pt.death
    lifespan = death - birth

    # This gives the index of the persistence pair
    pair = homology.pair(pt.data)

    # This is the actual cycle (a Chain)
    chain = homology[pair]

    vertex_indices = set()
    cycle_gen = []

    for entry in chain:             # iterate over ChainEntry objects
        simplex_idx = entry.index   # get the index in filtration
        #coeff = entry.coeff         # get the coefficient
        simplex = filt[simplex_idx] # get the actual simplex
        vertices = list(simplex)    # vertices of the simplex
        for vertex in vertices:
            vert_name = df_combined.iloc[vertex]["Name"]
            if vert_name not in cycle_gen:
                cycle_gen.append(vert_name)
        vertex_indices.update(vertices)
    vertex_indices = sorted(vertex_indices)
    #print("Vertex indices in cycle:", vertex_indices)

    H1_cycles.append({
        "birth": round(birth,3),
        "death": round(death,3),
        "lifespan": round(lifespan,3),
        "chain": cycle_gen
    })


H1_cycles.sort(key=lambda x: x["lifespan"], reverse=True)

print(H1_cycles[:8])


# Plot the diagram too

h1_points = dgms[1]  # dimension 1

births = [pt.birth for pt in h1_points]
deaths = [pt.death for pt in h1_points]

plt.figure(figsize=(6,6))
plt.scatter(births, deaths, color='blue', label='H1 points')
plt.plot([0, max(deaths)], [0, max(deaths)], 'k--', label='Diagonal')  # y=x line
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title('H1 Persistence Diagram')
plt.legend()
plt.show()
