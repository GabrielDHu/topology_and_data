import numpy as np

check = [1,2,3,4,5,6,7]

#for item1, item2 in zip(check, check[1:]):
#    print(item1, item2)




## CODE FROM https://orangewire.xyz/mathematics/2022/03/25/simplicial-homology.html


X = ["A", "B", "C", "D", "AB", "AC", "BC", "BD", "CD", "BCD"]

opposites_identify_disc = ["A", "B", "C", "D", "E", "F", "AB", "AC", "AD", "AE", "AF", "BC", "BD", "BE", "BF", "CD", "CE", "CF", "DF", "EF", "ABC", "ABD", "ABE", "ABF", "ACD", "AEF", "BCE", "BDF", "CDF", "CEF"]

torus = ["A","B","C","D","E","F","G","H","I","AB","AC","AD","AE","AG","AH","BC","BD","BF","BH","BI","CE","CF","CG","CI","DE","DF","DG","DI","EF","EH","EI","FG","FH","GH","GI","HI","ABD","ABH","ACE","ACG","ADG","AEH","BCF","BCI","BDF","BHI","CEI","CFG","DEF","DEI","DGI","EFH","FGH","GHI"]

tetra = ["A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD"]

def boundary(complex):
    maxdim = len(max(complex, key=len))
    simplices = [sorted([spx for spx in complex if len(spx)==i]) for i in range(1,maxdim+1)]

    # Iterate over consecutive groups (dim k and k+1)
    bnd = []
    for spx_k, spx_kp1 in zip(simplices, simplices[1:]):
        mtx = []
        for sigma in spx_kp1:
            faces = get_faces(sigma)
            mtx.append([get_coeff(spx, faces) for spx in spx_k])
        bnd.append(np.array(mtx).T)

    return bnd

def get_faces(lst):
    return [lst[:i] + lst[i+1:] for i in range(len(lst))]


def get_coeff(simplex, faces):
    if simplex in faces:
        idx = faces.index(simplex)
        return 1 if idx%2==0 else -1
    else:
        return 0

def kernel(A, tol=1e-5):
    _, s, vh = np.linalg.svd(A)
    singular = np.zeros(vh.shape[0], dtype=float)
    singular[:s.size] = s
    null_space = np.compress(singular <= tol, vh, axis=0)
    return null_space.T

def cokernel(A, tol=1e-5):
    u, s, _ = np.linalg.svd(A)
    singular = np.zeros(u.shape[1], dtype=float)
    singular[:s.size] = s
    return np.compress(singular <= tol, u, axis=1)

def homology(boundary_ops, tol=1e-5):
    # Insert zero maps
    mm = boundary_ops[-1].shape[1]
    nn = boundary_ops[0].shape[0]
    boundary_ops.insert(0, np.ones(shape=(0, nn)))
    boundary_ops.append(np.ones(shape=(mm, 0)))

    H = []
    for del_k, del_kp1 in zip(boundary_ops, boundary_ops[1:]):
        kappa = kernel(del_k, tol)
        # Solve for psi
        psi, _, _, _ = np.linalg.lstsq(kappa, del_kp1, rcond=None)
        # Compute homology
        ksi = cokernel(psi, tol)
        H.append(np.dot(kappa, ksi))

    return H

def betti(H):
    return [basis.shape[1] for basis in H]

bnd = boundary(tetra)
H = homology(bnd)
B = betti(H)

print(B)