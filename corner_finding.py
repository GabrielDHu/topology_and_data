import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import plotly.graph_objects as go



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
        print(slopes)

        j = 0
        prev = 0
        count = 0
        print(slopes)
        while j<m:
            split = 0
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









helix = []
x = np.cos(0)
y = np.sin(0)
z = 2*np.cos(0)
q = 2*np.sin(0)
helix = np.array([x,y,z,q])
for i in range(1,100):
    x = np.cos(0.1*i) + np.random.normal(0,0.005)
    y = np.sin(0.1*i) + np.random.normal(0,0.005)
    z = 2*np.cos(0.1*i) + np.random.normal(0,0.0025)
    q = 2*np.sin(0.1*i) + np.random.normal(0,0.0025)

    coord = np.array([x,y,z,q])
    helix = np.vstack((helix,coord))


#average_dim, dim_dict = est_dim(helix)

#print(dim_dict)



def est_dim_by_r(data, min_scale = 0, max_scale = np.inf, min_dim = 0):
    """
    This function looks at how dimension estimates of the underlying generating space of a data cloud
    changes as we vary the radius within which we consider points. For each point, radii considered 
    are every distance to another point in the data set, and we use two radius neighbours on either side 
    to estimate growth of nth eigenvalue as a function of growth in radius. Dimension eigenvalues should
    grow proportional to r^2, noise proportional to r^4. (Boundary terms will for now be ignored - only
    one term coming before is likely to give a bad estimate regardless)

    
    - data: Point cloud
    - min_scale: minimum radius within which to consider points
    - max_scale: maximum radius within which to consider points
    - min_dim: minimum dimension estimate
    """
    n,m = data.shape
    max_dim = m


    # Step 1: For each point, build a list of eigenvalues of PCA applied to vectors increasing by the next 
    # nearest vector
    pca = PCA(n_components=m)

    D = squareform(pdist(data))

    total_dict = {}

    for i in range(n):
        point_dict = {}
        pcr_eig_evo = []
        r_evo = []
        centre_point = data[i,:]
        vectors = data - centre_point
        distance_list = (D[i,:])

        min_dist = max(min_scale, 1e-6)
        mask = distance_list > min_dist
        distance_list = list(distance_list[mask])
        vectors = vectors[mask]
        order = np.argsort(distance_list)
        included_vectors = [vectors[order[0],:], vectors[order[1],:], vectors[order[2],:]]
        #r_evo = [distance_list[order[0]], distance_list[order[1]], distance_list[order[2]]]

        # We are going to scale by 1/r^2. This should make actual dimension roughly constant
        # and noise increasing
        for j in range(3,len(order)):
            included_vectors.append(vectors[order[j],:])

            pca.fit(included_vectors)
            xi = pca.explained_variance_
            xi_scaled_byr = xi #* 1/(distance_list[order[j]]**2)
            pcr_eig_evo.append(xi_scaled_byr)
            r_evo.append(distance_list[order[j]])
        
        #print("pcr evo is", pcr_eig_evo, "\n")
        #print("radius evo is", r_evo)
        #print("length of r", len(r_evo), "length of eigs", len(pcr_eig_evo))

        # What we have here: List of progressive eigenvalues of PCA as we progressively increase
        # radius for which we include vectors to do PCA on. 

        # Next, we find the optimal alpha s.t. y=c*r^alpha for each eigenvalue at
        # each radius using two radii below and two radii above. 
        
        # alpha_per_rad: List of vectors, with ith component of jth vector estimating
        # the alpha such that eigenvalue i of pca applied to all vectors within r_evo[j]
        # grows at rate proportional to r^alpha 
        alpha_per_rad = []
        rad_for_alpha = []
        dimension_per_rad = []
        for j in range(3,len(pcr_eig_evo) - 2):
            eig_5 = pcr_eig_evo[j:5+j]
            r_5 = r_evo[j:5+j]
            alphas = []
            dim = 0
            dim_check=True
            for k in range(m):
                y_vals = [xi_val[k] for xi_val in eig_5]
                slope = estimate_rate(y_vals, r_5)
                alphas.append(slope)
                # FOR NOW, we will estimate dimension by just having cutoff when alpha>2.5
                if slope>2.5:
                    dim_check = False
                if dim_check:
                    dim += 1
            dimension_per_rad.append(dim)   
            alpha_per_rad.append(alphas)
            rad_for_alpha.append(r_evo[j+2])
        
        point_dict["point"] = centre_point
        point_dict["dimensions"] = dimension_per_rad
        point_dict["radii"] = rad_for_alpha
        point_dict["all_alphas"] = alpha_per_rad

        total_dict[i] = point_dict

    return total_dict





long_rectangle = [np.random.uniform(0, 200), np.random.uniform(0, 10)]

for i in range(100):
    x_coord = np.random.uniform(0, 200)
    y_coord = np.random.uniform(0, 10)

    coord = np.array([x_coord, y_coord])
    long_rectangle = np.vstack((long_rectangle,coord))




dim_dict = est_dim_by_r(long_rectangle, min_scale=5)







### Now let's plot in 3D
print("Done with function, starting graphing animation")


'''
# collect all radii
all_radii = np.unique(
    np.concatenate([v["radii"] for v in dim_dict.values()])
)

all_radii.sort() # So we can have frame changes whenever radii change
'''

# We want a func. so that given r and dim list, we get height
def height_at_radius(r, radii, dims):
    idx = np.searchsorted(radii, r, side="right") - 1
    if idx < 0:
        return 0.0
    return dims[idx]



'''
# get all the points
points = np.array([v["point"] for v in dim_dict.values()])
entries = list(dim_dict.values())

# create the actual plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# set limits of graph
ax.set_xlim(points[:,0].min(), points[:,0].max())
ax.set_ylim(points[:,1].min(), points[:,1].max())
ax.set_zlim(0, max(np.max(v["dimensions"]) for v in entries))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("dimension")



# A function to update height of bars
def update(frame):
    ax.cla()

    r = all_radii[frame]

    ax.set_xlim(points[:,0].min(), points[:,0].max())
    ax.set_ylim(points[:,1].min(), points[:,1].max())
    ax.set_zlim(0, max(np.max(v["dimensions"]) for v in entries))

    for v in entries:
        x, y = v["point"]
        h = height_at_radius(r, v["radii"], v["dimensions"])

        if h > 0:
            ax.plot([x, x], [y, y], [0, h], linewidth=2)

    ax.set_title(f"r = {r:.3f}")


# Run the plot (and it probably won't work)
ani = FuncAnimation(
    fig,
    update,
    frames=len(all_radii),
    interval=300,
    repeat=False
)

plt.show()
'''


# prepare data for graphing
entries = list(dim_dict.values())

points = np.array([v["point"] for v in entries])
all_radii = np.unique(
    np.concatenate([v["radii"] for v in entries])
)
all_radii.sort()

max_dim = max(np.max(v["dimensions"]) for v in entries)

frames = []

for r in all_radii:
    xs, ys, zs = [], [], []
    
    for v in entries:
        x, y = v["point"]
        h = height_at_radius(r, v["radii"], v["dimensions"])

        if h > 0:
            xs.extend([x, x, None])
            ys.extend([y, y, None])
            zs.extend([0, h, None])

    frames.append(
        go.Frame(
            data=[
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(width=6)
                )
            ],
            name=f"{r:.3f}"
        )
    )

'''
# this is initial frame
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line=dict(width=6)
        )
    ],
    frames=frames
)
'''


fig = go.Figure(
    data=[
        # base points at z = 0
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=np.zeros(len(points)),
            mode="markers",
            marker=dict(
                size=4,
                color="black",
                opacity=0.6
            ),
            name="points"
        ),

        # placeholder for dim=1 bars
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line=dict(width=6, color="red"),
            name="dim = 1"
        ),

        # placeholder for dim=2 bars
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line=dict(width=6, color="blue"),
            name="dim = 2"
        ),
    ],
    frames=frames
)


fig.update_layout(
    scene=dict(
        #xaxis_title="x",
        #yaxis_title="y",
        zaxis_title="dimension",
        zaxis=dict(range=[0, max_dim]),
        xaxis=dict(
            title="x",
            range=[-10, 210]
        ),
        yaxis=dict(
            title="y",
            range=[-100, 110]
        ),
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=400, redraw=True),
                            fromcurrent=True
                        )
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0), mode="immediate")]
                )
            ]
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"{r:.3f}"],
                        dict(mode="immediate", frame=dict(duration=0))
                    ],
                    label=f"{r:.2f}"
                )
                for r in all_radii
            ],
            active=0
        )
    ]
)


fig.write_html("dimension_animation.html")
print("Saved dimension_animation.html")