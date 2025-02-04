import numpy as np
np.random.seed(42)
from typing import List
from tqdm import tqdm
import numpy as np


def doTransportIntegration(points,path,coord_1,coord_2,visited,direction_1,direction_2,normals):
    for i in range(len(path)-1):
        if visited[path[i+1]]:
            continue
        direction_1[path[i+1]]=direction_1[path[i]]
        direction_2[path[i+1]] = np.cross(normals[path[i+1]], direction_1[path[i+1]])
        direction_2[path[i+1]] /= np.linalg.norm(direction_2[path[i+1]])

        direction_1[path[i+1]]=np.cross(direction_2[path[i+1]],normals[path[i+1]])
        
        dr=points[path[i+1]]-points[path[i]]
        dri=dr-normals[path[i]]*np.dot(dr,normals[path[i]])
        dri1=dr-normals[path[i+1]]*np.dot(dr,normals[path[i+1]])
        d1idr=np.dot(direction_1[path[i]],dri)
        d2idr=np.dot(direction_2[path[i]],dri)
        d1i1dr=np.dot(direction_1[path[i+1]],dri1)
        d2i1dr=np.dot(direction_2[path[i+1]],dri1)
        
        dv1=(d1idr+d1i1dr)/2
        dv2=(d2idr+d2i1dr)/2
        coord_1[path[i+1]]=coord_1[path[i]]+dv1
        coord_2[path[i+1]]=coord_2[path[i]]+dv2
        
        visited[path[i+1]]=True
    return

def create_coord_system(mesh,orientation_df,rip_df,scales):
    view_dir=orientation_df.loc[orientation_df['name'] == 'e_n', ['z', 'y', 'x']].values[0]
    view_dir=view_dir/np.linalg.norm(view_dir)
    #print(rip_df)
    center_Line = np.stack(rip_df[['CP_z px', 'CP_y px', 'CP_x px']].values)*scales

    #determine center point with direction
    mid_ind=center_Line.shape[0]//2
    center_ind=mesh.find_closest_point(center_Line[mid_ind,:])
    mesh.compute_normals(point_normals=True,cell_normals=False) 
    if np.dot(mesh.point_normals[center_ind],view_dir)>0:
        mesh.flip_normals()
    center_direction=center_Line[mid_ind+1,:]-center_Line[mid_ind-1,:]
    center_direction=center_direction-mesh.point_normals[center_ind]*np.dot(mesh.point_normals[center_ind],center_direction)
    center_direction=center_direction/np.linalg.norm(center_direction)

    direction_1=np.zeros((mesh.points.shape[0],3),dtype=float)
    direction_2=np.zeros((mesh.points.shape[0],3),dtype=float)

    coord_1=np.zeros((mesh.points.shape[0]),dtype=float)
    coord_2=np.zeros((mesh.points.shape[0]),dtype=float)
    visited=np.zeros((mesh.points.shape[0]),dtype=bool)
  
    visited[center_ind]=1
    direction_1[center_ind]=center_direction
    direction_2[center_ind]=np.cross(mesh.point_normals[center_ind],direction_1[center_ind])
    direction_1[center_ind]=np.cross(direction_2[center_ind],mesh.point_normals[center_ind])


    for i in tqdm(range(mesh.points.shape[0]), desc="Processing mesh points", unit="point"):
        if visited[i]:
            continue
        try:
            path = mesh.geodesic(center_ind, i).point_data['vtkOriginalPointIds']
        except:
            return None
        doTransportIntegration(mesh.points,path,coord_1,coord_2,visited,direction_1,direction_2,mesh.point_normals)
    coord_1=coord_1-np.min(coord_1)
    
    mesh.point_data['coord_1']=coord_1
    mesh.point_data['coord_2']=coord_2
    mesh.point_data['direction_1']=direction_1
    mesh.point_data['direction_2']=direction_2
    return mesh

def calculateCurvatureTensor(mesh):
    cells = mesh.faces.reshape(-1, 4)[:, 1:]  # This line is adjusted for 'PolyData' objects

    curvature_tensors=np.zeros((mesh.n_points,2,2),dtype=float)

    for i in tqdm(range(mesh.n_points), desc="Processing mesh points", unit="point"):
        mask = np.any(cells == i, axis=1)
        cells_including_vertex = cells[mask]
        neighbors = np.unique(cells_including_vertex[cells_including_vertex != i])

        normal1=mesh.point_normals[i]
        point1=mesh.points[i]
        
        nn=len(neighbors)
        curvatures=np.zeros((nn,3),dtype=float)
        directions=np.zeros((nn,3),dtype=float)
        neighbor_points = mesh.points[neighbors]
        neighbor_normals = mesh.point_normals[neighbors]
        dr = neighbor_points - point1
        d = np.linalg.norm(dr, axis=1, keepdims=True) + 1e-8  # Avoid division by zero
        curvatures = (neighbor_normals - normal1) / d
        directions = dr / d


        direction2d=np.zeros((curvatures.shape[0],2),dtype=float)
        direction2d[:,0]=np.dot(directions,mesh.point_data['direction_1'][i])
        direction2d[:,1]=np.dot(directions,mesh.point_data['direction_2'][i])

        curvature2d=np.zeros((curvatures.shape[0],2),dtype=float)
        curvature2d[:,0]=np.dot(curvatures,mesh.point_data['direction_1'][i])
        curvature2d[:,1]=np.dot(curvatures,mesh.point_data['direction_2'][i])

        N = direction2d.shape[0]
        X_mod = np.zeros((2*N, 3))
        for k in range(N):
            x1, x2 = direction2d[k]
            X_mod[2*k] = [x1, x2, 0]  
            X_mod[2*k + 1] = [0, x1, x2]
        Y_flat = curvature2d.flatten()
        try:
            A_flat, _, _, _ = np.linalg.lstsq(X_mod, Y_flat, rcond=None)
        except:
            return None
        curvature_tensors[i] = np.array([[A_flat[0], A_flat[1]],
                                        [A_flat[1], A_flat[2]]])

    
    vals,vec=np.linalg.eigh(curvature_tensors)
    
    mean_curvature=(vals[:,0]+vals[:,1])/2
    gauss_curvature=(vals[:,0]*vals[:,1])

  

    mesh.point_data['curvature_tensor']=curvature_tensors
    mesh.point_data['main_curvature_directions']=vec
    mesh.point_data['main_curvatures']=vals
    mesh.point_data['mean_curvature']=mean_curvature
    mesh.point_data['gauss_curvature']=gauss_curvature

    return 1

