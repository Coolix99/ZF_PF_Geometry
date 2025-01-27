import numpy as np
import cv2
import math
from typing import List
import pandas as pd
import networkx as nx
from scipy import ndimage
from bisect import bisect_right
import pyvista as pv
import pymeshfix as mf
from scipy.spatial import cKDTree
from scipy.ndimage import label, binary_fill_holes


def calculate_positions_3d(x_positions, y_positions, z_positions,scales):
    x_diff = np.diff(x_positions)*scales[0]
    y_diff = np.diff(y_positions)*scales[1]
    z_diff = np.diff(z_positions)*scales[2]
    segment_distances = np.sqrt(x_diff*x_diff + y_diff*y_diff+ z_diff*z_diff)
    cumulative_distances = np.concatenate(([0], np.cumsum(segment_distances)))
    return cumulative_distances

def closest_point_on_segment(A, B, P):
    AB = B - A
    AP = P - A
    dot_product = np.sum(AP * AB, axis=1)
    denominator = np.sum(AB * AB, axis=1)
    factor = dot_product / denominator
    # Check if the factor is outside the [0, 1] range
    outside_segment = (factor < 0) | (factor > 1)
    # Set closest point to NaN if outside the segment
    closest_point = np.where(outside_segment[:, np.newaxis], np.inf, A + factor[:, np.newaxis] * AB)
    # Calculate distance between P and closest point X
    distance = np.linalg.norm(closest_point - P, axis=1)
    
    return closest_point, distance, factor

def closest_position_on_path(path,P):
    # Calculate the closest points and distances for each segment
    A_values = path[:-1]  # Starting points of segments
    B_values = path[1:]   # Ending points of segments
    _, distances, factors = closest_point_on_segment(A_values, B_values, P)
    # Find the index of the closest point with the minimal distance
    min_distance_index = np.argmin(distances)
   
    # Return the closest point with the minimal distance
    min_distance = distances[min_distance_index]
    min_factor = factors[min_distance_index]

    tree = cKDTree(path)
    min_distance_point, index_point = tree.query(P, k=1)
    if min_distance_point<min_distance:
        return calculate_positions_3d(path[:index_point+1,0],path[:index_point+1,1],path[:index_point+1,2],np.ones((3)))[-1]
    
    x=calculate_positions_3d(path[:min_distance_index+1,0],path[:min_distance_index+1,1],path[:min_distance_index+1,2],np.ones((3)))[-1]
    x=x+min_factor*np.linalg.norm(A_values[min_distance_index,:]-B_values[min_distance_index,:])

    return x

def define_plane(point1, point2, direction):
    vector = np.array(point2) - np.array(point1)
    normal = np.cross(vector, np.array(direction))
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    return normal, np.array(point1)

def get_Plane(point1, point2, direction_dv,im_3d,centroid_3d):
    plane_normal, plane_point = define_plane(point1, point2, direction_dv)
    vector_to_point = centroid_3d - plane_point
    projection = vector_to_point - np.dot(vector_to_point, plane_normal) * plane_normal
    center_plane_point = plane_point + projection

    size_2d = (100,np.max(im_3d.shape)+50)

    direction_1 = np.array([0, point2[1]-point1[1], point2[2]-point1[2]])  
    dot_product = np.dot(direction_1, plane_normal)
    direction_1 -= dot_product / np.dot(plane_normal, plane_normal) * plane_normal
    direction_1 = direction_1 / np.linalg.norm(direction_1)

    direction_2=np.cross(plane_normal,direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    return center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal

def calculate_relative_positions_3d(x_positions, y_positions, z_positions,scales):
    x_diff = np.diff(x_positions)*scales[0]
    y_diff = np.diff(y_positions)*scales[1]
    z_diff = np.diff(z_positions)*scales[2]
    segment_distances = np.sqrt(x_diff*x_diff + y_diff*y_diff+ z_diff*z_diff)
    cumulative_distances = np.concatenate(([0], np.cumsum(segment_distances)))
    s = cumulative_distances / cumulative_distances[-1]
    return s

def interpolate_path(path,s_values, s_position):
    if s_values.shape[0]!=path.shape[0]:
        raise ValueError("s_values and path not same length")


    if s_position < 0 or s_position > 1:
        raise ValueError("s_position should be between 0 and 1")

    num_points = len(path)
    if num_points < 2:
        raise ValueError("Path should have at least two points")

    # Find the index of the rightmost s value less than or equal to s_position
    index = bisect_right(s_values, s_position)

    if index == 0:
        return index,path[0,0],path[0,1],path[0,2]

    if index == s_values.shape[0]:
        return index,path[-1,0],path[-1,1],path[-1,2]

    # Interpolation factor within the segment
    s_start = s_values[index - 1]
    s_end = s_values[index]
    segment_length = s_end - s_start
    interpolation_factor = (s_position - s_start) / segment_length

    # Interpolate the coordinates of the point
    x1, y1, z1 = path[index - 1]
    x2, y2, z2 = path[index]

    interpolated_x = x1 + (x2 - x1) * interpolation_factor
    interpolated_y = y1 + (y2 - y1) * interpolation_factor
    interpolated_z = z1 + (z2 - z1) * interpolation_factor

    return index,interpolated_x, interpolated_y, interpolated_z

def map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, y_2d, x_2d):
    z_3d = center_plane_point[0] + direction_1[0]*(x_2d-size_2d[1]/2) + direction_2[0]*(y_2d-size_2d[0]/2)
    y_3d = center_plane_point[1] + direction_1[1]*(x_2d-size_2d[1]/2) + direction_2[1]*(y_2d-size_2d[0]/2)
    x_3d = center_plane_point[2] + direction_1[2]*(x_2d-size_2d[1]/2) + direction_2[2]*(y_2d-size_2d[0]/2)

    return z_3d, y_3d, x_3d

def create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d,  image_3d):
    y_2d, x_2d = np.meshgrid(np.arange(size_2d[0]) , 
                             np.arange(size_2d[1]) , indexing='ij')

    z_3d, y_3d, x_3d  = map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, y_2d, x_2d)
    
    z_3d = np.round(z_3d).astype(int)
    y_3d = np.round(y_3d).astype(int)
    x_3d = np.round(x_3d).astype(int)

    # Set positions outside the range to zero
    mask = (z_3d >= 0) & (z_3d < image_3d.shape[0]) & \
           (y_3d >= 0) & (y_3d < image_3d.shape[1]) & \
           (x_3d >= 0) & (x_3d < image_3d.shape[2])

    image_2d = np.zeros(size_2d)
    if not np.any(mask):
        return image_2d  # Return an empty image if no valid points are found
    image_2d[mask] = image_3d[z_3d[mask], y_3d[mask], x_3d[mask]]
    return image_2d

def vertical_center_line(prox_dist_pos,closed_image):
    start_vertical_position = prox_dist_pos
    end_vertical_position = np.max(np.where(np.any(closed_image > 0, axis=0)))

    all_centroid_x=[]
    all_centroid_y=[]
    for x_pos in range(int(start_vertical_position),end_vertical_position):
        horizontal_lines = closed_image[:,x_pos]
        non_zero_indices = np.where(horizontal_lines > 0)[0]
        if non_zero_indices.size == 0:
            continue  # Skip if no non-zero elements are found
        mean_positions = np.mean(non_zero_indices)
        all_centroid_y.append(mean_positions)
        all_centroid_x.append(x_pos)
    
    res=np.array((all_centroid_y,all_centroid_x)).T
    return res

def process_image(im_3d):
    mask = im_3d > 0
    labeled_array, num_features = label(mask)

    if num_features == 0:
        return np.zeros_like(mask)  

    largest_component_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    largest_component = labeled_array == largest_component_label
    filled_component = binary_fill_holes(largest_component)

    return filled_component

def construct_Surface(mask,df,center_line_path_3d,scales):
    scale_3d = np.array(scales)

    point1 = df.loc[df['name'] == 'Proximal_pt', ['z', 'y', 'x']].values[0]/scale_3d
    point2 = df.loc[df['name'] == 'Distal_pt', ['z', 'y', 'x']].values[0]/scale_3d
    direction_dv = df.loc[df['name'] == 'e_n', ['z', 'y', 'x']].values[0]/scale_3d
    direction_dv=direction_dv/np.linalg.norm(direction_dv)
    im_3d=process_image(mask)

    print(im_3d.shape)
    print(center_line_path_3d.shape)
    print(center_line_path_3d)

    import napari
    viewer=napari.Viewer(ndisplay=3)
    viewer.add_labels(im_3d)
    viewer.add_shapes(center_line_path_3d, shape_type='path', edge_color='green', edge_width=2)
    napari.run()

    centroid_3d = ndimage.center_of_mass(im_3d)

    center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal=get_Plane(point1, point2, direction_dv,im_3d,centroid_3d)

    t=calculate_relative_positions_3d(center_line_path_3d[:,0],center_line_path_3d[:,1],center_line_path_3d[:,2],scales)

    all_lines_3d=[]
    all_lines_PD=[]
    all_lines_centerPoint=[]
    all_lines_AP=[]
    direction_1[:]=plane_normal[:]
    rel_position=np.linspace(0,1,100)
    for i in range(rel_position.shape[0]):
        #print(i)
        ind,x,y,z=interpolate_path(center_line_path_3d,t,rel_position[i])
        center_plane_point=np.array((x,y,z))
        if(ind==center_line_path_3d.shape[0]):
            plane_normal=center_line_path_3d[ind-2,:]-center_line_path_3d[ind-1,:]
        else:
            plane_normal=center_line_path_3d[ind-1,:]-center_line_path_3d[ind,:]
        plane_normal=plane_normal/np.linalg.norm(plane_normal)
        direction_2=np.cross(direction_1,plane_normal)
        
        im = create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d, im_3d)

        im = np.array(im, dtype=np.uint8)
        kernel_size = (11, 11) 
        blurred_image = cv2.GaussianBlur(im, kernel_size, 0)
        threshold_value = np.max(blurred_image)/2
        _, binary_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        non_zero_pixels = cv2.findNonZero(closed_image)
        x, y, width, height = cv2.boundingRect(non_zero_pixels)
        if max(width,height)<10:
                continue

        path_data_ver=vertical_center_line(0,closed_image)

        z_3d, y_3d, x_3d=map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, path_data_ver[:,0],path_data_ver[:,1])
        line_path_3d = np.column_stack((z_3d, y_3d, x_3d))
        all_lines_3d.append(line_path_3d)
        all_lines_PD.append(rel_position[i])
        all_lines_centerPoint.append(center_plane_point)

        positions_AP=calculate_positions_3d(z_3d, y_3d, x_3d,scales)
        mid_pos=closest_position_on_path(line_path_3d*scales,center_plane_point*scales)
        #print(mid_pos)
        positions_AP=positions_AP-mid_pos 
        all_lines_AP.append(positions_AP)

    print(all_lines_PD)
    L=calculate_positions_3d(center_line_path_3d[:,0],center_line_path_3d[:,1],center_line_path_3d[:,2],scales)[-1]
    all_lines_PD=np.array(all_lines_PD-all_lines_PD[0])*L
   
    # viewer = napari.Viewer(ndisplay=3)
    # im_layer = viewer.add_labels(im_3d)
    # for i in range(len(all_lines_3d)):
    #     viewer.add_shapes(all_lines_3d[i], shape_type='path', edge_color='green', edge_width=2)
    # napari.run()
      
    data = {'path px': all_lines_3d, 
            'PD_position mum': all_lines_PD,
            'center_point px': all_lines_centerPoint,
            'AP_position mum':all_lines_AP}
    

    df = pd.DataFrame(data)
    # print(df)
    # print(df['AP_position mum'])
    #get_first_element = lambda x: x[0]
    #min_first_element = df['AP_position mum'].apply(get_first_element).min()
    # print("Min of the first elements:", min_first_element)
    #get_last_element = lambda x: x[-1]
    #max_last_element = df['AP_position mum'].apply(get_last_element).max()
    # print("Maximum of the last elements:", max_last_element)
    
    P=df['path px'][0]
    P_mum=P*scales
    P_coord=np.zeros((P.shape[0],2))
    P_coord[:,0]=df['AP_position mum'][0]
    P_coord[:,1]=df['PD_position mum'][0]
    for i in range(1,df.shape[0]):
        p=df['path px'][i]
        P=np.concatenate((P,p),axis=0)
        P_mum=np.concatenate((P_mum,p*scales),axis=0)
        p_coord=np.zeros((p.shape[0],2))
        p_coord[:,0]=df['AP_position mum'][i]
        p_coord[:,1]=df['PD_position mum'][i]
        P_coord=np.concatenate((P_coord,p_coord),axis=0)
    
    cloud = pv.PolyData(P_mum)
    surf = cloud.delaunay_2d(tol=0.01,alpha=20)

    meshfix = mf.MeshFix(surf.triangulate())
    holes = meshfix.extract_holes()
    indices1 = np.arange(len(holes.lines)) % 3 == 1
    indices2 = np.arange(len(holes.lines)) % 3 == 2
    v1=holes.lines[indices1]
    v2=holes.lines[indices2]
    result=np.zeros((v1.shape[0],2),dtype=np.uint)
    result[:,0] = v1
    result[:,1] = v2

    G = nx.Graph()
    G.add_edges_from(result)
    cycles = list(nx.simple_cycles(G))
    small_cycles = [cycle for cycle in cycles if len(cycle) < 20]
    for cycle in small_cycles:
        point_indices = []
        for i in range(holes.points[cycle].shape[0]):
            idx = surf.find_closest_point(holes.points[cycle][i])
            point_indices.append(idx)
        point_indices_np=np.array((len(point_indices),*point_indices),dtype=np.uint)
        surf.faces=np.concatenate((surf.faces,point_indices_np),dtype=int)
    surf=surf.triangulate()

    smooth_surface = surf.smooth(1000, feature_smoothing=False,boundary_smoothing=False)
    smooth_surface = smooth_surface.clean()

    tree = cKDTree(P_mum)
    point_coord=np.zeros((smooth_surface.points.shape[0],2))
    for i in range(smooth_surface.points.shape[0]):
        distances, indices = tree.query(smooth_surface.points[i], k=3)
        if distances[0] == 0:
            point_coord[i, :] = P_coord[indices[0], :]
            continue
        inverse_distances = 1.0 / distances
        interpolated_value = np.sum(P_coord[indices, :] * inverse_distances[:, np.newaxis], axis=0) / np.sum(inverse_distances)
        point_coord[i, :] = interpolated_value
    smooth_surface.point_data['AP mum']=point_coord[:,0]
    smooth_surface.point_data['PD mum']=point_coord[:,1]
    smooth_surface.point_data['Coord px']=smooth_surface.points/scales

    
    df.drop(columns=['path px'], inplace=True)
    df.drop(columns=['AP_position mum'], inplace=True)
    df[['CP_z px', 'CP_y px', 'CP_x px']] = pd.DataFrame(df['center_point px'].tolist(), index=df.index)
    df.drop(columns=['center_point px'], inplace=True)
    # print(df)
    return smooth_surface,df




