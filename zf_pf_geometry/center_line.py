import numpy as np
import cv2
import math
from typing import List
import napari
from scipy.interpolate import UnivariateSpline

        
def define_plane(point1, point2, direction):
    v = np.array(point2) - np.array(point1)
    normal = np.cross(v, np.array(direction))
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    return normal, np.array(point1)

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
    image_2d[mask] = image_3d[z_3d[mask], y_3d[mask], x_3d[mask]]

    return image_2d

def radial_center_line(prox_dist_pos,y_pos,closed_image):
    outside_point = (y_pos,prox_dist_pos)  

    num_angles = 91  # Adjust as needed
    angles = np.linspace(0, math.pi / 2, num=num_angles)[::-1]

    all_centroid_x=[]
    all_centroid_y=[]

    for i, angle in enumerate(angles):
        line_length = max(closed_image.shape) * 1.5  # Adjust as needed
        end_y = outside_point[0] - line_length * math.cos(angle)
        end_x = outside_point[1] - line_length * math.sin(angle)

        # Create an array of coordinates along the line from outside_point to the endpoint
        x_coords = np.linspace(outside_point[1], end_x, num=int(line_length))
        y_coords = np.linspace(outside_point[0], end_y, num=int(line_length))
        line_coords = np.column_stack((y_coords,x_coords)).astype(int)

        # Ensure that coordinates are within image bounds
        valid_coords_mask = np.logical_and.reduce(
            (line_coords[:, 0] >= 0, line_coords[:, 0] < closed_image.shape[0],
            line_coords[:, 1] >= 0, line_coords[:, 1] < closed_image.shape[1]))
        valid_line_coords = line_coords[valid_coords_mask]

        # Find intersections with the binary image
        intersections_mask = closed_image[valid_line_coords[:, 0], valid_line_coords[:, 1]] > 0
        intersections = valid_line_coords[intersections_mask]

        if intersections.shape[0] > 10:
            all_centroid_x.append(np.mean(intersections[:, 1]))
            all_centroid_y.append(np.mean(intersections[:, 0]))
    res=np.array((all_centroid_y,all_centroid_x)).T
    return res

def vertical_center_line(prox_dist_pos,closed_image):
    start_vertical_position = prox_dist_pos
    end_vertical_position = np.max(np.where(np.any(closed_image > 0, axis=0)))

    all_centroid_x=[]
    all_centroid_y=[]
    for x_pos in range(int(start_vertical_position),end_vertical_position):
        horizontal_lines = closed_image[:,x_pos]
        mean_positions = np.mean(np.where(horizontal_lines > 0))
        if math.isnan(mean_positions):
            continue
        all_centroid_y.append(mean_positions)
        all_centroid_x.append(x_pos)
    
    res=np.array((all_centroid_y,all_centroid_x)).T
    return res
 
def get_Plane(point1, point2, direction_dv,im_3d,centroid_3d):

    plane_normal, plane_point = define_plane(point1, point2, direction_dv)
    
    vector_to_point = centroid_3d - plane_point
    projection = vector_to_point - np.dot(vector_to_point, plane_normal) * plane_normal
    center_plane_point = plane_point + projection

    size_2d = (im_3d.shape[0]+20,np.max(im_3d.shape)+200)

    direction_1 = np.array([0, point2[1]-point1[1], point2[2]-point1[2]])  # You can adjust the x and y components as needed
    dot_product = np.dot(direction_1, plane_normal)
    direction_1 -= dot_product / np.dot(plane_normal, plane_normal) * plane_normal
    direction_1 = direction_1 / np.linalg.norm(direction_1)

    direction_2=np.cross(plane_normal,direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    return center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal

def calculate_relative_positions(x_positions, y_positions):
    x_diff = np.diff(x_positions)
    y_diff = np.diff(y_positions)
    segment_distances = np.sqrt(x_diff**2 + y_diff**2)
    
    cumulative_distances = np.concatenate(([0], np.cumsum(segment_distances)))
    
    s = cumulative_distances / cumulative_distances[-1]
    
    return s

def center_line_session(mask_3d,df,scale_3d):
    scale_3d = np.array(scale_3d)

    point1 = df.loc[df['name'] == 'Proximal_pt', ['z', 'y', 'x']].values[0]
    point2 = df.loc[df['name'] == 'Distal_pt', ['z', 'y', 'x']].values[0]
    direction_dv = df.loc[df['name'] == 'e_n', ['z', 'y', 'x']].values[0]
   
    point1 = point1 / scale_3d
    point2 = point2 / scale_3d
    direction_dv = direction_dv / scale_3d

    centroid_3d=np.array(mask_3d.shape,dtype=float)/2

    center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal=get_Plane(point1, point2, direction_dv,mask_3d,centroid_3d)

    im = create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d, mask_3d).astype(int)
    im_mask=im>0

    im_mask = np.array(im_mask, dtype=np.uint8)
    kernel_size = (11, 11) 
    blurred_image = cv2.GaussianBlur(im_mask, kernel_size, 0)
    threshold_value = np.max(blurred_image)/2
    _, binary_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    last_pos=None
    path_layer=None

    non_zero_pixels = cv2.findNonZero(closed_image)
    x, y, width, height = cv2.boundingRect(non_zero_pixels)
    y_pos_px=y+height
    line_data=np.array([[y_pos_px,0],[y_pos_px,im.shape[1]-1]])

    viewer = napari.Viewer(ndisplay=2)
    viewer.add_shapes(data=line_data,shape_type='line',edge_color='red',edge_width=2,scale=scale_3d[[0,1]])

    im_layer = viewer.add_labels(im,scale=scale_3d[[0,1]])

    @im_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        nonlocal last_pos
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        last_pos=event.position
        
    @viewer.bind_key('a')
    def add_point_a(viewer):
        nonlocal last_pos,path_layer
        prox_dist_pos=last_pos[1]
        if last_pos is not None:
            line_data=np.array([[0,prox_dist_pos],[im.shape[0]-1,prox_dist_pos]])/scale_3d[[0,1]]
            viewer.add_shapes(data=line_data,shape_type='line',edge_color='green',edge_width=2,scale=scale_3d[[0,1]])

            path_data_rad=radial_center_line(prox_dist_pos/scale_3d[1],y_pos_px,closed_image)
            path_data_ver=vertical_center_line(prox_dist_pos/scale_3d[1],closed_image)
            path_data=np.concatenate((path_data_rad, path_data_ver), axis=0)
            
            viewer.add_shapes(path_data, shape_type='path', edge_color='red', edge_width=2,scale=scale_3d[[0,1]])
           
            s=calculate_relative_positions(path_data[:,0],path_data[:,1])
            sp0=UnivariateSpline(s, path_data[:,0],k=3, s=200)
            sp1=UnivariateSpline(s, path_data[:,1],k=3, s=200)
            t=np.linspace(0, 1, 100)
            interpolated_data=np.zeros((t.shape[0]+2,2))
            interpolated_data[1:-1,0]=sp0(t)
            interpolated_data[1:-1,1]=sp1(t)
            v=interpolated_data[2,:]-interpolated_data[1,:]
            v=v/np.linalg.norm(v)
            interpolated_data[0,:]=interpolated_data[1,:]-v*100
            v=interpolated_data[-2,:]-interpolated_data[-3,:]
            v=v/np.linalg.norm(v)
            interpolated_data[-1,:]=interpolated_data[-2,:]+v*100

            path_layer=viewer.add_shapes(interpolated_data, shape_type='path', edge_color='green', edge_width=2,scale=scale_3d[[0,1]])

    napari.run()
    try:
        resulting_path = path_layer.data[0]/scale_3d[[0,1]]
    except:
        return None
 
    z_3d, y_3d, x_3d=map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, resulting_path[:,0],resulting_path[:,1])
    center_line_path_3d = np.column_stack((z_3d, y_3d, x_3d))*scale_3d
    return center_line_path_3d


