import numpy as np
import pyvista as pv
from scipy.ndimage import label, binary_fill_holes
import skimage as ski


def getPx(r0,normal,a,vol_img):
    coord = np.round(r0 + normal * a).astype(int)
    coord = np.clip(coord, 0, np.array(vol_img.shape) - 1)  # Ensure within bounds
    return vol_img[tuple(coord)]


def getIntersections(vol_img,r0,normal,max_value=50,precision=1):
    start=None
    alternating_list = [0] + [j for i in range(1, 11) for j in (i, -i)]
    for a in alternating_list:
        f_a=getPx(r0,normal,a,vol_img)
        if f_a:
            start=a
            break
    if start is None:
        return
                    
    a=start
    b = max_value
    while abs(b - a) > precision:
        midpoint = (a + b) / 2.0
        f_a=getPx(r0,normal,a,vol_img)
        f_mid=getPx(r0,normal,midpoint,vol_img)
        if f_mid == f_a:
            a = midpoint
        else:
            b = midpoint
    best_upper = (a + b) / 2.0

    a = start
    b = -max_value
    while abs(b - a) > precision:
        midpoint = (a + b) / 2.0
        f_a=getPx(r0,normal,a,vol_img)
        f_mid=getPx(r0,normal,midpoint,vol_img)
        if f_mid == f_a:
            a = midpoint
        else:
            b = midpoint
    best_lower = (a + b) / 2.0

    return r0+normal*best_upper, r0+normal*best_lower

def process_image(im_3d):
    mask = im_3d > 0

    labeled_array, num_features = label(mask)

    if num_features == 0:
        return np.zeros_like(mask) 
    
    largest_component_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    largest_component = labeled_array == largest_component_label
    filled_component = binary_fill_holes(largest_component)

    return filled_component

def calculate_Thickness(vol_img,mesh:pv.PolyData,scales):
    vol_img=process_image(vol_img)

    vol_img=ski.filters.gaussian(vol_img, sigma=(2,5,5),truncate=3)>0.5
    points_px=mesh.point_data['Coord px']
    normals_px=mesh.point_normals/scales

    dist=np.zeros(points_px.shape[0],dtype=float)
    for i in range(points_px.shape[0]):
        res=getIntersections(vol_img,points_px[i,:],normals_px[i,:])
        if res is None:
            dist[i]=0
            continue
        upper_point, lower_point=res
        dist[i]=np.linalg.norm((upper_point-lower_point)*scales)
    
    mesh.point_data['thickness']=dist

    return mesh
