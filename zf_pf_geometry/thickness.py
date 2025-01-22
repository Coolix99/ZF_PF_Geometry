import numpy as np
from typing import List
import pyvista as pv
from scipy.ndimage import label, binary_fill_holes
import skimage as ski


def getPx(r0,normal,a,vol_img):
    coord=np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int)

    if np.all(coord >= 0) and np.all(coord < np.array(vol_img.shape)):
        return vol_img[coord[0], coord[1], coord[2]]
    else:
        return 0

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

    # viewer = napari.Viewer(ndisplay=3)
    # faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # surface = (points_px, faces,dist)
    # viewer.add_surface(surface)
    # viewer.add_labels(vol_img)
    # #viewer.add_labels(blur)
    # #viewer.add_points(points_plot)
    # napari.run()

    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="thickness", color='grey', ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="lower_dist", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="diff", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.hist(upper_dist-lower_dist, bins=30, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Values Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    return mesh



if __name__ == "__main__":
    make_Thickness() 

