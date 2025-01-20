import numpy as np
import napari
import pandas as pd
from typing import List,Tuple

from zf_pf_geometry.utils import check_array_type

def extract_coordinate(df, name):
    row = df[df['name'] == name]
    if not row.empty:
        return row['coordinate_mum'].iloc[0]
    else:
        return None

def orient_session(images:List[np.ndarray],scale:List[float],verbose:bool=False) -> Tuple[pd.DataFrame,str]:
    viewer = napari.Viewer(ndisplay=3)
    im_layer=None
    for img in images:
        if check_array_type(img) == 'discrete':
            im_layer = viewer.add_labels(img,scale=scale)
        else:
            im_layer = viewer.add_image(img,scale=scale)
    
    last_pos=None
    last_viewer_direction=None
    points = []
    points_data=None
    line_layer=None

    @im_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        nonlocal last_pos,last_viewer_direction
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        last_pos=event.position
        last_viewer_direction=event.view_direction
        if verbose:
            print(event.position,
                event.view_direction,
                event.dims_displayed)
        
    @viewer.bind_key('a')
    def first(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('b')
    def second(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        if verbose:
            print(line)
            print(viewer.camera)
            print(viewer)
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='red', edge_width=2)
        except:
            if verbose:
                print('catched')

        viewer.layers.select_previous()
        points_data = [
        {'coordinate_mum': np.array(points[0]), 'name': 'Proximal_pt'},
        {'coordinate_mum': np.array(points[1]), 'name': 'Distal_pt'},
        {'coordinate_mum': last_viewer_direction, 'name': 'viewer_direction_DV'}
        ]

    @viewer.bind_key('c')
    def first2(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('d')
    def second2(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='green', edge_width=2)
        except:
            if verbose:
                print('catched')
        viewer.layers.select_previous()
        viewer.layers.select_previous()
        points_data =points_data+ [
        {'coordinate_mum': np.array(points[0]), 'name': 'Anterior_pt'},
        {'coordinate_mum': np.array(points[1]), 'name': 'Posterior_pt'},
        {'coordinate_mum': last_viewer_direction, 'name': 'viewer_direction_DP'}
        ]

    @viewer.bind_key('e')
    def first3(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('f')
    def second3(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='blue', edge_width=2)
        except:
            if verbose:
                print('catched')
        viewer.layers.select_previous()
        viewer.layers.select_previous()
        viewer.layers.select_previous()
        points_data =points_data+ [
        {'coordinate_mum': np.array(points[0]), 'name': 'Proximal2_pt'},
        {'coordinate_mum': np.array(points[1]), 'name': 'Distal2_pt'},
        {'coordinate_mum': last_viewer_direction, 'name': 'viewer_direction_AP'}
        ]

    fin_side=None
    @viewer.bind_key('r')
    def end_right_fin(viewer):
        nonlocal fin_side
        fin_side='right'
        viewer.close()

    @viewer.bind_key('l')
    def end_left_fin(viewer):
        nonlocal fin_side
        fin_side='left'
        viewer.close()

    @viewer.bind_key('q')
    def end_left_fin(viewer):
        viewer.close()

    napari.run()

    df = pd.DataFrame(points_data)
    
    v2=extract_coordinate(df,'Anterior_pt')-extract_coordinate(df,'Posterior_pt')
    v3=extract_coordinate(df,'Proximal2_pt')-extract_coordinate(df,'Distal2_pt')
    n=np.cross(v3,v2)
    n = n / np.linalg.norm(n)

    v1=extract_coordinate(df,'Proximal_pt')-extract_coordinate(df,'Distal_pt')
    view_1=extract_coordinate(df,'viewer_direction_DV')
    v1 = v1 - view_1*(np.dot(v1,n)/np.dot(view_1,n)) 
    v1 = v1 / np.linalg.norm(v1)
    #think about this
    if np.isnan(n).any():
        raise ValueError("The array contains NaN values.")

    new_rows = pd.DataFrame({'coordinate_mum': [n,v1], 'name': ['e_n','e_PD']})
    df = pd.concat([df, new_rows], ignore_index=True)
    
    if verbose:
        print(df)

    return df, fin_side


