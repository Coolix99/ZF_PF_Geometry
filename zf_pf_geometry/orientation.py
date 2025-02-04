import numpy as np
import napari
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

from zf_pf_geometry.utils import check_array_type

def extract_coordinate(df, name):
    row = df[df['name'] == name]
    if not row.empty:
        return row['coordinate_mum'].iloc[0]
    else:
        return None

def orientation_session(images: List[np.ndarray], scale: List[float]) -> Tuple[pd.DataFrame, str]:
    viewer = napari.Viewer(ndisplay=3)
    im_layer = None
    for img in images:
        if check_array_type(img) == 'discrete':
            im_layer = viewer.add_labels(img, scale=scale)
        else:
            im_layer = viewer.add_image(img, scale=scale)
    
    last_pos = None
    last_viewer_direction = None
    points = []
    points_data = []
    line_layer = None

    @im_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        nonlocal last_pos, last_viewer_direction
        _, _ = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        last_pos = event.position
        last_viewer_direction = event.view_direction
        logger.debug(f"Position: {event.position}, View Direction: {event.view_direction}, Dims Displayed: {event.dims_displayed}")

    def add_point(viewer, color, points_data_key, n_previous=1):
        nonlocal points, last_pos, points_data, line_layer
        points.append(last_pos)
        line = np.array([points])
        logger.debug(f"Line: {line}")
        logger.debug(f"Camera: {viewer.camera}")
        logger.debug(f"Viewer: {viewer}")
        try:
            line_layer = viewer.add_shapes(line, shape_type='line', edge_color=color, edge_width=2)
        except Exception as e:
            logger.debug(f"Exception caught: {e}")

        for _ in range(n_previous):
            viewer.layers.select_previous()
        points_data.append({'coordinate_mum': np.array(points[0]), 'name': points_data_key[0]})
        points_data.append({'coordinate_mum': np.array(points[1]), 'name': points_data_key[1]})
        points_data.append({'coordinate_mum': last_viewer_direction, 'name': points_data_key[2]})

    def set_first_point(viewer):
        nonlocal points, last_pos
        points = [last_pos]

    @viewer.bind_key('a')
    def first(viewer):
        set_first_point(viewer)

    @viewer.bind_key('b')
    def second(viewer):
        add_point(viewer, 'red', ['Proximal_pt', 'Distal_pt', 'viewer_direction_DV'])

    @viewer.bind_key('c')
    def first2(viewer):
        set_first_point(viewer)

    @viewer.bind_key('d')
    def second2(viewer):
        add_point(viewer, 'green', ['Anterior_pt', 'Posterior_pt', 'viewer_direction_DP'], 2)

    @viewer.bind_key('e')
    def first3(viewer):
        set_first_point(viewer)

    @viewer.bind_key('f')
    def second3(viewer):
        add_point(viewer, 'blue', ['Proximal2_pt', 'Distal2_pt', 'viewer_direction_AP'])

    fin_side = None
    @viewer.bind_key('r')
    def end_right_fin(viewer):
        nonlocal fin_side
        fin_side = 'right'
        viewer.close()

    @viewer.bind_key('l')
    def end_left_fin(viewer):
        nonlocal fin_side
        fin_side = 'left'
        viewer.close()

    @viewer.bind_key('q')
    def end_left_fin(viewer):
        viewer.close()

    napari.run()

    df = pd.DataFrame(points_data)
    
    v2 = extract_coordinate(df, 'Anterior_pt') - extract_coordinate(df, 'Posterior_pt')
    v3 = extract_coordinate(df, 'Proximal2_pt') - extract_coordinate(df, 'Distal2_pt')
    n = np.cross(v3, v2)
    n = n / np.linalg.norm(n)

    v1 = extract_coordinate(df, 'Distal_pt') - extract_coordinate(df, 'Proximal_pt')
    view_1 = extract_coordinate(df, 'viewer_direction_DV')
    v1 = v1 - view_1 * (np.dot(v1, n) / np.dot(view_1, n)) 
    v1 = v1 / np.linalg.norm(v1)

    new_rows = pd.DataFrame({'coordinate_mum': [n,v1], 'name': ['e_n','e_PD']})
    df = pd.concat([df, new_rows], ignore_index=True)
    df[['z', 'y', 'x']] = pd.DataFrame(df['coordinate_mum'].tolist(), index=df.index)
    df.drop(columns=['coordinate_mum'], inplace=True)

    return df, fin_side