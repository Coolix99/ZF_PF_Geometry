import os
import git
from simple_file_checksum import get_checksum
import pandas as pd
import numpy as np

from zf_pf_geometry.metadata_manager import should_process, write_JSON
from zf_pf_geometry.orientation import orientation_session
from zf_pf_geometry.center_line import center_line_session
from zf_pf_geometry.utils import make_path
from zf_pf_geometry.image_operations import get_Image

def do_orientation(input_dirs,key_0, output_dir):
    """
    Processes orientation from input directories and saves the results to the output directory.
    Parameters:
    input_dirs (list of str): List of input directories containing image data.
    key_0 (str): Key to access specific data in the input metadata.
    output_dir (str): Directory where the output data will be saved.
    Returns:
    None
    The function performs the following steps:
    1. Iterates through subdirectories in the first input directory.
    2. Checks if processing is needed based on metadata.
    3. Loads images from corresponding subdirectories in all input directories.
    4. Processes the images to determine orientation.
    5. Saves the orientation data to a CSV file in the output directory.
    6. Updates metadata with orientation results and saves it as a JSON file.
    """
   
    output_key='orientation'

    input_dir_0=input_dirs[0]
    img_0_folder_list= [item for item in os.listdir(input_dir_0) if os.path.isdir(os.path.join(input_dir_0, item))]
    for data_name in img_0_folder_list:
        print(data_name)
        input_folder_0=os.path.join(input_dir_0,data_name)
        output_folder=os.path.join(output_dir,data_name)
        make_path(output_folder)

        res=should_process([input_folder_0],[key_0],output_folder,output_key,verbose=True)
        if not res:
            continue
        input_data,input_checksum=res

        img_folder_list=[input_folder_0]
        for i in range(len(input_dirs)-1):
            img_folder_list.append(os.path.join(input_dirs[i+1],data_name))
        
        images=[]
        for img_folder in img_folder_list:
            img_list = [item for item in os.listdir(img_folder) if item.endswith('.tif')]
            length = len(img_list)
            if length == 1:
                images.append(get_Image(os.path.join(img_folder,img_list[0])))

        df, fin_side=orientation_session(images,input_data[key_0]['scale'])
        
        file_name='orientation.csv'
        df_csv_file=os.path.join(output_folder,file_name)
        df.to_csv(df_csv_file,index=False)

        print(input_data)
        res_MetaData = input_data[key_0]
        if fin_side is not None:
            res_MetaData['fin_side']=fin_side

        res_MetaData['df file name']=file_name
        res_MetaData['df checksum']=get_checksum(df_csv_file, algorithm="SHA1")
        res_MetaData['input_data_checksum']=input_checksum

        repo = git.Repo('.',search_parent_directories=True)
        sha = repo.head.object.hexsha
        res_MetaData['git hash']=sha
        res_MetaData['git origin url']=repo.remotes.origin.url

        write_JSON(output_folder,output_key,res_MetaData)



def do_center_line(orientation_dir, mask_dir,mask_key,output_dir):
    output_key='center line'

    orientation_folder_list= [item for item in os.listdir(orientation_dir) if os.path.isdir(os.path.join(orientation_dir, item))]
    for data_name in orientation_folder_list:
        print(data_name)
        orientation_folder=os.path.join(orientation_dir,data_name)
        mask_folder=os.path.join(mask_dir,data_name)
        output_folder=os.path.join(output_dir,data_name)
        make_path(output_folder)

        res=should_process([orientation_folder,mask_folder],['orientation',mask_key],output_folder,output_key,verbose=True)

        if not res:
            continue
        input_data,input_checksum=res

        orientation_file=os.path.join(orientation_folder,input_data['orientation']['df file name'])
        orientation_df=pd.read_csv(orientation_file)

        img_list = [item for item in os.listdir(mask_folder) if item.endswith('.tif')]
        length = len(img_list)
        if length == 1:
            mask_image = get_Image(os.path.join(mask_folder, img_list[0]))
        else:
            print("Not one mask image found")
            continue

        center_line_path_3d=center_line_session(mask_image,orientation_df,input_data['orientation']['scale'])

        CenterLine_file_name=data_name+'_CenterLine.npy'
        CenterLine_file=os.path.join(output_folder,CenterLine_file_name)
        np.save(CenterLine_file,center_line_path_3d)

        res_MetaData = input_data['orientation']
        res_MetaData['CenterLine file name']=CenterLine_file_name
        res_MetaData['CenterLine checksum']=get_checksum(CenterLine_file, algorithm="SHA1")
        res_MetaData['input_data_checksum']=input_checksum
        repo = git.Repo('.', search_parent_directories=True)
        sha = repo.head.object.hexsha
        res_MetaData['git hash'] = sha
        res_MetaData['git origin url'] = repo.remotes.origin.url

        write_JSON(output_folder, output_key, res_MetaData)


    return

   