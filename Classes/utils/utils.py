import os
import torch
from torch.autograd import Variable
import numpy as np
import math
import inspect
from pdf2image import convert_from_path
import imageio
import shutil


def to_var(x, volatile=False, use_cuda = True):
    if torch.cuda.is_available() and use_cuda:
        if isinstance(x, torch.Tensor):
            x = x.to("cuda:0", non_blocking=True)
        elif isinstance(x, dict):
            for key in x.keys():
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to("cuda:0", non_blocking=True)
    else:
        if isinstance(x, torch.Tensor):
            x = x.cpu()
        elif isinstance(x, dict):
            for key in x.keys():
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].cpu()
    return Variable(x, volatile=volatile) if isinstance(x, torch.Tensor) else x


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def batch_cov_torch(points):
    B, N, D = points.size()
    if N == 1:
        return torch.zeros((B, D, D), dtype=points.dtype, device=points.device)
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)

# Function to create sequences
def create_sequences(tensor, sequence_length):
    num_elements, *element_shape = tensor.shape
    new_shape = [num_elements - sequence_length + 1, sequence_length] + element_shape
    sequences = torch.zeros(new_shape, dtype=tensor.dtype)
    
    for i in range(num_elements - sequence_length + 1):
        sequences[i] = tensor[i:i + sequence_length]
        
    return sequences

def batch_cov_numpy(points):
    B, N, D = points.shape
    if N == 1:
        return np.zeros((B, D, D), dtype=points.dtype)
    mean = np.mean(points, axis=1, keepdims=True)
    diffs = (points - mean).reshape(B * N, D)
    prods = np.einsum('ij,ik->ijk', diffs, diffs).reshape(B, N, D, D)
    bcov = np.sum(prods, axis=1) / (N - 1)
    return bcov  # (B, D, D)

# Print the name of the variable and its value
def print_name_and_value(var, obj = None):

    frame = inspect.currentframe().f_back
    lines, starting_lineno = inspect.getsourcelines(frame)

    # Get the line where print_name_and_value was called
    line = lines[frame.f_lineno - starting_lineno].strip()

    if "print_name_and_value(" in line:
        start = line.index("print_name_and_value(") + len("print_name_and_value(")
        end = line.index(")", start)
        var_name = line[start:end].split(",")[0].strip()

        if 'self.' in var_name:
            var_name = var_name.split('self.')[1]

        print(f"{var_name} = {var}")
    else:
        print("Couldn't find the calling line.")

def return_numpy(variable, cpu=1):
    if isinstance(variable, torch.Tensor):
        if cpu:
            return variable.cpu().data.numpy()
        else:
            return variable.data.numpy()
    elif isinstance(variable, np.ndarray):
        return variable
    else:
        return variable

def return_tensor(variable, use_cuda = True):
    if isinstance(variable, torch.Tensor):
        return to_var(variable, use_cuda = use_cuda)
    elif isinstance(variable, np.ndarray):
        return to_var(torch.tensor(variable).float(), use_cuda = use_cuda)
    else:
        return variable
    
# For Artificial_2D
# def dataloader_to_numpy(dataloader):
#     # Placeholder lists to store the data
#     data_list = []
#     labels_list = []
    
#     # Iterate over the DataLoader
#     for data, labels in dataloader:
#         # Convert tensors to numpy arrays and append to the list
#         data_list.append(data.numpy())
#         labels_list.append(labels.numpy())
    
#     # Concatenate all batches together
#     data_array = np.concatenate(data_list, axis=0)
#     labels_array = np.concatenate(labels_list, axis=0)
    
#     return data_array, labels_array

# For Ray_tracing
def dataloader_to_numpy(dataloader):
    # Create a dictionary to store each tensor
    tensor_dict = {}
    
    # Iterate over the DataLoader
    for batch in dataloader:
        # Each batch is a tuple of tensors
        for i, tensor in enumerate(batch):
            # Convert tensor to numpy array
            np_tensor = tensor.numpy()
            
            # Add to dictionary
            if i not in tensor_dict:
                tensor_dict[i] = []
            tensor_dict[i].append(np_tensor)
    
    # Concatenate the numpy arrays for each tensor
    for i in tensor_dict:
        tensor_dict[i] = np.concatenate(tensor_dict[i], axis=0)
        
    return tensor_dict

def serialize_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def rsqueeze(arr):
    while np.any(np.array(arr.shape) == 1):
        arr = np.squeeze(arr)
    return arr

def get_last_value(nested_structure):
    """Get the last value from a nested structure."""
    while hasattr(nested_structure, '__getitem__'):
        try:
            nested_structure = nested_structure[0]
        except (TypeError, IndexError, KeyError):
            break
    return nested_structure

def get_value_at_layer(nested_structure, layer):
    """Get the value at a specific nested layer."""
    current_layer = 0
    while hasattr(nested_structure, '__getitem__'):
        if current_layer == layer:
            return nested_structure
        try:
            nested_structure = nested_structure[0]
        except (TypeError, IndexError, KeyError):
            break
        current_layer += 1
    return None

def coord2xyz(lat, lon, h):
    a = 6378137.0  # meters
    b = 6356752.314245  # meters

    f = (a - b) / a  # flattening
    lambda_ = math.radians(lat)
    e2 = f * (2 - f)  # Square of eccentricity
    phi = math.radians(lon)
    sin_lambda = math.sin(lambda_)
    cos_lambda = math.cos(lambda_)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    N = a / math.sqrt(1 - e2 * sin_lambda * sin_lambda)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e2) * N) * sin_lambda
    r = [x, y, z]

    return r

def xyz2coord(x, y, z):
    a = 6378137.0  # meters
    b = 6356752.314245  # meters

    f = (a - b) / a  # Flattening
    e2 = f * (2 - f)  # Square of eccentricity

    eps = e2 / (1.0 - e2)

    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * a), (p * b))

    sin_q = math.sin(q)
    cos_q = math.cos(q)

    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q

    phi = math.atan2((z + eps * b * sin_q_3), (p - e2 * a * cos_q_3))
    lam = math.atan2(y, x)

    v = a / math.sqrt(1.0 - e2 * math.sin(phi) * math.sin(phi))
    h = (p / math.cos(phi)) - v

    lat = math.degrees(phi)
    lon = math.degrees(lam)

    pos = [lat, lon, h]

    return pos

def from_latlongh_to_xyz_matrix(latlonhObs, latlonhRef, dim = 3):

    # latlonhObs should be 1/N_BS x 3
    # latlonhRef should be 1/N_BS x 3

    latlonhObs = np.array(latlonhObs).squeeze()
    latlonhRef = np.array(latlonhRef).squeeze()

    if len(latlonhObs.shape) == 1:
        latlonhObs = np.expand_dims(latlonhObs, axis=0).reshape(-1, 3)
    if len(latlonhRef.shape) == 1:
        latlonhRef = np.expand_dims(latlonhRef, axis=0).reshape(-1, 3)

    if len(latlonhObs.shape) == 1:
        latlonhRef = np.tile(latlonhRef, (latlonhObs.shape[0], 1))

    # latlonhObs = latlonhObs.transpose(1,0)
    # latlonhRef = latlonhRef.transpose(1,0)
        
    R = 6371e3  # Earth radius in meters
    
    latObs = np.radians(latlonhObs[:, 0])
    lonObs = np.radians(latlonhObs[:, 1])
    latRef = np.radians(latlonhRef[:, 0])
    lonRef = np.radians(latlonhRef[:, 1])
    
    dLat = latObs - latRef
    dLon = lonObs - lonRef
    dH = latlonhObs[:, 2] - latlonhRef[:, 2]
    
    xEast = dLon * R * np.cos(latRef)
    yNorth = dLat * R
    zUp = dH
    
    return np.column_stack((xEast, yNorth, zUp))

def from_xyz_to_latlonh_matrix(xyzObs, latlonhRef, dim = 3):

    # xyzObs should be 1/N_BS x 3
    # latlonhRef should be 1/N_BS x 3

    xyzObs = np.array(xyzObs).squeeze()
    latlonhRef = np.array(latlonhRef).squeeze()

    if len(xyzObs.shape) == 1:
        xyzObs = np.expand_dims(xyzObs, axis=0).reshape(-1, 3)
    if len(latlonhRef.shape) == 1:
        latlonhRef = np.expand_dims(latlonhRef, axis=0).reshape(-1, 3)

    if len(xyzObs.shape) == 1:
        latlonhRef = np.tile(latlonhRef, (xyzObs.shape[0], 1))

    # xyzObs = xyzObs.transpose(1,0)
    # latlonhRef = latlonhRef.transpose(1,0)
    
    R = 6371e3  # Earth radius in meters
    
    latRef = np.radians(latlonhRef[:, 0])
    lonRef = np.radians(latlonhRef[:, 1])
    
    dLat = xyzObs[:, 1] / R
    dLon = xyzObs[:, 0] / (R * np.cos(latRef))
    h = latlonhRef[:, 2] + xyzObs[:, 2]
    
    lat = np.degrees(latRef + dLat)
    lon = np.degrees(lonRef + dLon)
    
    return np.column_stack((lat, lon, h))

def convert_pdfs_to_images(params, prefix, evaluation_epochs, convert = 1):

    folder_path = params.Figures_dir
    images_path = os.path.join(folder_path, f'{prefix}_images')
    os.makedirs(images_path, exist_ok=True)

    if convert:
        for epoch in evaluation_epochs:
            pdf_path = os.path.join(folder_path, f'{prefix}_{epoch}.pdf')
            jpg_file_name = f'{prefix}_{epoch}.jpg'  # Adjust if your enumeration doesn't start with 0
            existing_jpg_path = os.path.join(folder_path, jpg_file_name)
            destination_jpg_path = os.path.join(images_path, jpg_file_name)

            # Check if JPG file already exists in the folder_path
            if os.path.exists(existing_jpg_path):
                # Copy existing JPG to destination folder (overwrites if exists)
                shutil.copy(existing_jpg_path, destination_jpg_path)
            else:
                # Convert PDF to JPG
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    image.save(os.path.join(images_path, f'{prefix}_{epoch}.jpg'), 'JPEG')

    return images_path

# take_1_every: keep 1 image every take_1_every images
def create_movie(images_path, prefix, output_path, evaluation_epochs, fps = 5, take_1_every = 1, num_max_images = 43):

    num_images = 0
    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for counter, epoch in enumerate(evaluation_epochs):
            # Consider only the first page of the PDF if it contains multiple pages
            image_file = os.path.join(images_path, f'{prefix}_{epoch}.jpg')

            if counter % take_1_every == 0:

                if os.path.exists(image_file):
                    image = imageio.imread(image_file)
                    writer.append_data(image)

                    num_images += 1
                    if num_images >= num_max_images:
                        break
    print(f"Processes {num_images} images\n")

def format_dBm(value, tick_number):
    if value <= 0:
        return ''
    return '-' + str(round(-toDeciBellMilliWatt(value)))

def asCartesian2D(r, theta_phi, M = 64, Ng = 352):
    # M, N antennas. Ng cyclic prefix duration

    #takes list rthetaphi (single coord)
    theta_phi   = 180/64 * theta_phi* np.pi/180 # to radian
    # r = 100/Ng * r
    x = r * np.cos( theta_phi )
    y = r * np.sin( theta_phi )
    return [x,y]  

def toDeciBellMilliWatt(RSSI):
    """Returns RSSI value in dBm, assuming input is mW"""
    if RSSI <= 0:
        RSSI = np.nan
    return 10 * math.log10(RSSI)


def rotate_axis(xyz, az, el, roll, backtransformation=None):
    # Convert angles from degrees to radians
    az = np.deg2rad(az)
    el = np.deg2rad(el)
    roll = np.deg2rad(roll)
    
    # Set default value for backtransformation
    if backtransformation is None:
        backtransformation = 0
    
    # Ensure the input vector is a column vector
    if len(xyz.shape) == 1:
        xyz = xyz.reshape(-1, 1)
    if xyz.shape[1] > xyz.shape[0]:
        xyz = xyz.T
    
    # Define the rotation matrices
    Rz = np.array([[np.cos(az), -np.sin(az), 0], 
                   [np.sin(az), np.cos(az), 0], 
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(el), 0, np.sin(el)], 
                   [0, 1, 0], 
                   [-np.sin(el), 0, np.cos(el)]])
    
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll), -np.sin(roll)], 
                   [0, np.sin(roll), np.cos(roll)]])
    
    # Compute the overall rotation matrix
    R = np.dot(np.dot(Rz, Ry), Rx)
    R = R.T
    
    if backtransformation:
        R = np.linalg.inv(R)
    
    # Rotate the axes (not the vector) by using the transpose of the rotation matrix
    rotated = np.dot(R, xyz)
    
    return rotated