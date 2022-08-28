import PIL
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')




def imToArrshape(filename : str, Nx : int, Ny: int, show = False):
    """Function that takes png image into array of (Nx,Ny,3)
    to define geometries. where there is color becomes 1 and no shape is 0

    Args:
        filename (str): name of png file
        Nx (int): desired size in x direction
        Ny (int): desired size in y direction

    Returns:
        ndarray: boolean array containing your new geometry
    """
    large_image_data = PIL.Image.open(filename)
    large_image_data = large_image_data.resize((Ny,Nx))
    arr_like_im = np.asarray(large_image_data)
    smaller_data = np.zeros((Nx,Ny,3))
    smaller_data[:,:,0] = arr_like_im[:,:,0] + arr_like_im[:,:,1] + arr_like_im[:,:,2] + arr_like_im[:,:,3]
    """
    for i in range(Nx):
        for j in range(Ny):
            smaller_data[i,j,0] = np.average(large_image_data[i*step_x:(step_x*(i+1)),j*step_y:(step_y*(j+1)),0]) + np.average(large_image_data[i*step_x:(step_x*(i+1)),j*step_y:(step_y*(j+1)),1])
    """
    """split_arr = np.hsplit(large_image_data[:,:,0],Nx)
    print(split_arr) """
    smaller_data[:,:,1],smaller_data[:,:,2] = smaller_data[:,:,0],smaller_data[:,:,0]
    smaller_data[smaller_data>0] = 1
    smaller_data = np.nan_to_num(smaller_data)
    if show:
        plt.imshow(smaller_data[:,:,0])
        plt.show()
    return smaller_data





if __name__ == '__main__':
    
    imToArrshape("Structures/latticeSHNO.png", 30,30)