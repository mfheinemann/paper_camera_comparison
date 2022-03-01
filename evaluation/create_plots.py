from unicodedata import name
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import csv


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV", ".csv")])

    # camera index
    cams = {
        'oak-d':0,
        'oak-d-pro':1,
        'orbbec':2,
        'rsd435':3,
        'rsd455':4,
        'zed2':5
    }

    bias = np.zeros([6,7])
    precision = np.zeros([6,7])
    nan_ratio = np.zeros([6,7])
    ep = np.zeros([6,7])    # 1-5 meters + dark + shiny
    nan_edge_ratio = np.zeros([6,7])
    #quad_err = np.zeros([6,7])
    #adr = np.zeros([6,3])  
    adp = np.zeros([6,3])                      # 20-60 deg
    rad_mean = np.zeros([6,4])
    rad_std = np.zeros([6,4])
    sphere_err = np.zeros([6,4])     # 1-3 meters + dark


    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #sort results
        for row in reader:
            #skip first row
            if row[0] == 'camera': continue
            # setup 1
            elif row[2] == '1':
                if row[1] == '11': row[1] = '7'
                bias[int(cams[row[0]]),int(row[1])-1] = float(row[3])
                precision[int(cams[row[0]]),int(row[1])-1] = float(row[4])
                nan_ratio[int(cams[row[0]]),int(row[1])-1] = float(row[5])
                r6 = row[6].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('[ ', '').replace(' ]', '').replace('[', '').replace(']', '')
                ep[int(cams[row[0]]),int(row[1])-1] = np.mean([float(i) for i in r6.split(' ')])
                nan_edge_ratio[int(cams[row[0]]),int(row[1])-1] = float(row[7])
                #quad_err[int(cams[row[0]]),int(row[1])-1] = float(row[8])
            # setup 2
            elif row[2] == '2':
                adp[int(cams[row[0]]),int(row[1])-12] = float(row[3])/1000
                #std_adr[int(cams[row[0]]),int(row[1])-12] = float(row[4])
            elif row[2] == '3':
                rad_mean[int(cams[row[0]]),int(row[1])-7] = float(row[3])
                rad_std[int(cams[row[0]]),int(row[1])-7] = float(row[4])
                sphere_err[int(cams[row[0]]),int(row[1])-7] = float(row[5])
            else: continue

    plt.figure()
    plt.plot([1,2,3,4,5], bias[0,0:5], 'x-', [1,2,3,4,5], bias[1,0:5], 'x-', [1,2,3,4,5], bias[2,0:5], 'x-', [1,2,3,4,5], bias[3,0:5], 'x-', [1,2,3,4,5], bias[4,0:5], 'x-', [1,2,3,4,5], bias[5,0:5], 'x-')
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('distance [m]')
    plt.ylabel('bias [m]')

    plt.figure()
    plt.plot([1,2,3,4,5], precision[0,0:5], 'x-', [1,2,3,4,5], precision[1,0:5], 'x-', [1,2,3,4,5], precision[2,0:5], 'x-', [1,2,3,4,5], precision[3,0:5], 'x-', [1,2,3,4,5], precision[4,0:5], 'x-', [1,2,3,4,5], precision[5,0:5], 'x-')
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('distance [m]')
    plt.ylabel('precision [m]')

    plt.figure()
    plt.plot([1,2,3,4,5], nan_ratio[0,0:5], 'x-', [1,2,3,4,5], nan_ratio[1,0:5], 'x-', [1,2,3,4,5], nan_ratio[2,0:5], 'x-', [1,2,3,4,5], nan_ratio[3,0:5], 'x-', [1,2,3,4,5], nan_ratio[4,0:5], 'x-', [1,2,3,4,5], nan_ratio[5,0:5], 'x-')
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('distance [m]')
    plt.ylabel('nan-ratio')

    # precision_scaled = np.multiply(precision, nan_ratio) + precision
    # plt.figure()
    # plt.plot([1,2,3,4,5], precision_scaled[0,0:5], [1,2,3,4,5], precision_scaled[1,0:5], [1,2,3,4,5], precision_scaled[2,0:5], [1,2,3,4,5], precision_scaled[3,0:5], [1,2,3,4,5], precision_scaled[4,0:5], [1,2,3,4,5], precision_scaled[5,0:5])
    # plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    # plt.xlabel('distance [m]')
    # plt.ylabel('nan-ratio')

    plt.figure()
    plt.plot([1,2,3,4,5], ep[0,0:5], 'x-', [1,2,3,4,5], ep[1,0:5], 'x-', [1,2,3,4,5], ep[2,0:5], 'x-', [1,2,3,4,5], ep[3,0:5], 'x-', [1,2,3,4,5], ep[4,0:5], 'x-', [1,2,3,4,5], ep[5,0:5], 'x-')
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('distance [m]')
    plt.ylabel('edge precision [m]')

    
    # plt.figure()
    # plt.plot([1,2,3,4,5], nan_edge_ratio[0,0:5], [1,2,3,4,5], nan_edge_ratio[1,0:5], [1,2,3,4,5], nan_edge_ratio[2,0:5], [1,2,3,4,5], nan_edge_ratio[3,0:5], [1,2,3,4,5], nan_edge_ratio[4,0:5], [1,2,3,4,5], nan_edge_ratio[5,0:5])
    # plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    # plt.title('NaN-Edge-Ratio')

    # ep_scaled = np.multiply(ep, nan_edge_ratio) + ep
    # plt.figure()
    # plt.plot([1,2,3,4,5], ep_scaled[0,0:5], [1,2,3,4,5], ep_scaled[1,0:5], [1,2,3,4,5], ep_scaled[2,0:5], [1,2,3,4,5], ep_scaled[3,0:5], [1,2,3,4,5], ep_scaled[4,0:5], [1,2,3,4,5], ep_scaled[5,0:5])
    # plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    # plt.title('Scaled Edge Precision')

    # plt.figure()
    # plt.plot([1,2,3,4,5], quad_err[0,0:5], [1,2,3,4,5], quad_err[1,0:5], [1,2,3,4,5], quad_err[2,0:5], [1,2,3,4,5], quad_err[3,0:5], [1,2,3,4,5], quad_err[4,0:5], [1,2,3,4,5], quad_err[5,0:5])
    # plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    # plt.title('Quad Reconstruction Error')

    angles = [0, 20, 40, 60]
    plt.figure()
    data = np.hstack([precision[:,1].reshape(6,1) ,adp[:,0:3]]).T
    plt.plot(angles, data, 'x-')
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('angle [°]')
    plt.ylabel('angle dependent precision [m]')
    plt.ylim([-0.005, 0.4])
    plt.title('Angle Dependent Precision')

    # plt.figure()
    # plt.plot([1,2,3], rad_mean[0,0:3], [1,2,3], rad_mean[1,0:3], [1,2,3], rad_mean[2,0:3], [1,2,3], rad_mean[3,0:3], [1,2,3], rad_mean[4,0:3], [1,2,3], rad_mean[5,0:3])
    # plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    # plt.title('Radius Reconstruction Error Mean')

    plt.figure()
    plt.plot([1,2,3], sphere_err[2,0:3]/1000, 'x-', [1,2,3], sphere_err[3,0:3]/1000, 'x-', [1,2,3], sphere_err[4,0:3]/1000, 'x-', [1,2,3], sphere_err[5,0:3]/1000, 'x-')
    plt.legend(['Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('distance [m]')
    plt.ylabel('sphere radius mean error [m]')
    # plt.title('Sphere Reconstruction Error')

    plt.figure()
    plt.plot([1,2,3], rad_std[2,0:3], 'x-', [1,2,3], rad_std[3,0:3], 'x-', [1,2,3], rad_std[4,0:3], 'x-', [1,2,3], rad_std[5,0:3], 'x-')
    plt.legend(['Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.xlabel('distance [m]')
    plt.ylabel('sphere radius standard deviation [m]')
    # plt.title('Sphere Reconstruction Error Standard Deviation')

    reference_data = np.array([bias[:,1], precision[:,1], sphere_err[:,1], rad_std[:,1]])
    low_light_data = np.array([bias[:,5], precision[:,5], sphere_err[:,3], rad_std[:,3]])
    low_light_increase = np.array(((reference_data-low_light_data)/reference_data)*100)
    low_light_increase[2:4, 2] = np.nan # sphere values for orbbec are zero but should be nan 

    
    # plt.figure()
    # plt.plot(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], low_light_increase[0,:],'rs', 
    #     ['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], low_light_increase[1,:], 'gs',
    #     ['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], low_light_increase[2,:],'bs', 
    #     ['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], low_light_increase[3,:], 'cs',
    #     )

    # plt.legend(['bias', 'precision', 'sphere mean', 'sphere std'])
    # plt.ylabel('low light performance [%]')
    # # plt.title('Low Light Perfomance')
    # plt.ylim([-1100, 150])


    reference_data = np.array([bias[:,1], precision[:,1], nan_ratio[:,1], ep[:,1]])
    shiny_data = np.array([bias[:,6], precision[:,6], nan_ratio[:,6], ep[:,6]])
    shiny_data_increase = np.array(((reference_data-shiny_data)/reference_data)*100)
    # low_light_increase[2:4, 2] = np.nan # sphere values for orbbec are zero but should be nan 

    
    # plt.figure()
    # plt.plot(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], shiny_data_increase[0,:],'rs', 
    #     ['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], shiny_data_increase[1,:], 'gs',
    #     # ['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], shiny_data_increase[2,:],'bs', 
    #     # ['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'], shiny_data_increase[3,:], 'cs',
    #     )
    # plt.legend(['bias', 'precision'])
    # plt.ylabel('reflection performance [%]')
    # # plt.title('Low Light Perfomance')
    # plt.ylim([-1100, 150])


    plt.show()

    return

if __name__ == "__main__":
    main()
