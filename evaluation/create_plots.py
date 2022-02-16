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
    adr = np.zeros([6,3])  
    std_adr = np.zeros([6,3])                      # 20-60 deg
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
            # setup 2
            elif row[2] == '2':
                adr[int(cams[row[0]]),int(row[1])-12] = float(row[3])
                std_adr[int(cams[row[0]]),int(row[1])-12] = float(row[4])
            elif row[2] == '3':
                rad_mean[int(cams[row[0]]),int(row[1])-7] = float(row[3])
                rad_std[int(cams[row[0]]),int(row[1])-7] = float(row[4])
                sphere_err[int(cams[row[0]]),int(row[1])-7] = float(row[5])
            else: continue

    plt.figure()
    plt.plot([1,2,3,4,5], bias[0,0:5], [1,2,3,4,5], bias[1,0:5], [1,2,3,4,5], bias[2,0:5], [1,2,3,4,5], bias[3,0:5], [1,2,3,4,5], bias[4,0:5], [1,2,3,4,5], bias[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Bias')

    plt.figure()
    plt.plot([1,2,3,4,5], precision[0,0:5], [1,2,3,4,5], precision[1,0:5], [1,2,3,4,5], precision[2,0:5], [1,2,3,4,5], precision[3,0:5], [1,2,3,4,5], precision[4,0:5], [1,2,3,4,5], precision[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Precision')

    plt.figure()
    plt.plot([1,2,3,4,5], nan_ratio[0,0:5], [1,2,3,4,5], nan_ratio[1,0:5], [1,2,3,4,5], nan_ratio[2,0:5], [1,2,3,4,5], nan_ratio[3,0:5], [1,2,3,4,5], nan_ratio[4,0:5], [1,2,3,4,5], nan_ratio[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('NaN-Ratio')

    precision_scaled = np.multiply(precision, nan_ratio) + precision
    plt.figure()
    plt.plot([1,2,3,4,5], precision_scaled[0,0:5], [1,2,3,4,5], precision_scaled[1,0:5], [1,2,3,4,5], precision_scaled[2,0:5], [1,2,3,4,5], precision_scaled[3,0:5], [1,2,3,4,5], precision_scaled[4,0:5], [1,2,3,4,5], precision_scaled[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Scaled Precision')

    plt.figure()
    plt.plot([1,2,3,4,5], ep[0,0:5], [1,2,3,4,5], ep[1,0:5], [1,2,3,4,5], ep[2,0:5], [1,2,3,4,5], ep[3,0:5], [1,2,3,4,5], ep[4,0:5], [1,2,3,4,5], ep[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Edge Precision')

    
    plt.figure()
    plt.plot([1,2,3,4,5], nan_edge_ratio[0,0:5], [1,2,3,4,5], nan_edge_ratio[1,0:5], [1,2,3,4,5], nan_edge_ratio[2,0:5], [1,2,3,4,5], nan_edge_ratio[3,0:5], [1,2,3,4,5], nan_edge_ratio[4,0:5], [1,2,3,4,5], nan_edge_ratio[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('NaN-Edge-Ratio')

    ep_scaled = np.multiply(ep, nan_edge_ratio) + ep
    plt.figure()
    plt.plot([1,2,3,4,5], ep_scaled[0,0:5], [1,2,3,4,5], ep_scaled[1,0:5], [1,2,3,4,5], ep_scaled[2,0:5], [1,2,3,4,5], ep_scaled[3,0:5], [1,2,3,4,5], ep_scaled[4,0:5], [1,2,3,4,5], ep_scaled[5,0:5])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Scaled Edge Precision')

    plt.figure()
    plt.plot([20,40,60], adr[0,0:3], [20,40,60], adr[1,0:3], [20,40,60], adr[2,0:3], [20,40,60], adr[3,0:3], [20,40,60], adr[4,0:3], [20,40,60], adr[5,0:3])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Angle Dependent Reflectivity')

    plt.figure()
    plt.plot([1,2,3], rad_mean[0,0:3], [1,2,3], rad_mean[1,0:3], [1,2,3], rad_mean[2,0:3], [1,2,3], rad_mean[3,0:3], [1,2,3], rad_mean[4,0:3], [1,2,3], rad_mean[5,0:3])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Radius Reconstruction Error Mean')

    plt.figure()
    plt.plot([1,2,3], rad_std[0,0:3], [1,2,3], rad_std[1,0:3], [1,2,3], rad_std[2,0:3], [1,2,3], rad_std[3,0:3], [1,2,3], rad_std[4,0:3], [1,2,3], rad_std[5,0:3])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Radius Reconstruction Error Standard Deviation')

    plt.figure()
    plt.plot([1,2,3], sphere_err[0,0:3], [1,2,3], sphere_err[1,0:3], [1,2,3], sphere_err[2,0:3], [1,2,3], sphere_err[3,0:3], [1,2,3], sphere_err[4,0:3], [1,2,3], sphere_err[5,0:3])
    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Sphere Reconstruction Error')

    # plot low ligh performance scaled compared to bright light
    max_bias = np.nanmax(np.abs([bias[:,6], bias[:,1]]))
    max_precision = np.nanmax(np.abs([precision[:,6], precision[:,1]]))
    max_ep = np.nanmax(np.abs([ep[:,6], ep[:,1]]))
    max_nan_ratio = np.nanmax(np.abs([nan_ratio[:,6], nan_ratio[:,1]]))
    max_nan_edge_ratio = np.nanmax(np.abs([nan_edge_ratio[:,6], nan_edge_ratio[:,1]]))
    max_rad_mean = np.nanmax(np.abs([rad_mean[:,3], rad_mean[:,1]]))
    max_rad_std = np.nanmax(np.abs([rad_std[:,3], rad_std[:,1]]))
    max_sphere_err = np.nanmax(np.abs([sphere_err[:,3], sphere_err[:,1]]))
    low_light_measures_scaled = np.transpose(np.array([np.abs(bias[:,6])/max_bias,np.abs(precision[:,6])/max_precision,np.abs(ep[:,6])/max_ep,np.abs(nan_ratio[:,6])/max_nan_ratio,np.abs(nan_edge_ratio[:,6])/max_nan_edge_ratio,np.abs(rad_mean[:,3])/max_rad_mean,np.abs(rad_std[:,3])/max_rad_std,np.abs(sphere_err[:,3])/max_sphere_err]))
    bright_light_measures_scaled = np.transpose(np.array([np.abs(bias[:,1])/max_bias,np.abs(precision[:,1])/max_precision,np.abs(ep[:,1])/max_ep,np.abs(nan_ratio[:,1])/max_nan_ratio,np.abs(nan_edge_ratio[:,1])/max_nan_edge_ratio,np.abs(rad_mean[:,1])/max_rad_mean,np.abs(rad_std[:,1])/max_rad_std,np.abs(sphere_err[:,1])/max_sphere_err]))
    plt.figure()
    plt.plot(['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], low_light_measures_scaled[0,:],'rs', 
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], low_light_measures_scaled[1,:], 'gs',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], low_light_measures_scaled[2,:], 'bs',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], low_light_measures_scaled[3,:], 'cs',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], low_light_measures_scaled[4,:], 'ms',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], low_light_measures_scaled[5,:], 'ys',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], bright_light_measures_scaled[0,:],'r^', 
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], bright_light_measures_scaled[1,:], 'g^',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], bright_light_measures_scaled[2,:], 'b^',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], bright_light_measures_scaled[3,:], 'c^',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], bright_light_measures_scaled[4,:], 'm^',
        ['bias','precision','ep','nan ratio','ep nan','rad mean','rad std','sre'], bright_light_measures_scaled[5,:], 'y^'
        )

    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Low Light Perfomance')

    # plot performance on shiny target scaled compared to non-shiny target
    max_bias = np.nanmax(np.abs([bias[:,5], bias[:,1]]))
    max_precision = np.nanmax(np.abs([precision[:,5], precision[:,1]]))
    max_ep = np.nanmax(np.abs([ep[:,5], ep[:,1]]))
    max_nan_ratio = np.nanmax(np.abs([nan_ratio[:,5], nan_ratio[:,1]]))
    max_nan_edge_ratio = np.nanmax(np.abs([nan_edge_ratio[:,5], nan_edge_ratio[:,1]]))
    shiny_measures_scaled = np.transpose(np.array([np.abs(bias[:,5])/max_bias,np.abs(precision[:,5])/max_precision,np.abs(ep[:,5])/max_ep,np.abs(nan_ratio[:,5])/max_nan_ratio,np.abs(nan_edge_ratio[:,5])/max_nan_edge_ratio]))
    standard_measures_scaled = np.transpose(np.array([np.abs(bias[:,1])/max_bias,np.abs(precision[:,1])/max_precision,np.abs(ep[:,1])/max_ep,np.abs(nan_ratio[:,1])/max_nan_ratio,np.abs(nan_edge_ratio[:,1])/max_nan_edge_ratio]))
    plt.figure()
    plt.plot(['bias','precision','ep','nan ratio','ep nan'], shiny_measures_scaled[0,:],'rs', 
        ['bias','precision','ep','nan ratio','ep nan'], shiny_measures_scaled[1,:], 'gs',
        ['bias','precision','ep','nan ratio','ep nan'], shiny_measures_scaled[2,:], 'bs',
        ['bias','precision','ep','nan ratio','ep nan'], shiny_measures_scaled[3,:], 'cs',
        ['bias','precision','ep','nan ratio','ep nan'], shiny_measures_scaled[4,:], 'ms',
        ['bias','precision','ep','nan ratio','ep nan'], shiny_measures_scaled[5,:], 'ys',
        ['bias','precision','ep','nan ratio','ep nan'], standard_measures_scaled[0,:],'r^', 
        ['bias','precision','ep','nan ratio','ep nan'], standard_measures_scaled[1,:], 'g^',
        ['bias','precision','ep','nan ratio','ep nan'], standard_measures_scaled[2,:], 'b^',
        ['bias','precision','ep','nan ratio','ep nan'], standard_measures_scaled[3,:], 'c^',
        ['bias','precision','ep','nan ratio','ep nan'], standard_measures_scaled[4,:], 'm^',
        ['bias','precision','ep','nan ratio','ep nan'], standard_measures_scaled[5,:], 'y^'
        )

    plt.legend(['Oak-D', 'Oak-D Pro', 'Astra Stereo', 'D435', 'D455', 'ZED2'])
    plt.title('Shiny Performance')


    plt.show()

    return

if __name__ == "__main__":
    main()
