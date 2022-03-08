**Status Repo:** Released

**Status Paper:** Submitted

## paper_camera_comparison
Code for the paper "A metrological and application related comparison of six consumer grade stereo depth cameras for the use in robotics"

Paper is submitted to IROS 2022 conference: https://iros2022.org/

This repository is citable; DOI: https://doi.org/10.5281/zenodo.6338080

All data recorded is accessible through TUHH Open Research and citeable; DOI: https://doi.org/10.15480/336.4225

### Structure
- this repository has six folders on the top level. Five for the different cameras and one for evaluation
- for each camera the script /scripts/data_analysis/align_target.py was used to stream depth and RGB from the camera and project a shape of the target onto these streams to align the camera
- the scripts /scripts/data_collection/save_point_cloud.py were used to record depth and RGB streams and save them in /logs/
- the depth data was saved into numpy arrays representing the point clouds
- each 'data' array in the recorded ziped numpy files has the dimension [i,j,k,3], with i: number of frames; j: x-resolution; k: y-resolution; 3: x,y,z-components; the saved format is therefore not the usual point cloud format but the last layer [i,:,:,2] represents the depth image
