import numpy as np
import cv2

def main():
    array = np.load("logs/log_zed2_220110161834.npz")
    data = array['data']
    timestamp = array['timestamp']
    
    print(data.shape)
    cv2.imshow("ZED | map", data[1,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
