import numpy as np
import cv2

def main():
    array = np.load("test_depth.npz")
    data = array
    # timestamp = array['timestamp']
    
    print(data.shape)
    cv2.imshow("ZED | map", data[1,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
