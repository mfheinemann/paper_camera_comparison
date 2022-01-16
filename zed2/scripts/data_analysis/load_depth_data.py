import numpy as np
from datetime import datetime
import cv2

def main():
    array = np.load("log.npz")
    data = array['data']
    timestamp = array['timestamp']
    
    date = datetime.fromtimestamp(timestamp[0])
    cv2.imshow("ZED | map at {}".format(date), data[0,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
