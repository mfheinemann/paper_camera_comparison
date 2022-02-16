## Constants for evaluation
# Target definition
TARGET_SIZE     = (0.5, 0.5)    # in meter
REDUCE_TARGET   = 0.05          # in meter
SPHERE_RADIUS   = 0.139 / 2.0   # in meter
EDGE_WIDTH      = 50            # in px per meter
DISTANCE_FRAME  = 0.08

# Offset of camera sensor to mounting position
OFFSET = {'rsd435'   : 0.007, 
          'rsd455'   : 0.0075,
          'zed2'    : 0.015,
          'orbbec'  : 0.035,
          'oak-d'     : 0.054,
          'oak-d-pro'  : 0.03,}