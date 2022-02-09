## Constants for evaluation
# Target definition
TARGET_SIZE     = (0.5, 0.5)
REDUCE_TARGET   = 0.02
SPHERE_RADIUS   = 0.139 / 2.0

# Offset of camera sensor to mounting position
OFFSET = {'rs435'   : 0.01, 
          'rs455'   : 0.012,
          'zed2'    : 0.015,
          'orbbec'  : 0.015,
          'oak'     : 0.054,
          'oakpro'  : 0.015,}