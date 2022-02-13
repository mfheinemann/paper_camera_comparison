## definiton of experiments for automated evaluation

CAMERAS = {
    'rsd435',
    'rsd455',
    'zed2',  
    'orbbec',
    'oak-d',   
    'oak-d_pro',
}

SETUPS = {}

SETUPS['1'] = {}

SETUPS['1']['experiments'] = [1,2,3,4,5,6,11]
SETUPS['1']['distances'] = [1,2,3,4,5,2,2]

SETUPS['2'] = {}

SETUPS['2']['experiments'] = [12,13,14]
SETUPS['2']['angles'] = [20,40,60]

SETUPS['3'] = {}

SETUPS['3']['experiments'] = [7,8,9,10]
SETUPS['3']['distances'] = [1,2,3,2]