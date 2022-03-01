import numpy as np
import copy
import open3d as o3d
import csv

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def load_data(path, delimiter = ' '):
    """Loads point cloud data of type `CSV`, `PLY` and `PCD`.

    The file should contain one point per line where each number is separated by the `delimiter` character.
    Any none numeric lines will be skipped.

    Args:
        path (str): The path to the file.
        delimiter (char): Separation of numbers in each line of the file.

    Returns:
        A ndarray of shape NxD where `N` are the number of points in the point cloud and `D` their dimension.
    """
    data = list()
    with open(path, newline='\n') as file:
        reader = csv.reader(file, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        lines = 0
        skips = 0
        while True:
            try:
                row = next(reader)
                row = [x for x in row if not isinstance(x, str)]
                if len(row) in [3, 6, 9]:
                    data.append(row[:3])
                else:
                    skips += 1
            except ValueError:
                skips += 1
                pass
            except StopIteration:
                print(f"Found {lines} lines. Skipped {skips}. Loaded {lines - skips} points.")
                break
            lines += 1
    return np.array(data)