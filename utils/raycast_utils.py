import numpy as np
from modules.semantic.semantic_mapper import ObjectType, object_to_color

def get_color_name(r, g, b):
    """
    Convert RGB values to a descriptive color name.
    Input:
        r, g, b: The RGB values of the pixel.
    Output:
        color_name: A string representing the color ('black', 'blue', 'red', etc.)
    """
    if r < 0.1 and g < 0.1 and b < 0.1:
        return 'black'
    elif r > 0.8 and g < 0.2 and b < 0.2:
        return 'red'
    elif r < 0.2 and g < 0.2 and b > 0.8:
        return 'blue'
    else:
        return None
    
def cast_ray(occ, pos, theta, dist_max=1500, min_dist=5):
    """
    Cast a single ray with modified behavior:
      - When a red pixel ('red') is encountered, record its info and continue.
      - When a black or blue pixel is encountered:
            * If a red pixel was seen previously, return a red (DOOR) hit with the current distance.
            * Otherwise, return the hit with its corresponding type:
                  'black' -> WALL, 'blue' -> WINDOW.
      - If the ray goes out-of-bound:
            * If a red pixel was recorded, return that hit as DOOR.
            * Otherwise, return an unknown hit.
    """
    img_height, img_width = occ.shape[:2]
    pos_x, pos_y = pos

    sin_ang = np.sin(theta)
    cos_ang = np.cos(theta)
    
    last_red_hit = None  # Record (distance, (x, y)) for the last red pixel encountered

    for dist in range(5, dist_max,1):
        new_x = int(pos_x + dist * cos_ang)
        new_y = int(pos_y - dist * sin_ang)  # Note: image y-coordinates increase downwards

        # Out-of-bound check
        if new_x < 0 or new_x >= img_width or new_y < 0 or new_y >= img_height:
            if last_red_hit is not None:
                red_dist, red_coord = last_red_hit
                return red_dist, ObjectType.DOOR.value, red_coord
            else:
                return dist, ObjectType.UNKNOWN.value, (new_x, new_y)

        occ_value = occ[new_y, new_x]

        # Use only the first three channels assuming RGB
        if len(occ_value) > 3:
            occ_value = occ_value[:3]
        r, g, b = occ_value

        color_name = get_color_name(r, g, b)
        if color_name:
            if color_name == 'red':
                # Record red hit and continue the ray
                last_red_hit = (dist, (new_x, new_y))
                continue
            elif color_name in ['black', 'blue']:
                # If a red pixel was seen earlier, override and return as DOOR (red)
                if last_red_hit:
                    last_hit_dist, _ = last_red_hit
                if last_red_hit is not None and last_hit_dist > 60:                
                    return dist, ObjectType.DOOR.value, (new_x, new_y)
                else:
                    # Return the black or blue hit with the proper enum type
                    if color_name == 'black':
                        return dist, ObjectType.WALL.value, (new_x, new_y)
                    elif color_name == 'blue':
                        return dist, ObjectType.WINDOW.value, (new_x, new_y)
    
    # If no decisive pixel is found within dist_max:
    if last_red_hit:
        red_dist, red_coord = last_red_hit
        return red_dist, ObjectType.DOOR.value, red_coord
    else:
        return dist_max, ObjectType.UNKNOWN.value, (new_x, new_y)

def ray_cast(occ, pos, theta, dist_max=1500, min_dist=5):# Cast the rimary ray
    dist, obj_type, hit_coords = cast_ray(occ, pos, theta, dist_max, min_dist)
    return dist, obj_type, hit_coords