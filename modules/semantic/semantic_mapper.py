from enum import Enum

class ObjectType(Enum):
    WALL = 0
    WINDOW = 1
    DOOR = 2
    UNKNOWN = 3

object_to_color = {
    ObjectType.WALL: 'black',
    ObjectType.WINDOW: 'blue',
    ObjectType.DOOR: 'red',
    ObjectType.UNKNOWN: 'pink',
}

room_type_to_id = {
    "living room": 0,
    "kitchen": 1,
    "bedroom": 2,
    "bathroom": 3,
    "balcony": 4,
    "corridor": 5,
    "dining room": 6,
    "study": 7,
    "studio": 8,
    "store room": 9,
    "garden": 10,
    "laundry room": 11,
    "office": 12,
    "basement": 13,
    "garage": 14,
    "undefined": 15
}

zind_room_type_to_id = {
    "bedroom": 0,
    "closet": 1,
    "hallway": 2,
    "living room": 3,
    "kitchen": 4,
    "bathroom": 5,
    "basement": 6,
    "dining room": 7,
    "loft": 8,
    "garage": 9,
    "laundry": 10,
    "office": 11,
    "stairs": 12,
    "undefined": 13
}
