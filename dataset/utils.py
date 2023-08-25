
def name_2_ficosa_id(name):
    name_to_id = {
        'car': 0,
        'truck': 1,
        'bicycle': 2,
        'person': 3,
        'motorbike': 4,
        'bus': 5,
        'traffic_sign': 6,
        'traffic_light': 7
    }
    return name_to_id[name]