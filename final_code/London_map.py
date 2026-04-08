"""
London Map Data
===============
30-node London-inspired graph for safety-aware routing.

Each node is a recognisable London location placed at its real GPS coordinates
(latitude, longitude). The adjacency list reflects approximate geographic
proximity and major transport links. Three zones: central (safe), inner
(mixed), outer (high crime), reflecting London's actual crime distribution.

This file is consumed by safety_aware_a_star.load_map("london").
"""

# Greater London bounding box used to normalise GPS coordinates to [0, 1]
BOUNDING_BOX = {
    "lat_min": 51.33,   # south (Croydon)
    "lat_max": 51.62,   # north (Edmonton)
    "lon_min": -0.31,   # west  (Hammersmith)
    "lon_max":  0.08,   # east  (Barking)
}

NODE_NAMES = {
    0: "Westminster",   1: "Covent Garden",  2: "City of London", 3: "South Bank",
    4: "Soho",          5: "Kings Cross",    6: "Tower Bridge",   7: "Waterloo",
    8: "Marylebone",    9: "Bloomsbury",    10: "Camden",        11: "Shoreditch",
    12: "Elephant & Castle", 13: "Notting Hill", 14: "Islington", 15: "Greenwich",
    16: "Clapham",     17: "Hammersmith",  18: "Hackney",        19: "Stratford",
    20: "Tottenham",   21: "Croydon",      22: "Lewisham",       23: "Barking",
    24: "Brixton",     25: "Peckham",      26: "Wood Green",     27: "Woolwich",
    28: "Edmonton",    29: "Seven Sisters",
}

# Real GPS coordinates (latitude, longitude) for each node
GPS = {
    0: (51.4947, -0.1353),  1: (51.5117, -0.1240),  2: (51.5155, -0.0922),
    3: (51.5055, -0.1160),  4: (51.5133, -0.1312),  5: (51.5317, -0.1240),
    6: (51.5055, -0.0754),  7: (51.5031, -0.1132),  8: (51.5225, -0.1544),
    9: (51.5218, -0.1278), 10: (51.5390, -0.1426), 11: (51.5264, -0.0769),
    12: (51.4946, -0.1006), 13: (51.5092, -0.1964), 14: (51.5362, -0.1032),
    15: (51.4769, -0.0005), 16: (51.4620, -0.1380), 17: (51.4928, -0.2236),
    18: (51.5450, -0.0553), 19: (51.5430, -0.0034), 20: (51.5880, -0.0720),
    21: (51.3762, -0.0986), 22: (51.4415, -0.0117), 23: (51.5362,  0.0808),
    24: (51.4613, -0.1150), 25: (51.4738, -0.0693), 26: (51.5975, -0.1096),
    27: (51.4893,  0.0654), 28: (51.6137, -0.0625), 29: (51.5833, -0.0726),
}

# Adjacency list: which nodes connect to which (geographic proximity and major
# transport links). Connections create competing route options where fast paths
# through dangerous areas can be compared against safer detours.
ADJACENCY = {
    0:  [1, 3, 4, 7, 8, 13, 16, 24],   1:  [0, 2, 3, 4, 9],
    2:  [1, 3, 6, 11, 14],              3:  [0, 1, 2, 6, 7, 12],
    4:  [0, 1, 8, 9, 10, 13],           5:  [9, 10, 14, 20, 26],
    6:  [2, 3, 11, 12, 15, 25],         7:  [0, 3, 12, 16, 24],
    8:  [0, 4, 9, 10, 13, 17],          9:  [1, 4, 5, 8, 10, 14],
    10: [4, 5, 8, 9, 14, 20, 26],       11: [2, 6, 14, 18, 19],
    12: [3, 6, 7, 15, 16, 22, 24, 25],  13: [0, 4, 8, 17],
    14: [2, 5, 9, 10, 11, 18],          15: [6, 12, 22, 25, 27],
    16: [0, 7, 12, 21, 24],              17: [8, 13],
    18: [11, 14, 19, 20, 23, 29],       19: [11, 18, 23, 27],
    20: [5, 10, 18, 26, 28, 29],        21: [16, 22, 25],
    22: [12, 15, 21, 25, 27],            23: [18, 19, 27],
    24: [0, 7, 12, 16, 25],              25: [6, 12, 15, 21, 22, 24],
    26: [5, 10, 20, 28, 29],             27: [15, 19, 22, 23],
    28: [20, 26, 29],                    29: [18, 20, 26, 28],
}