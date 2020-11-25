cube_data = {
    "materials": [{"vertexshader": "shaders/vs-cube.txt", "fragmentshader": "shaders/fs-cube.txt", "numindices": 36}],
    "indices": [0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7, 8, 9, 10, 10, 9, 11, 12, 13, 14, 14, 13, 15, 16, 17, 18, 18, 17, 19,
                20, 21, 22, 22, 21, 23],
    "vertexPositions": [-0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                        -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5,
                        -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
                        -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5,
                        -0.5, 0.5, 0.5],
    "vertexNormals": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, 0,
                      -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0,
                      0, -1, 0, 0, -1, 0, 0, -1, 0, 0],

}

cube_data['vertexPositions'] = [cube_data['vertexPositions'][i:i + 3] for i in
                                range(0, len(cube_data['vertexPositions']), 3)]

cube_data['vertexNormals'] = [cube_data['vertexNormals'][i:i + 3] for i in
                              range(0, len(cube_data['vertexNormals']), 3)]

normal_face_mapping = {
    # relative to the original camera start pos
    "top": [0, 0, 1],
    "bottom": [0, 0, -1],

    "back": [0, 1, 0],
    "front": [0, -1, 0],

    "right": [-1, 0, 0],
    "left": [1, 0, 0]
}

face_normals = set([tuple(elem) for elem in cube_data['vertexNormals']])

normal_index_mapping = {
    # which normal corresponds to which vertex
    normal: [x for x in cube_data['indices'] if cube_data['vertexNormals'][x] == list(normal)]
    for normal in face_normals
}

texture_mapping = [
    # top
    [0, 0], [1, 0], [0, 1], [1, 1],

    # bottom
    [0, 0], [1, 0], [0, 1], [1, 1],

    # back
    [0, 0], [1, 0], [0, 1], [1, 1],

    # front
    [0, 0], [1, 0], [0, 1], [1, 1],

    # left
    [0, 0], [1, 0], [0, 1], [1, 1],

    # right
    [0, 0], [1, 0], [0, 1], [1, 1],

]
