from panda3d.core import WindowProperties, CompassEffect
from noise import pnoise2
import numpy as np
from direct.showbase.ShowBase import ShowBase, DirectionalLight
from panda3d.core import GeomTriangles, Geom, GeomNode, GeomVertexData, GeomVertexWriter, GeomVertexFormat, \
    DirectionalLight, AmbientLight, PointLight, Patchfile
from cube import cube_data, normal_face_mapping
from copy import copy
from noise import pnoise2
import numpy as np
import time
from copy import deepcopy
from panda3d.bullet import BulletRigidBodyNode, BulletTriangleMeshShape, BulletTriangleMesh
import sys


def make_cube_geom(pos=(0, 0, 0), geom_color=(1, 1, 1, 1), scale=2):
    format = GeomVertexFormat.getV3n3cpt2()

    shift_x, shift_y, shift_z = pos

    cube_vertex_list = deepcopy(cube_data["vertexPositions"])
    for vert in cube_vertex_list:
        # vert[0] *= scale
        # vert[1] *= scale
        # vert[2] *= scale
        vert[0] += shift_x
        vert[1] += shift_y
        vert[2] += shift_z

    vdata = GeomVertexData('square', format, Geom.UHDynamic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    # define available vertexes here
    [vertex.addData3(*v) for v in cube_vertex_list]
    [normal.addData3(*n) for n in cube_data["vertexNormals"]]
    [color.addData4f(*geom_color) for _ in cube_vertex_list]
    [texcoord.addData2f(1, 1) for _ in cube_vertex_list]

    """ CREATE A NEW PRIMITIVE """
    prim = GeomTriangles(Geom.UHStatic)

    # [prim.addVertex(v) for v in vertexes]
    indices = [ind for ind in cube_data['indices']]
    [prim.addVertex(v) for v in indices]

    #  N.B: this must correspond to a vertex defined in vdata
    geom = Geom(vdata)  # create a new geometry
    geom.addPrimitive(prim)  # add the primitive to the geometry

    return geom


class PlayerMovement:
    def __init__(self, app_class):
        self.app = app_class
        self.accept = app_class.accept
        self.camera = app_class.camera

        # vars for camera rotation
        self.heading = 0
        self.pitch = 0

        # movement
        self.forward = False
        self.reverse = False
        self.left = False
        self.right = False
        self.up = False
        self.down = False
        self.enabled = True
        self.wireframe = False

        self.app.disableMouse()

        # hide mouse cursor, comment these 3 lines to see the cursor
        props = WindowProperties()
        props.setCursorHidden(True)
        self.app.win.requestProperties(props)

        # dummy node for camera, attach player to it
        self.ref_node = self.app.ref_node
        self.ref_node.reparentTo(self.app.render)  # inherit transforms
        # self.ref_node.setEffect(CompassEffect.make(self.app.render))  # NOT inherit rotation

        # the camera
        self.app.camera.reparentTo(self.ref_node)
        # self.app.camera.lookAt(self.ref_node)
        # self.app.camera.setY(0)  # camera distance from model

        self.lastMouseX, self.lastMouseY = None, None
        self.rotateX, self.rotateY = -.5, -.5
        self.mouseMagnitude = 1
        self.previous_deltas = []

        self.app.accept("mouse3", self.mvnt_forward)
        self.app.accept("mouse3-up", self.mvnt_forward)
        self.app.accept("w", self.mvnt_forward)
        self.app.accept("w-up", self.mvnt_stopForward)
        self.app.accept("s", self.mvnt_reverse)
        self.app.accept("s-up", self.mvnt_stopReverse)
        self.app.accept("a", self.mvnt_left)
        self.app.accept("a-up", self.mvnt_stopLeft)
        self.app.accept("d", self.mvnt_right)
        self.app.accept("d-up", self.mvnt_stopRight)
        self.app.accept("space", self.mvnt_up)
        self.app.accept("space-up", self.mvnt_stop_up)
        self.app.accept("c", self.mvnt_down)
        self.app.accept("c-up", self.mvnt_stop_down)
        self.app.accept('r', self.reset_pos)
        self.app.accept('enter', self.toggle)
        self.app.accept('l', self.make_light)
        self.app.accept('q', self.toggle_wireframe)
        self.app.accept('escape', self.exit_app)

        self.app.taskMgr.add(self.cameraTask, 'cameraTask')

    # camera rotation task
    def cameraTask(self, task):
        mw = self.app.mouseWatcherNode
        dx = dy = 0
        x = y = 0
        hasMouse = mw.hasMouse()
        if hasMouse:
            # get the window manager's idea of the mouse position
            x, y = mw.getMouseX(), mw.getMouseY()

            if self.lastMouseX is not None:
                # get the delta
                dx, dy = x - self.lastMouseX, y - self.lastMouseY
            else:
                # no data to compare with yet
                dx, dy = 0, 0

        # TODO: fix mouse repositioning
        if x == 0 or y == 0:
            self.recenterMouse()

        self.lastMouseX, self.lastMouseY = x, y

        self.rotateX += dx * 150 * self.mouseMagnitude
        self.rotateY += dy * 150 * self.mouseMagnitude
        current_x, current_y = self.ref_node.getH(), self.ref_node.getP()

        dx *= 100
        dy *= 100

        self.ref_node.setHpr(current_x + dx, current_y + dy, 0)

        if self.forward:
            self.ref_node.setY(self.app.cam, self.ref_node.getY(self.app.cam) + 2)
        if self.reverse:
            self.ref_node.setY(self.app.cam, self.ref_node.getY(self.app.cam) - 2)
        if self.left:
            self.ref_node.setX(self.app.cam, self.ref_node.getX(self.app.cam) - 2)
        if self.right:
            self.ref_node.setX(self.app.cam, self.ref_node.getX(self.app.cam) + 2)
        if self.up:
            self.ref_node.setZ(self.app.cam, self.ref_node.getX(self.app.cam) + 2)
        if self.down:
            self.ref_node.setZ(self.app.cam, self.ref_node.getX(self.app.cam) - 2)

        return task.cont

    def recenterMouse(self):
        self.app.win.movePointer(0,
                                 int(self.app.win.getProperties().getXSize() / 2),
                                 int(self.app.win.getProperties().getYSize() / 2))

    def exit_app(self):
        self.app.destroy()
        sys.exit(1)

    def mvnt_forward(self):
        self.forward = True

    def mvnt_stopForward(self):
        self.forward = False

    def mvnt_reverse(self):
        self.reverse = True

    def mvnt_stopReverse(self):
        self.reverse = False

    def mvnt_left(self):
        self.left = True

    def mvnt_stopLeft(self):
        self.left = False

    def mvnt_right(self):
        self.right = True

    def mvnt_stopRight(self):
        self.right = False

    def mvnt_up(self):
        self.up = True

    def mvnt_stop_up(self):
        self.up = False

    def mvnt_down(self):
        self.down = True

    def mvnt_stop_down(self):
        self.down = False

    def reset_pos(self):
        self.ref_node.setX(0)
        self.ref_node.setY(0)
        self.ref_node.setZ(0)

    def toggle(self):
        if self.enabled:
            self.app.enableMouse()
            self.enabled = False
        else:
            self.app.disable_mouse()
            self.enabled = True

    def toggle_wireframe(self):
        if self.wireframe:
            self.app.render.setRenderModeFilled()
            self.wireframe = False
        else:
            self.app.render.setRenderModeWireframe()
            self.wireframe = True

    def make_light(self):

        d_light = DirectionalLight('dlight')
        d_light.setColor((0.2, 0.3, 0.3, 1))

        # d_light_np = self.app.render.attachNewNode(self.app.d_light)
        d_light = DirectionalLight('dlight')
        d_light.setColor((0.3, 0.3, 0.3, 1))
        d_light_np = self.app.render.attachNewNode(d_light)
        self.app.render.setLight(d_light_np)

        cam_cube_node = GeomNode('cam_cube')
        cam_cube_node.add_geom(make_cube_geom(geom_color=(1, 1, 0, 1)))
        cam_cube_path = self.app.render.attachNewNode(cam_cube_node)
        cam_cube_path.reparentTo(d_light_np)

        d_light_np.setPos(self.ref_node.getPos())
        d_light_np.setHpr(self.ref_node.getHpr())


class World:
    def __init__(self, app_class, n_grids=64):
        self.app = app_class
        self.cell_size = 8
        self.world_size = n_grids * self.cell_size
        self.view_distance_chunks = 6
        self.view_distance = self.cell_size * self.view_distance_chunks
        self.map_height_variation = 100
        self.app.accept('e', self.remove_block)
        self.app.accept('t', self.place_block)

        self.map_octaves = [1, 2, 3, 4, ]

        self.seen_cells = set()
        self.nearby_cells = []
        self.cells = {(x, y): {'visible': False,
                               'node_path': None,
                               'chunk_data': None,
                               'bullet_mesh': None}
                      for x in range(n_grids)
                      for y in range(n_grids)}

        self.app.taskMgr.add(self.build_world, 'build-world')

    def build_world(self, task):
        x = self.app.ref_node.getX()
        y = self.app.ref_node.getY()

        # TODO: Make it so we iterate over only a list of known grid coords to save computation
        # convert the players x and y coords so that we only consider chunks nearby
        # rel_x = int(x / self.cell_size)
        # rel_y = int(y / self.cell_size)
        #
        # self.nearby_cells = [
        #     [n_x, n_y]
        #     for n_x in range(rel_x - self.view_distance_chunks, rel_x + self.view_distance_chunks)
        #     for n_y in range(rel_y - self.view_distance_chunks, rel_y + self.view_distance_chunks)
        #     if n_x >= 0
        #     if n_y >= 0
        # ]
        #
        # if (rel_x, rel_y) not in self.seen_cells:
        #     self.seen_cells.add((rel_x, rel_y))
        #
        #     for cell in self.nearby_cells:
        #
        #         if cell not in list(self.cells.keys()):
        #             self.cells[tuple(cell)] = {'visible': False,
        #                                        'node_path': None,
        #                                        'chunk_data': None,
        #                                        'bullet_mesh': None}
        #
        # for k in self.cells.keys():
        #     if k in self.nearby_cells:
        #         self.cells[k]['visible'] = True
        #     else:
        #         self.cells[k]['visible'] = False

        # TODO: fix this complicated logic tree
        for k, v in self.cells.items():
            # k = tuple(cell_ind)
            # v = self.cells[k]

            min_px, min_py = k
            min_px, min_py = min_px * self.cell_size, min_py * self.cell_size
            max_px, max_py = min_px + self.cell_size, min_py + self.cell_size

            middle_x, middle_y = min_px + ((max_px - min_px) / 2), min_py + ((max_py - min_py) / 2)

            if self.pythag((x, y), (middle_x, middle_y)) < self.view_distance:

                if v['visible']:
                    pass
                else:
                    if v['node_path'] is None:

                        chunk = Chunk(
                            name=k,
                            x_range=(min_px, max_px),
                            y_range=(min_py, max_py),
                            octaves=self.map_octaves,
                            map_variation=self.map_height_variation,
                            total_world_size=self.world_size,
                        )

                        v['node_path'] = self.app.render.attachNewNode(chunk.chunk_node)
                        v['chunk_data'] = chunk
                        v['bullet_mesh'] = self.app.render.attachNewNode(BulletRigidBodyNode('bullet-' + str(k)))
                        v['bullet_mesh'].node().addShape(BulletTriangleMeshShape(chunk.bullet_mesh, dynamic=False))
                        self.app.bullet_world.attach(v['bullet_mesh'].node())


                    else:
                        v['node_path'].reparentTo(self.app.render)
                        v['bullet_mesh'].reparentTo(self.app.render)
                        self.app.bullet_world.attach(v['bullet_mesh'].node())

                    v['visible'] = True

            else:
                if v['visible']:  # if the cell if currently visible, rm from render graph
                    if v['node_path'] is not None:
                        v['node_path'].detachNode()
                        v['bullet_mesh'].detachNode()
                        self.app.bullet_world.remove(v['bullet_mesh'].node())
                v['visible'] = False

        return task.cont

    def remove_block(self):
        from_pos = self.app.ref_node.getPos(self.app.render)
        to_pos = self.app.player_ray_end.getPos(self.app.render)
        result = self.app.bullet_world.rayTestClosest(from_pos, to_pos)
        self.update_from_raycast(result, 'remove')

    def place_block(self):
        from_pos = self.app.ref_node.getPos(self.app.render)
        to_pos = self.app.player_ray_end.getPos(self.app.render)
        result = self.app.bullet_world.rayTestClosest(from_pos, to_pos)
        self.update_from_raycast(result, 'place')

    def update_from_raycast(self, result, method):
        has_hit = result.hasHit()
        if has_hit:
            # if we hit, then first find out which chunk to change
            chunk = result.getNode()

            cell_location = tuple(map(int, chunk.name.replace('bullet-(', '').rstrip(')').split(',')))

            # remove block and update chunk data
            self.cells[cell_location]['chunk_data'].edit_block(result.getHitPos(), method=method)

            # detach and remove meshes
            self.app.bullet_world.remove(self.cells[cell_location]['bullet_mesh'].node())
            self.cells[cell_location]['bullet_mesh'].removeNode()
            self.cells[cell_location]['node_path'].removeNode()
            # self.cells[cell_location]['bullet_mesh'].remove_node()

            # attach and get a new paths
            self.cells[cell_location]['bullet_mesh'] = self.app.render.attachNewNode(
                BulletRigidBodyNode('bullet-' + str(cell_location)))
            self.cells[cell_location]['node_path'] = self.app.render.attachNewNode(
                self.cells[cell_location]['chunk_data'].chunk_node)

            # re-add the bullet mesh to the bullet node
            self.cells[cell_location]['bullet_mesh'].node().addShape(
                BulletTriangleMeshShape(self.cells[cell_location]['chunk_data'].bullet_mesh, dynamic=False))
            self.app.bullet_world.attach(self.cells[cell_location]['bullet_mesh'].node())

    @staticmethod
    def pythag(pos1, pos2):
        """
        get the pythagorean distance between two points
        :param pos1: (tuple or np.arr) position of one point(s)
        :param pos2: (tuple or np.arr) position of one point(s)
        :return: the distance in whatever units the tuples were in
        """

        if isinstance(pos1, tuple):
            pos1_x, pos1_y, = pos1
            pos2_x, pos2_y = pos2
        else:
            pos1_x, pos1_y, = pos1[0], pos1[1]
            pos2_x, pos2_y = pos2[0], pos2[1]

        x = abs(pos1_x - pos2_x)
        y = abs(pos1_y - pos2_y)

        # a^2 + b^2 = c^2
        # c = sqrt(a^2 + b^2)

        return (x ** 2 + y ** 2) ** 0.5


class Chunk:
    """ This class keeps track of what cubes are where. Will also generate complete geoms and not draw
    the faces which are not in contact with the air """

    # TODO: move all mappings to a settings.py file or something
    dict_mapping = {
        0: 'front',
        1: 'back',
        2: 'top',
        3: 'bottom',
        4: 'left',
        5: 'right',
    }

    block_id_mapping = {
        0: {'name': 'dirt', 'color': (0.5, 0.5, 0.3, 1)},
        1: {'name': 'grass', 'color': (0.5, 0.75, 0.5, 1)},
        2: {'name': 'stone', 'color': (0.44, 0.5, 0.56, 1)},
        3: {'name': 'water', 'color': (0.12, 0.56, 1.0, 0.5)},
        4: {'name': 'sand', 'color': (0.98, 0.85, 0.37, 1)},
        5: {'name': 'bedrock', 'color': (0, 0, 0, 1)},

    }

    # this array will track where all of the blocks are, including what they are, and what faces are hidden
    # HOW TO USE
    # arr[x][y][z][data]
    # data = [block or no block,    | item: 0
    #           block_type,         | item: 1
    #           front hidden,       | item: 2
    #           back hidden,        | item: 3
    #           top hidden,         | item: 4
    #           bottom hidden,      | item: 5
    #           left hidden,        | item: 6
    #           right hidden]       | item: 7
    # NOTE: TOTAL 8 POSSIBILITIES FOR DATA

    def __init__(self, name, x_range, y_range, octaves, map_variation, total_world_size):
        self.chunk_size = x_range[1] - x_range[0]
        self.name = name
        self.min_game_bound = -2  # lowest possible point is -1 chunk down
        self.max_game_bound = 2

        if not isinstance(name, str):
            self.name = str(name)

        self.block_arr = np.zeros((self.chunk_size, self.chunk_size, self.max_game_bound - self.min_game_bound, 8))
        self.arr_x = self.block_arr.shape[0]
        self.arr_y = self.block_arr.shape[1]
        self.arr_z = self.block_arr.shape[2]

        self.x_range, self.y_range, self.octaves, self.map_variation, self.total_world_size = x_range, y_range, octaves, map_variation, total_world_size
        self.terrain_height_noise = self.calc_pnoise2_in_range()

        self.chunk_node = None
        self.bullet_mesh = None

        self.place_blocks()
        self.generate_chunk()

    def generate_chunk(self):
        """ this function will generate the actual geometry that is the chunk """
        self.update_block_faces()
        self.chunk_node = GeomNode(self.name)
        self.bullet_mesh = BulletTriangleMesh()
        for i in range(self.arr_x):
            for j in range(self.arr_y):
                for k in range(self.arr_z):
                    if self.block_arr[i, j, k][0] == 1:
                        # convert from chunk coordinates to world coordinates so the goemetry is in the right place
                        rel_x = i + self.x_range[0]
                        rel_y = j + self.y_range[0]
                        # only x and y need to be shifted, z is fine as is
                        geom = self.make_cube_geom(pos=(rel_x, rel_y, k),
                                                   faces_no_draw=self.block_arr[i, j, k][2:],
                                                   geom_color=self.block_id_mapping[self.block_arr[i, j, k][1]][
                                                       'color'])

                        self.chunk_node.addGeom(geom)
                        self.bullet_mesh.addGeom(geom)

    def place_blocks(self):
        # first, place blocks at the top level, and fill everything beneath with dirt
        for i in range(self.chunk_size):
            for j in range(self.chunk_size):
                z = int(round(self.terrain_height_noise[i][j]))

                if z >= self.arr_z:
                    z = self.arr_z - 1
                elif z < 0:
                    z = 0

                self.block_arr[i, j, z][0:2] = [1, 1]  # there is a block here, and it is grass
                if z == 0:  # halfway between top and bottom is water
                    self.block_arr[i, j, z][0:2] = [1, 3]  # there is a block here and it is water
                elif z == 1:  # this is just some sand so waters have beaches
                    self.block_arr[i, j, z][0:2] = [1, 4]  # there is a block here and it is sand

                if z > 0:  # dont add any extra if you're at the very bottom
                    self.block_arr[i, j, 0:z] = [[1, 0] + [0] * 6] * z  # there is a block here and it is dirt

                    if z - 3 > 0:
                        self.block_arr[i, j, 0:z - 3] = [[1, 2] + [0] * 6] * (
                                z - 3)  # there is a block here and it is stone

    def edit_block(self, hit_pos, method='remove'):
        """ call this function to remove a geometry, and redraw """

        # convert from world coords to chunk coords
        hit_x, hit_y, hit_z = hit_pos
        chunk_hit_pos = hit_x - self.x_range[0], hit_y - self.y_range[0], hit_z
        chunk_hit_pos_rounded = list(map(round, chunk_hit_pos))
        chunk_x, chunk_y, chunk_z = chunk_hit_pos_rounded

        search_range = 2
        range_x = [chunk_x - search_range, chunk_x + search_range]
        range_y = [chunk_y - search_range, chunk_y + search_range]
        range_z = [chunk_z - search_range, chunk_z + search_range]

        if range_y[0] < 0:
            range_y[0] = 0
        elif range_y[1] >= self.arr_y:
            range_y[1] = self.arr_y

        if range_x[0] < 0:
            range_x[0] = 0
        elif range_x[1] >= self.arr_x:
            range_x[1] = self.arr_x

        if range_z[0] < 0:
            range_z[0] = 0
        elif range_z[1] >= self.arr_z:
            range_z[1] = self.arr_z

        # we know there was a hit (otherwise this function would not be called)
        # so get a range of possible points, and pick the closest one that has something
        # inside of it

        if method == 'place':
            search = 0
            replace_data = [1, 1] + [0] * 6
        else:
            search = 1
            replace_data = [0] * 8

        closest_coords = None
        smallest_dist = None
        for i in range(*range_x):
            for j in range(*range_y):
                for k in range(*range_z):

                    # TODO: figure out why blocks are not placed properly and fix here
                    dist = self.pythag_3d(chunk_hit_pos, (i, j, k))
                    if self.block_arr[i, j, k][0] == search:
                        if smallest_dist is None:
                            smallest_dist = dist
                            closest_coords = i, j, k
                        else:
                            if dist < smallest_dist and self.block_arr[i, j, k][0] == search:
                                closest_coords = i, j, k
                                smallest_dist = dist

        if closest_coords is not None:
            chunk_x, chunk_y, chunk_z = closest_coords
            self.block_arr[chunk_x, chunk_y, chunk_z] = replace_data  # replace data with nothing
            # regenerate chunk mesh
            self.generate_chunk()

    def update_block_faces(self):
        """ this function governs what faces are drawn """
        for i in range(self.arr_x):
            for j in range(self.arr_y):
                for k in range(self.arr_z):

                    if self.block_arr[i, j, k][0] == 1:  # so long as we're not talking about air...

                        if k == 0:
                            # there is a block at the very bottom, do not draw the bottom face
                            self.block_arr[i, j, k][5] = 1

                        if k < self.arr_z - 1:  # draw a top at the max height
                            # if there is a block above this one, then draw the top face
                            if self.block_arr[i, j, k + 1][0] == 0:
                                self.block_arr[i, j, k][4] = 0
                            else:
                                self.block_arr[i, j, k][4] = 1

                        if k > 0:
                            # if there is a block below this one, do not draw the face below
                            if self.block_arr[i, j, k - 1][0] == 1:
                                self.block_arr[i, j, k][5] = 1
                            else:
                                self.block_arr[i, j, k][5] = 0

                        if j > 0:
                            # if there is a block to the right of this block, then do not draw that face
                            if self.block_arr[i, j - 1, k][0] == 1:
                                self.block_arr[i, j, k][2] = 1
                            else:
                                self.block_arr[i, j, k][2] = 0

                        if j < self.arr_y - 1:
                            # if there is a block to the right of this block, then do not draw that face
                            if self.block_arr[i, j + 1, k][0] == 1:
                                self.block_arr[i, j, k][3] = 1
                            else:
                                self.block_arr[i, j, k][3] = 0

                        if i > 0:
                            # if there is a block to the left of this block, then do not draw that face
                            if self.block_arr[i - 1, j, k][0] == 1:
                                self.block_arr[i, j, k][7] = 1
                            else:
                                self.block_arr[i, j, k][7] = 0

                        if i < self.arr_x - 1:
                            # if there is a block to the left of this block, then do not draw that face
                            if self.block_arr[i + 1, j, k][0] == 1:
                                self.block_arr[i, j, k][6] = 1
                            else:
                                self.block_arr[i, j, k][6] = 0

    def calc_pnoise2_in_range(self):
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        arr = np.zeros((int(x_max - x_min), int(y_max - y_min)))

        for octave in self.octaves:
            index_i = 0
            for i in range(x_min, x_max):
                index_j = 0

                for j in range(y_min, y_max):
                    arr[index_i][index_j] = pnoise2(i / self.total_world_size, j / self.total_world_size,
                                                    octave) * self.map_variation

                    index_j += 1
                index_i += 1

        return arr

    def make_cube_geom(self, pos=(0, 0, 0), geom_color=(1, 1, 1, 1), faces_no_draw=(0, 0, 0, 0, 0, 0)):
        format = GeomVertexFormat.getV3n3cpt2()

        shift_x, shift_y, shift_z = pos

        cube_vertex_list = deepcopy(cube_data["vertexPositions"])
        for vert in cube_vertex_list:
            vert[0] += shift_x
            vert[1] += shift_y
            vert[2] += shift_z

        vdata = GeomVertexData('square', format, Geom.UHDynamic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')

        # define available vertexes here
        [vertex.addData3(*v) for v in cube_vertex_list]
        [normal.addData3(*n) for n in cube_data["vertexNormals"]]
        [color.addData4f(*geom_color) for _ in cube_vertex_list]
        [texcoord.addData2f(1, 1) for _ in cube_vertex_list]

        """ CREATE A NEW PRIMITIVE """
        prim = GeomTriangles(Geom.UHStatic)

        # convert from numpy arr mapping to normal faces mapping...
        excluded_normals = [normal_face_mapping[self.dict_mapping[i]] for i in range(len(faces_no_draw)) if
                            faces_no_draw[i] == 1]

        # only use acceptable indices
        indices = [ind for ind in cube_data['indices'] if cube_data['vertexNormals'][ind] not in excluded_normals]
        [prim.addVertex(v) for v in indices]

        #  N.B: this must correspond to a vertex defined in vdata
        geom = Geom(vdata)  # create a new geometry
        geom.addPrimitive(prim)  # add the primitive to the geometry

        return geom

    @staticmethod
    def pythag_3d(pos1, pos2):
        pos1_x, pos1_y, pos1_z = pos1
        pos2_x, pos2_y, pos2_z = pos2

        x = abs(pos1_x - pos2_x)
        y = abs(pos1_y - pos2_y)
        z = abs(pos1_z - pos2_z)

        return (x ** 2 + y ** 2 + z ** 2) ** 0.5
