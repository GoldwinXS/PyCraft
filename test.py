from direct.showbase.ShowBase import ShowBase, DirectionalLight
from panda3d.core import GeomTriangles, Geom, GeomNode, GeomVertexData, GeomVertexWriter, GeomVertexFormat, \
    DirectionalLight, AmbientLight, PointLight, TextNode, Point3, NodePath, Fog, BitMask32, TransformState, Vec3, \
    ClockObject
from direct.gui.OnscreenText import OnscreenText
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletTriangleMeshShape, BulletTriangleMesh, \
    BulletPlaneShape, BulletBoxShape, BulletDebugNode

from copy import copy
from noise import pnoise2
import numpy as np
import time
from ProjectUtils import PlayerMovement, World, make_cube_geom


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        """ SETUP CONFIG VARS  """
        # self.render.setShaderAuto()
        self.globalClock = ClockObject().getGlobalClock()
        self.bullet_world = BulletWorld()
        self.bullet_world.setGravity((0, 0, -9.81))

        """ PLAYER SETUP """
        # player camera
        self.ref_node = self.render.attachNewNode('camparent')

        self.debugNP = self.render.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.show()

        self.player_body_geom = make_cube_geom(geom_color=(1, 0, 0, 1))
        player_body_geom_node = GeomNode('player')
        player_body_geom_node.addGeom(self.player_body_geom)
        self.player_body_geom_np = self.render.attachNewNode(player_body_geom_node)

        # Box
        shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))

        self.box_np = self.render.attachNewNode(BulletRigidBodyNode('Box'))
        self.box_np.node().setMass(1.0)
        self.box_np.node().addShape(shape)
        self.box_np.node().addShape(shape, TransformState.makePos(Point3(0, 1, 0)))
        self.box_np.node().setDeactivationEnabled(False)
        self.player_body_geom_np.reparentTo(self.box_np)
        self.box_np.setPos(10, 10, 10)
        self.box_np.setCollideMask(BitMask32.allOn())

        self.bullet_world.attachRigidBody(self.box_np.node())

        # self.player_body_geom = make_cube_geom(geom_color=(1, 0, 0, 1))
        # player_body_geom_node= GeomNode('player')
        # player_body_geom_node.addGeom(self.player_body_geom)
        # self.player_body_geom_np = self.render.attachNewNode(player_body_geom_node)
        #
        # shape = BulletTriangleMesh()
        # shape.addGeom(self.player_body_geom)
        # player_body_col = self.render.attachNewNode(BulletRigidBodyNode('player'))
        # player_body_col.node().addShape(BulletTriangleMeshShape(shape, dynamic=False))
        # self.bullet_world.attach(player_body_col.node())

        # TODO: move player stuff to a new class
        # TODO: add cross hairs on player hud
        self.player_ray_end = NodePath('player-ray')
        self.player_ray_end.reparentTo(self.render)
        self.player_ray_end.reparentTo(self.ref_node)
        self.player_ray_end.setY(+10)  # put this in front of the player
        self.ref_node.lookAt(self.player_ray_end)

        """ SETUP VARS AND USE UTIL CLASSES """

        self.map_scale = 10
        self.map_size = 100
        self.camLens.setFov(90)
        total_size = 32 * 8

        PlayerMovement(self)
        World(self, n_grids=32)

        """ SKY BOX SETUP """

        skybox = GeomNode('skybox')
        skybox.addGeom(make_cube_geom(pos=(0, 0, 0), geom_color=(0.2, 0.2, 0.7, 1)))
        skybox_np = self.render.attachNewNode(skybox)
        skybox_np.reparentTo(self.ref_node)
        skybox_np.setPos((0, 1200, 0))
        skybox_np.setScale(1200)

        """ ADD LIGHTS """

        # ambient light
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0.2, 0.2, 0.2, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        # a geom to represent the sun
        sun_geom_node = GeomNode('sun')
        sun_geom_node.addGeom(make_cube_geom(geom_color=(0.5, 0.5, 0.3, 1)))
        sun_geom_node_path = self.render.attachNewNode(sun_geom_node)
        sun_geom_node_path.setScale(10)

        # sun
        sun = PointLight('sun')
        sun.setColor((0.4, 0.3, 0.27, 1))
        sun.setAttenuation((0.1, 0, 0))
        sun_np = self.render.attachNewNode(sun)
        sun_geom_node_path.reparentTo(sun_np)

        self.render.setLight(sun_np)
        sun_np.setPos(total_size / 2, total_size / 2, 200)

        """ SETUP FOR ONSCREEN TEXT """

        # TODO: add chunk coords to display
        self.text = OnscreenText(text='fps', pos=(-1, -0.95), fg=(0, 1, 0, 1), align=TextNode.ALeft, scale=0.1)
        self.location_text = OnscreenText(text='location', pos=(-1, 0.8), fg=(0, 1, 0, 1), align=TextNode.ALeft,
                                          scale=0.1)

        self.previous_time = time.time()
        self.taskMgr.add(self.update_text, 'fps')
        self.taskMgr.add(self.update, 'physics')

    def update_text(self, task):
        time_since_last_called = time.time() - self.previous_time
        self.text.setText('fps: ' + str(round(1 / time_since_last_called)))
        self.previous_time = time.time()

        x, y, z = self.ref_node.getPos()
        self.location_text.setText(f'Player location x: {round(x)}, y: {round(y)}, z: {round(z)},')

        return task.cont

    def update(self, task):
        dt = self.globalClock.getDt()

        # self.processInput(dt)
        print(self.box_np.getPos())
        self.bullet_world.doPhysics(dt, 10, 0.008)
        # self.processContacts()

        return task.cont


app = MyApp()
app.run()
