from direct.showbase.ShowBase import ShowBase, DirectionalLight
from panda3d.core import GeomTriangles, Geom, GeomNode, GeomVertexData, GeomVertexWriter, GeomVertexFormat, \
    DirectionalLight, AmbientLight, PointLight, TextNode, Point3, NodePath
from direct.gui.OnscreenText import OnscreenText
from panda3d.bullet import BulletWorld

from copy import copy
from noise import pnoise2
import numpy as np
import time
from ProjectUtils import PlayerMovement, World, make_cube_geom


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        """ REGISTER A FORMAT """
        # self.format = GeomVertexFormat.getV3n3cpt2()
        # self.render.setShaderAuto()

        # player camera
        self.ref_node = self.render.attachNewNode('camparent')

        # TODO: move player stuff to a new class
        # TODO: add cross hairs on player hud
        self.player_ray_end = NodePath('player-ray')
        self.player_ray_end.reparentTo(self.render)
        self.player_ray_end.reparentTo(self.ref_node)
        self.player_ray_end.setY(+10)  # put this in front of the player
        self.ref_node.lookAt(self.player_ray_end)

        # TODO: add chunk coords to display
        self.text = OnscreenText(text='fps', pos=(-1, -0.95), fg=(0, 1, 0, 1), align=TextNode.ALeft, scale=0.1)
        self.location_text = OnscreenText(text='location', pos=(-1, 0.8), fg=(0, 1, 0, 1), align=TextNode.ALeft,
                                          scale=0.1)

        self.camLens.setFov(90)

        self.bullet_world = BulletWorld()
        PlayerMovement(self)
        World(self)

        self.map_scale = 10
        self.map_size = 100

        # Create Ambient Light
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0.1, 0.1, 0.1, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        self.previous_time = time.time()
        self.taskMgr.add(self.update_text, 'fps')


    def update_text(self, task):
        time_since_last_called = time.time() - self.previous_time
        self.text.setText('fps: ' + str(round(1 / time_since_last_called)))
        self.previous_time = time.time()


        x, y, z = self.ref_node.getPos()
        self.location_text.setText(f'Player location x: {round(x)}, y: {round(y)}, z: {round(z)},')

        return task.cont


app = MyApp()
app.run()
