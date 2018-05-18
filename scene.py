from vpython import *

class Scene():

    def __init__(self, sim, target):

        self._copter = box(
                length=sim.dims[1],
                width=sim.dims[0],
                height=sim.dims[2])
        self._shadow = box(
                length=sim.dims[1],
                width=sim.dims[0],
                height=0.001,
                color=color.black)
        ground = box(
                pos=vector(0, 0, -0.001),
                length=10,
                width=10,
                height=0.001,
                color=color.green)
        target = vector(target[0], target[2], target[1])
        self._axis_x = arrow(
                pos = target,
                axis = vector(1, 0, 0),
                color = color.red)
        self._axis_y = arrow(
                pos = target,
                axis = vector(0, 0, 1),
                color = color.green)
        self._axis_z = arrow(
                pos = target,
                axis = vector(0, 1, 0),
                color = color.blue)
        scene.background = color.blue

    def update(self, pose):
        rate(20)
        coords = pose[:3]
        angles = pose[3:]
        self._copter.pos = vector(coords[0], coords[2], coords[1])
        self._shadow.pos = vector(coords[0], 0.001, coords[1])
        scene.center = self._copter.pos
        return
