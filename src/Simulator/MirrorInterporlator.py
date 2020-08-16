from src.Simulator.Vector import Vector
import numpy as np
import json

with open('../config.json') as config_file:
    config = json.load(config_file)


class MirrorInterpolator:
    a00, a01, a02, a03 = 0, 0, 0, 0
    a10, a11, a12, a13 = 0, 0, 0, 0
    a20, a21, a22, a23 = 0, 0, 0, 0
    a30, a31, a32, a33 = 0, 0, 0, 0
    p = np.empty([4, 4], float)
    mirror = np.empty([config["mirrorGridDensity"], config["mirrorGridDensity"]])

    def __init__(self, _mirror, _xInterpPoint, _yInterpPoint):
        self.mirror = _mirror
        self.setpvalues(self.mirror, _xInterpPoint, _yInterpPoint)
        return

    def setpvalues(self, mirror, x, y):
        self.p = 0

        print(mirror)


        return

    def updatecoefficients(self):
        self.a00 = self.p[1][1]
        self.a01 = -.5 * self.p[1][0] + .5 *self.p[1][2]
        self.a02 = self.p[1][0] - 2.5 * self.p[1][1] + 2 * self.p[1][2] - .5 * self.p[1][3]
        self.a03 = -.5 * self.p[1][0] + 1.5 * self.p[1][1] - 1.5 * self.p[1][2] + .5 * self.p[1][3]
        self.a10 = -.5 * self.p[0][1] + .5 * self.p[2][1]
        self.a11 = .25 * self.p[0][0] - .25 * self.p[0][2] - .25 *self.p[2][0] + .25 *self.p[2][2]
        self.a12 = -.5 *self.p[0][0] + 1.25 *self.p[0][1] -self.p[0][2] + .25 *self.p[0][3] + .5 *self.p[2][0] - 1.25 *self.p[2][1] +\
                  self.p[2][2] - .25 *self.p[2][3]
        self.a13 = .25 *self.p[0][0] - .75 *self.p[0][1] + .75 *self.p[0][2] - .25 *self.p[0][3] - .25 *self.p[2][0] + \
                   .75 *self.p[2][1] - .75 *self.p[2][2] + .25 *self.p[2][3]
        self.a20 =self.p[0][1] - 2.5 *self.p[1][1] + 2 *self.p[2][1] - .5 *self.p[3][1]
        self.a21 = -.5 *self.p[0][0] + .5 *self.p[0][2] + 1.25 *self.p[1][0] - 1.25 *self.p[1][2] -self.p[2][0] +self.p[2][2] + .25 *self.p[3][0] - .25 * \
             self.p[3][2]
        self.a22 =self.p[0][0] - 2.5 *self.p[0][1] + 2 *self.p[0][2] - .5 *self.p[0][3] - 2.5 *self.p[1][0] + 6.25 *self.p[1][1] - 5 *self.p[1][2] + 1.25 * \
             self.p[1][3] + 2 *self.p[2][0] - 5 *self.p[2][1] + 4 *self.p[2][2] -self.p[2][3] - .5 *self.p[3][0] + 1.25 *self.p[3][1] -self.p[3][2] + .25 * \
             self.p[3][3]
        self.a23 = -.5 *self.p[0][0] + 1.5 *self.p[0][1] - 1.5 *self.p[0][2] + .5 *self.p[0][3] + 1.25 *self.p[1][0] - 3.75 *self.p[1][1] + 3.75 *self.p[1][
            2] - 1.25 *self.p[1][3] -self.p[2][0] + 3 *self.p[2][1] - 3 *self.p[2][2] +self.p[2][3] + .25 *self.p[3][0] - .75 *self.p[3][1] + .75 * \
             self.p[3][2] - .25 *self.p[3][3]
        self.a30 = -.5 *self.p[0][1] + 1.5 *self.p[1][1] - 1.5 *self.p[2][1] + .5 *self.p[3][1]
        self.a31 = .25 *self.p[0][0] - .25 *self.p[0][2] - .75 *self.p[1][0] + .75 *self.p[1][2] + .75 *self.p[2][0] - .75 *self.p[2][2] - .25 *self.p[3][
            0] + .25 *self.p[3][2]
        self.a32 = -.5 *self.p[0][0] + 1.25 *self.p[0][1] -self.p[0][2] + .25 *self.p[0][3] + 1.5 *self.p[1][0] - 3.75 *self.p[1][1] + 3 *self.p[1][
            2] - .75 *self.p[1][3] - 1.5 *self.p[2][0] + 3.75 *self.p[2][1] - 3 *self.p[2][2] + .75 *self.p[2][3] + .5 *self.p[3][0] - 1.25 *self.p[3][
                  1] +self.p[3][2] - .25 *self.p[3][3]
        self.a33 = .25 *self.p[0][0] - .75 *self.p[0][1] + .75 *self.p[0][2] - .25 *self.p[0][3] - .75 *self.p[1][0] + 2.25 *self.p[1][1] - 2.25 *self.p[1][
            2] + .75 *self.p[1][3] + .75 *self.p[2][0] - 2.25 *self.p[2][1] + 2.25 * self.p[2][2] - .75 * self.p[2][3] - .25 * self.p[3][0] + .75 * \
              self.p[3][1] - .75 * self.p[3][2] + .25 * self.p[3][3]
        return


    def getvalue(self ,x , y):
        x2 = x * x
        x3 = x2 * x
        y2 = y * y
        y3 = y2 * y

        interpolatedValueCalculation = (self.a00 + self.a01 * y + self.a02 * y2 + self.a03 * y3) + (self.a10 + self.a11
                                        * y + self.a12 * y2 + self.a13 * y3) * x + (self.a20 + self.a21 * y + self.a22 *
                                        y2 + self.a23 * y3) * x2 + (self.a30 + self.a31 * y + self.a32 * y2 + self.a33 *
                                                                    y3) * x3


        return interpolatedValueCalculation