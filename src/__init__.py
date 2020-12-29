# from src.GeneticAlgorithm.GAManager import startSimulation
#
# startSimulation()

# import pycuda.driver as drv
# import numpy
#
# from pycuda.compiler import SourceModule
# mod = SourceModule("""
# __global__ void multiply_them(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] = a[i] * b[i];
# }
# """)
#
# multiply_them = mod.get_function("multiply_them")
#
# a = numpy.random.randn(400).astype(numpy.float32)
# b = numpy.random.randn(400).astype(numpy.float32)
#
# dest = numpy.zeros_like(a)
# multiply_them(
#         drv.Out(dest), drv.In(a), drv.In(b),
#         block=(400,1,1), grid=(1,1))
#
# print(dest-a*b)

# from src.examples.test2d import run_test
#
# run_test()

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.interactive(False)
# import numpy as np
# plt.hist(np.random.randn(100))
# plt.show()

from src.GeneticAlgorithm.GAManager import startSimulation

startSimulation()