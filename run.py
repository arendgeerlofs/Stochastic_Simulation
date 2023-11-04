from functions import *

iterations = 400
samples = 10**2

area_random = MC_integration(iterations, samples, randomS=True, LatinS=False)
# NOTE: very slow!!
area_Latin = MC_integration(iterations, samples, randomS=False, LatinS=True)

print(area_random, area_Latin)