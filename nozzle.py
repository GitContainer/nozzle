import math
import scipy.optimize

gamma = 1.4

def nu(M):

	gm1 = gamma - 1.0
	gp1 = gamma + 1.0
	G = math.sqrt(gp1 / gm1)
	phi = math.sqrt(M * M - 1.0)
	return G * math.atan(1.0 / G * phi) - math.atan(phi)

def Mach(nu):

	M = scipy.optimize.newton(lambda M: PrandtlMeyer(M) - nu, 1.1)
	return M

if __name__ == "__main__":
	print "nu = ", math.degrees(PrandtlMeyer(1.2))

	nu = math.radians(3.558)
	print "nu = ", nu
	print "M = ", PrandtlMeyerInverse(nu)

