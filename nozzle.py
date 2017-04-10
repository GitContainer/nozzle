import math
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
R_THROAT = 1.0
X_INFLECTION = 1.0
THETA_INFLECTION = 0.1
MACH_EXIT = 2.0
THETA_EXIT = 0.5

def RWallEquation(x, yThroat, xInflection, thetaInflection):

	x1 = xInflection
	a = -math.tan(thetaInflection) / 3.0 / x1 / x1
	b = math.tan(thetaInflection) / x1
	c = 0.01
	d = yThroat

	r = a * x * x * x + b * x * x + c * x + d
	drdx = 3.0 * a * x * x + 2.0 * b * x + c

	return r, drdx

def RExpWall(x):

	return RWallEquation(x, R_THROAT, X_INFLECTION, THETA_INFLECTION)

class PM(object):

	def __init__(self, gamma_):
		self.gamma = gamma_

	def nu(self, M):
		G = math.sqrt((self.gamma + 1.0) / (self.gamma - 1.0))
		phi = math.sqrt(M * M - 1.0)
		return G * math.atan(1.0 / G * phi) - math.atan(phi)

	def Mach(self, nu):
		M = scipy.optimize.newton(lambda M: self.nu(M) - nu, 1.1)
		return M

def TestPM():

	pm = PM(1.4)
	print "nu = ", math.degrees(pm.nu(1.2))

	nu = math.radians(3.558)
	print "nu = ", nu
	print "M = ", pm.Mach(nu)

def IntegrateInteriorPoint(pm, x1, r1, theta1, nu1, M1, x2, r2, theta2, nu2, M2):
	"""
	Along C- characteristics from P1 and C+ characteristics from P2,
	integrate compatibility equations to obtain condition at P3
	"""

	print "Interior point"
	print "  Point 1: ", x1, r1, math.degrees(theta1), math.degrees(nu1), M1
	print "  Point 2: ", x2, r2, math.degrees(theta2), math.degrees(nu2), M2

	mu1 = math.asin(1.0 / M1)
	Km1 = theta1 + nu1
	drdxm1 = math.tan(theta1 - mu1)

	mu2 = math.asin(1.0 / M2)
	Kp2 = theta2 - nu2
	drdxp2 = math.tan(theta2 + mu2)

	x3 = (drdxm1 * x1 - drdxp2 * x2 - r1 + r2) / (drdxm1 - drdxp2)
	r3 = r1 + drdxm1 * (x3 - x1)

	r13 = 0.5 * (r1 + r3)
	r23 = 0.5 * (r2 + r3)
	dKmdr =  math.sin(theta1) / (math.sin(theta1) * math.sqrt(M1 * M1 - 1.0) - math.cos(theta1)) / r13
	dKpdr = -math.sin(theta2) / (math.sin(theta2) * math.sqrt(M2 * M2 - 1.0) + math.cos(theta2)) / r23

	Km3 = Km1 + dKmdr * (r3 - r1)
	Kp3 = Kp2 + dKpdr * (r3 - r2)

	theta3 = 0.5 * (Km3 + Kp3)
	nu3    = 0.5 * (Km3 - Kp3)

	return x3, r3, theta3, nu3

def IntersectExpansionWallPoint(x2, r2, theta2, nu2, M2, RWallFunc, xMax):

	if x2 > xMax:
		return None, None, None

	mu2 = math.asin(1.0 / M2)
	Kp2 = theta2 - nu2
	drdxp2 = math.tan(theta2 + mu2)

	print "IntersectExpWallPoint:", x2, r2, drdxp2
	f = lambda x3: abs(r2 + drdxp2 * (x3 - x2) - RWallFunc(x3)[0])
	#fprime = lambda x3: drdxp2 - RWallFunc(x3)[1]
	x3, fval, ierr, numfunc = scipy.optimize.fminbound(f, x2, xMax, full_output = True)
	print "Residual: ", fval
	if x3 > xMax or fval > 0.01 * r2:
		return None, None, None
	r3 = r2 + drdxp2 * (x3 - x2)
	theta3 = math.atan(RWallFunc(x3)[1])
	return x3, r3, theta3

def IntegrateExpansionWallPoint(pm, x2, r2, theta2, nu2, M2, x3, r3, theta3):
	"""
	Along C+ characteristics from P2, integrate C+ compatibility equation
	to obtain condition at downstream wall point P3.
	RWallFunc takes x coordinate of the wall and returns r, and drdx.
	Note: theta3 is known, so nu3 follows from C2+ compatibility relation.
	"""

	mu2 = math.asin(1.0 / M2)
	Kp2 = theta2 - nu2
	drdxp2 = math.tan(theta2 + mu2)

	dKpdr = -1.0 / (math.sqrt(M2 * M2 - 1.0) + 1.0 / math.tan(theta2)) / r2
	Kp3 = Kp2 + dKpdr * (r3 - r2)
	nu3 = theta3 - Kp3

	return x3, r3, theta3, nu3

def IntegrateStraighteningWallPoint(pm, x1, r1, theta1, nu1, M1, x2, r2, theta2, nu2, M2):
	"""
	Along C+ characteristics from P2, integrate C+ compatibility equation
	to obtain condition at downstream wall point P3.
	Note: we impose theta3 = theta2, which results in nu3 != nu2
	in order to satisfy C+ compatibility relation.
	"""

	drdx1 = math.tan(theta1)

	mu2 = math.asin(1.0 / M2)
	Kp2 = theta2 - nu2
	drdxp2 = math.tan(theta2 + mu2)

	x3 = (drdx1 * x1 - drdxp2 * x2 - r1 + r2) / (drdx1 - drdxp2)
	r3 = r1 + drdx1 * (x3 - x1)

	dKpdr = -1.0 / (math.sqrt(M2 * M2 - 1.0) + 1.0 / math.tan(theta2)) / r2
	Kp3 = Kp2 + dKpdr * (r3 - r2)

	theta3 = theta2
	nu3    = theta3 - Kp3

	return x3, r3, theta3, nu3

def IntegrateSymmetryPoint(pm, x1, r1, theta1, nu1, M1):
	"""
	Along C- characteristics from P1 to symmetry point at P2,
	integrate C- compatibility equation to obtain condition at P2.
	"""

	mu1 = math.asin(1.0 / M1)
	Km1 = theta1 + nu1
	drdxm1 = math.tan(theta1 - mu1)

	x2 = x1 - r1 / drdxm1
	r2 = 0.0

	dKmdr =  1.0 / (math.sqrt(M1 * M1 - 1.0) - 1.0 / math.tan(theta1)) / r1

	Km2 = Km1 + dKmdr * (r2 - r1)

	theta2 = 0.0
	nu2    = Km2

	return x2, r2, theta2, nu2

def Main():

	pm = PM(1.4)

	nptsStreamwiseMax = 80
	nptsSonicLine = 10

	print "Inflection point:"
	rInflection, drdxInflection = RExpWall(X_INFLECTION)
	print X_INFLECTION, rInflection, THETA_INFLECTION, pm.Mach(2.0 * THETA_INFLECTION)

	mesh = np.zeros((nptsStreamwiseMax, nptsSonicLine, 5)) # (x, r, theta, nu, mach)

	# Sonic line
	x = 0.0
	r, drdx = RExpWall(x)
	theta = math.atan(drdx)
	r_sonicline = R_THROAT / math.sin(theta)
	x0_sonicline = -R_THROAT / math.tan(theta) # x coord of the center of the sonic line arc
	nu = theta
	M = pm.Mach(nu)
	print x, r, math.degrees(theta), r_sonicline, x0_sonicline, M

	mesh[0, 0, :] = x, r, theta, nu, M
	mesh[0, -1, :] = 0.0, 0.0, 0.0, nu, M
	for j in range(1, nptsSonicLine - 1):
		xi = float(j) / float(nptsSonicLine - 1)
		theta = (1.0 - xi) * mesh[0, 0, 2] + xi * 0.0
		"""
		x = 0.0
		r = (1.0 - xi) * R_THROAT + xi * 0.0
		"""
		x = x0_sonicline + r_sonicline * math.cos(theta)
		r = r_sonicline * math.sin(theta)
		nu, M = mesh[0, 0, 3:5]
		mesh[0, j, :] = x, r, theta, nu, M

	# generic characteristics-tracing loop
	idim, jdim = mesh.shape[0:2]
	nptsStreamwise = nptsStreamwiseMax
	nozzleExit = False
	for i in range(1, idim):
		if i % 2 == 1:
			for j in range(0, jdim - 1):
				x1, r1, theta1, nu1, M1 = mesh[i - 1, j    , :]
				x2, r2, theta2, nu2, M2 = mesh[i - 1, j + 1, :]
				x3, r3, theta3, nu3 = IntegrateInteriorPoint(pm, x1, r1, theta1, nu1, M1, x2, r2, theta2, nu2, M2)
				M3 = pm.Mach(nu3)
				mesh[i, j, :] = x3, r3, theta3, nu3, M3
		else:
			# Wall point
			j = 0
			x2, r2, theta2, nu2, M2 = mesh[i - 1, j, :]
			x3, r3, theta3 = IntersectExpansionWallPoint(x2, r2, theta2, nu2, M2, RExpWall, X_INFLECTION)
			if x3 != None:
				# Expansion section
				x3, r3, theta3, nu3 = IntegrateExpansionWallPoint(pm, x2, r2, theta2, nu2, M2, x3, r3, theta3)
				M3 = pm.Mach(nu3)
				mesh[i, j, :] = x3, r3, theta3, nu3, M3
			else:
				# Straightening section
				x1, r1, theta1, nu1, M1 = mesh[i - 2, j, :]
				x3, r3, theta3, nu3 = IntegrateStraighteningWallPoint(pm, x1, r1, theta1, nu1, M1, x2, r2, theta2, nu2, M2)
				M3 = pm.Mach(nu3)
				mesh[i, j, :] = x3, r3, theta3, nu3, M3
				if theta3 < math.radians(THETA_EXIT):
					nozzleExit = True
			print " %d: Nozzle wall Mach = %f at x = %f, r = %f, theta = %f deg" % (i, mesh[i, j, 4], x3, r3, math.degrees(theta3))

			# Interior points
			for j in range(1, jdim - 1):
				x1, r1, theta1, nu1, M1 = mesh[i - 1, j - 1, :]
				x2, r2, theta2, nu2, M2 = mesh[i - 1, j    , :]
				x3, r3, theta3, nu3 = IntegrateInteriorPoint(pm, x1, r1, theta1, nu1, M1, x2, r2, theta2, nu2, M2)
				M3 = pm.Mach(nu3)
				mesh[i, j, :] = x3, r3, theta3, nu3, M3

			# Symmetry point
			j = jdim - 1
			x1, r1, theta1, nu1, M1 = mesh[i - 1, j - 1, :]
			x2, r2, theta2, nu2 = IntegrateSymmetryPoint(pm, x1, r1, theta1, nu1, M1)
			M2 = pm.Mach(nu2)
			mesh[i, j, :] = x2, r2, theta2, nu2, M2
			print " %d: Symmetry line Mach = %f at x = %f" % (i, M2, x2)
			if M2 > MACH_EXIT:
				nozzleExit = True
		if nozzleExit:
			nptsStreamwise = i
			break

	for j in range(nptsSonicLine):
		if j < nptsSonicLine - 1:
			xx, rr = [], []
			for i in range(0, nptsStreamwise, 2):
				xx.append(mesh[i, j, 0])
				rr.append(mesh[i, j, 1])
				xx.append(mesh[i + 1, j, 0])
				rr.append(mesh[i + 1, j, 1])
			plt.plot(xx, rr)
		if j > 0:
			xx, rr = [], []
			for i in range(0, nptsStreamwise, 2):
				xx.append(mesh[i, j, 0])
				rr.append(mesh[i, j, 1])
				xx.append(mesh[i + 1, j - 1, 0])
				rr.append(mesh[i + 1, j - 1, 1])
			plt.plot(xx, rr)

	"""
	for j in range(nptsSonicLine):
		jj = j
		if j < nptsSonicLine - 1:
			xr = []
			xr.append(mesh[i, j, 0:2])
			dj = 0
			for i in range(1, nptsStreamwise):
				dj = 0 if i % 2 == 1 else 1
				xr.append(mesh[i, jj + dj, 0:2])
	"""

	xx = np.linspace(0.0, X_INFLECTION, 10)
	rr = []
	for i in range(10):
		rr.append(RExpWall(xx[i])[0])
	plt.plot(xx, rr)

	xr = []
	for i in range(0, nptsStreamwise, 2):
		if i % 2 == 1:
			continue
		xr.append(mesh[i, 0, 0:2])
	xr = np.array(xr)
	np.savetxt("nozzle_wall.csv", xr, delimiter = ",")
	plt.plot(xr[:, 0], xr[:, 1], color = "black", linewidth = 2)

	plt.axes().set_aspect("equal")
	plt.show()

if __name__ == "__main__":
	Main()

