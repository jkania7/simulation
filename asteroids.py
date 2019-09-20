#!/usr/bin/env python3
import numpy as np
from numba import jit 
import matplotlib.pyplot as plt
import matplotlib as mpl 

from matplotlib import rc #https://matplotlib.org/3.1.1/tutorials/text/usetex.html
rc('text', usetex=True)
mpl.rcParams['agg.path.chunksize'] = 10000 #https://github.com/matplotlib/matplotlib/issues/5907

@jit#use jit for speed up 
def nbody():
	a = 5.2 #SET initial conditions
	P = a**(3.0/2.0)
	print("I've calculated P = {0:.2f}".format(P))
	GM_dot = 4.0*np.pi**2.0
	x_0J = 5.2 #[AU]
	y_0J = 0.0
	v_x0J = 0.0 #[AU/yr]
	x_0J = 5.2 #[AU]
	y_0J = 0.0
	v_x0J = 0.0 #[AU/yr]
	v_y0J = np.sqrt(GM_dot/a) #[AU/yr]
	time = 60000 #lenght of simulation [year]
	dt = 0.01 #timestep [year]
	numAsteriods = 1000
	furthestAsteriod = a #[AU]s
	closestAsteriod = 1.5
	sjmr = 0.009765 #mass ratio of super jupiter to sun
	
	#end user options
	numStep = int(time/dt)#number of steps in simulation
	#print(f"I'm taking {numStep} steps")
	t = np.arange(start=0, stop=time+dt, step=dt, dtype=float)#need to add stepsize since stop in noninclusive
	#t = np.arange(2, dtype=float)

	v_xJ = np.zeros(numStep+1,dtype=float)#declare jupiter velocities
	v_yJ = np.zeros(numStep+1, dtype=float)
	#v_xJ = np.zeros(2,dtype=float)#declare jupiter velocities
	#v_yJ = np.zeros(2, dtype=float)#the depth 2 arrays are much more memory efficent, but they dod't seem to work well
	v_yJ[0] = v_y0J

	xJ = np.zeros(numStep+1, dtype=float)#declare jupiter positions
	xJ[0] = x_0J
	yJ = np.zeros(numStep+1, dtype=float)
	rJ = np.zeros(numStep+1, dtype=float)

	#xJ = np.zeros(2, dtype=float)#declare jupiter positions
	#xJ[0] = x_0J
	#yJ = np.zeros(2, dtype=float)
	#rJ = np.zeros(2, dtype=float)

	rJ[0] = np.sqrt(xJ[0]**2.0+yJ[0]**2.0)

	
	v_xA = np.zeros((numStep+1,numAsteriods),dtype=float)#initilize astriod positions and velocties
	v_yA = np.zeros((numStep+1,numAsteriods),dtype=float)
	xA = np.zeros((numStep+1, numAsteriods),dtype=float)#no jupiter test
	yA = np.zeros((numStep+1, numAsteriods),dtype=float)#no jupiter test
	rA = np.zeros((numStep+1, numAsteriods), dtype=float)
	rA = np.zeros((numStep+1, numAsteriods), dtype=np.single)
	
	#v_xA = np.zeros((2,numAsteriods),dtype=float)#initilize astriod positions and velocties
	#v_yA = np.zeros((2,numAsteriods),dtype=float)#again memory optimized,but gives weired results
	#xA = np.zeros((2, numAsteriods),dtype=float)#normal
	#yA = np.zeros((2, numAsteriods),dtype=float)#normal
	#rA = np.zeros((2, numAsteriods), dtype=float)
	
	#put astriods in equal steps away from the sun, give them velocites for circular orbits
	asteriodStep = (furthestAsteriod-closestAsteriod)/numAsteriods#eqaul spaced 
	for i,x in enumerate(np.arange(closestAsteriod,furthestAsteriod,asteriodStep,dtype=float)):
		xA[0,i] = x
		v_yA[0,i] =  np.sqrt(GM_dot/x)
	#print("xA[:,0]={0}".format(xA[:,0]))
	#print("v_yA[:,0]={0}".format(v_yA[:,0]))
	rA[0,:] = np.sqrt(xA[0,:]**2+yA[0,:]**2)
	rAStart = rA
	print("rA[0,:] inital = {0}".format(rA[0,:]))
	
	print("Running Euler-Cromer")
	for i in range(0, numStep):#Euler-Cromer it
		#print("loop = {0}".format(i))
		
		v_xJ[i+1] = v_xJ[i] - (GM_dot*xJ[i]*dt)/(rJ[i])**3#calculate jupiter velocities
		v_yJ[i+1] = v_yJ[i] - (GM_dot*yJ[i]*dt)/(rJ[i]**3)
		xJ[i+1] = xJ[i]+v_xJ[i+1]*dt#calculate jupiter positions
		yJ[i+1] = yJ[i]+v_yJ[i+1]*dt
		rJ[i+1] = np.sqrt(xJ[i+1]**2+yJ[i+1]**2)
		
		#print(xA[i,:])
		#print(v_xA[i,:] - (GM_dot*xA[i,:]*dt)/(rA[i,:])**3)
		v_xA[i+1,:] = v_xA[i,:] - (GM_dot*xA[i,:]*dt)/(rA[i,:])**3#calculate asteroid velocities
		v_yA[i+1,:] = v_yA[i,:] - (GM_dot*yA[i,:]*dt)/(rA[i,:]**3)
		xA[i+1,:] = xA[i,:]+v_xA[i+1,:]*dt#calculate asteroid positions
		yA[i+1,:] = yA[i,:]+v_yA[i+1,:]*dt
		rA[i+1,:] = np.sqrt(xA[i+1,:]**2+yA[i+1,:]**2)
		#print("rA[i,:]={0}".format(rA[i,:]))
		
		d_x=xJ[i]-xA[i]
		d_y=yJ[i]-yA[i]
		d = np.sqrt(d_x**2+d_y**2)
		#print(d_x)
		v_xA[i+1,:] = v_xA[i,:] - (GM_dot*xA[i,:]*dt)/(rA[i,:])**3 + sjmr*GM_dot*(d_x)*dt/d**3#calculate asteroid velocities
		v_yA[i+1,:] = v_yA[i,:] - (GM_dot*yA[i,:]*dt)/(rA[i,:]**3) + sjmr*GM_dot*(d_y)*dt/d**3
		xA[i+1,:] = xA[i,:]+v_xA[i+1,:]*dt#calculate asteroid positions
		yA[i+1,:] = yA[i,:]+v_yA[i+1,:]*dt
		rA[i+1,:] = np.sqrt(xA[i+1,:]**2+yA[i+1,:]**2)
		"""
		if i>= 1:
			xA[0,:] = xA[1,:]
			yA[0,:] = yA[1,:]
			v_xA[0,:] = v_xA[1,:]
			v_yA[0,:] = v_yA[1,:]
			t[0] = t[1]
			xJ[0] = xJ[1]
			yJ[0] = yJ[1]
			rJ[0] = rJ[1]
			v_xJ[0] = v_xJ[1]
			v_yJ[0] = v_yJ[1]
			rA[0,:] = rA[1,:]

		v_xJ[1] = v_xJ[0] - (GM_dot*xJ[0]*dt)/(rJ[0])**3#calculate jupiter velocities
		v_yJ[1] = v_yJ[0] - (GM_dot*yJ[0]*dt)/(rJ[0]**3)
		xJ[1] = xJ[0]+v_xJ[1]*dt#calculate jupiter positions
		yJ[1] = yJ[0]+v_yJ[1]*dt
		rJ[1] = np.sqrt(xJ[1]**2+yJ[1]**2)

		d_x=xJ[0]-xA[i,:]#no jupiter test
		d_y=yJ[0]-yA[i,:]#no jupiter test
		#d_x=xJ[0]-xA[0,:]#normal
		#d_y=yJ[0]-yA[0,:]#normal
		d = np.sqrt(d_x**2+d_y**2)
		#print(d_x)
		v_xA[1,:] = v_xA[0,:] - (GM_dot*xA[i,:]*dt)/(rA[0,:])**3 + sjmr*GM_dot*(d_x)*dt/d**3#calculate ast vel #no jupiter test
		v_yA[1,:] = v_yA[0,:] - (GM_dot*yA[i,:]*dt)/(rA[0,:]**3) + sjmr*GM_dot*(d_y)*dt/d**3 #no jupiter test
		#v_xA[1,:] = v_xA[0,:] - (GM_dot*xA[0,:]*dt)/(rA[0,:])**3 + sjmr*GM_dot*(d_x)*dt/d**3#calculate asteroid velocities
		#v_yA[1,:] = v_yA[0,:] - (GM_dot*yA[0,:]*dt)/(rA[0,:]**3) + sjmr*GM_dot*(d_y)*dt/d**3
		#xA[1,:] = xA[0,:]+v_xA[1,:]*dt#calculate asteroid positions
		#yA[1,:] = yA[0,:]+v_yA[1,:]*dt
		#rA[1,:] = np.sqrt(xA[1,:]**2+yA[1,:]**2)
		xA[i+1,:] = xA[i,:]+v_xA[1,:]*dt#calculate asteroid positions #no jupiter test
		yA[i+1,:] = yA[i,:]+v_yA[1,:]*dt#no jupiter test
		rA[1,:] = np.sqrt(xA[i+1,:]**2+yA[i+1,:]**2)#no jupiter test
		
		t[1] = t[0]+dt
		"""
	print("rA[-1,:] = {0}".format(rA[-1,:]))	
	finalAsteroidR = rA[-1,:][rA[-1,:]<a]#some asteriods leave the solar system

	#plt.plot(xJ,yJ)#plot position versus time
	#plt.xlabel("X distance Jupiter [AU]")
	#plt.ylabel("Y distance Jupiter [AU]")
	#plt.savefig("xy.png")
	#plt.show()
	#Plt.close()
	
	plt.plot(xA[:,123],yA[:,123])#plot position versus time #spotcheck 
	plt.title(r'Orbit of Astroid \#123', fontsize=17)
	plt.xlabel(r"X distance Asteriod [AU]", fontsize=17)
	plt.ylabel(r"Y distance Asteriod [AU]", fontsize=17)
	plt.savefig("xyA_J.png", dpi=450)
	plt.show()
	plt.close()
	
	#plt.plot(t,rJ)#Energy is conserved
	#plt.xlabel("time [year]")
	#plt.ylabel("Jupiter distance from sun [AU]")
	#plt.savefig('rt.png')
	#plt.show()
	#plt.close()
	
	#plt.plot(rA[-1,:][rA[-1,:]<50])
	plt.plot(rAStart[0,:][rA[-1,:]<5],rA[-1,:][rA[-1,:]<5])
	plt.axvline(x=a/(3.0)**(2.0/3.0), color='r', linestyle='-', linewidth=1, label=r'3:1')
	plt.axvline(x=a/(5.0/2.0)**(2.0/3.0), color='r',linestyle=':', linewidth=1, label=r'5:2')
	plt.axvline(x=a/(7.0/3.0)**(2.0/3.0), color='r',linestyle='--', linewidth=1, label=r'7:3')
	plt.axvline(x=a/(2.0)**(2.0/3.0), color='r', linestyle='-.', linewidth=1, label=r'2:1')
	plt.xlabel(r'Start Radius [AU]', fontsize=17)
	plt.ylabel(r'Radius after {0} yr [AU]'.format(time), fontsize=17)
	plt.title(r'End vs Start Solar Distances', fontsize=17)
	plt.savefig('distances.png', dpi=450)
	plt.close()
	
	plt.hist(finalAsteroidR,bins=40, range=[2.0, 3.5])
	plt.xlabel(r'Asteroid Distance from Sun [AU]', fontsize=17)
	plt.ylabel(r'Number of Asteroids', fontsize=17)
	plt.axvline(x=a/(3.0)**(2.0/3.0), color='r', linestyle='-', linewidth=1, label=r'3:1')
	plt.axvline(x=a/(5.0/2.0)**(2.0/3.0), color='r',linestyle=':', linewidth=1, label=r'5:2')
	plt.axvline(x=a/(7.0/3.0)**(2.0/3.0), color='r',linestyle='--', linewidth=1, label=r'7:3')
	plt.axvline(x=a/(2.0)**(2.0/3.0), color='r', linestyle='-.', linewidth=1, label=r'2:1')
	plt.legend()
	plt.xticks(np.arange(2.0, 3.6, step=0.1))
	plt.title(r'Asteroid Distribution', fontsize=17)
	plt.savefig('histo.png', dpi=450)
	plt.close()
	

nbody() 
