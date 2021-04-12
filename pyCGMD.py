#!/usr/bin/env python
from chemfiles import Trajectory
import numpy as np
import numba as nb
from tqdm import tqdm
from interpolation.splines import LinearSpline, CubicSpline
from interpolation import interp
import math
# unit:
# length: nm
# time: picosecond
# energy: kJ/mol

kB=0.00831445986144858

particle = np.dtype({'names':['m','x','y','z'], 
                             'formats':[np.float32, 
                                        np.float32, 
                                        np.float32, 
                                        np.float32]})

bond = np.dtype({'names':['atom_1','atom_2'], 
                             'formats':[np.int32, 
                                        np.int32]})

angle = np.dtype({'names':['atom_1','atom_2','atom_3'], 
                             'formats':[np.int32, 
                                        np.int32,
                                        np.int32]})

velocity_particle=np.dtype({'names':['vx','vy','vz'], 
                             'formats':[np.float32, 
                                        np.float32, 
                                        np.float32]})
image_particle=np.dtype({'names':['ix','iy','iz'], 
                             'formats':[np.int32, 
                                        np.int32, 
                                        np.int32]})

force_particle=np.dtype({'names':['fx','fy','fz'], 
                             'formats':[np.float32, 
                                        np.float32, 
                                        np.float32]})
thermostat=np.dtype({'names':['bond_eng','angle_eng','pair_eng','virial'], 
                             'formats':[np.float32, 
                                        np.float32, 
                                        np.float32, 
                                        np.float32]})
bondtable=np.dtype({'names':['index','d','v','f'], 
                             'formats':[np.int32, 
                                        np.float32, 
                                        np.float32,
                                        np.float32]})
angletable=np.dtype({'names':['index','rad','v','f'], 
                             'formats':[np.int32, 
                                        np.float32, 
                                        np.float32,
                                        np.float32]})
pairtable=np.dtype({'names':['index','d','v','f'], 
                             'formats':[np.int32, 
                                        np.float32, 
                                        np.float32,
                                        np.float32]})

def chemfile_read(filename,format_='LAMMPS Data'):
    data = Trajectory(filename,format=format_)
    frame=data.read()
    topology=frame.topology
    numberatom_permole=int(len(frame.atoms))
    atoms=frame.atoms
    bonds=topology.bonds[0:int(topology.bonds.shape[0])]
    angles=topology.angles[0:int(topology.angles.shape[0])]
    dihedrals=topology.dihedrals[0:int(topology.dihedrals.shape[0])]

    particles_=np.zeros(len(atoms),dtype=particle)
    for atomii in range(len(atoms)):
        particles_[atomii]['m']=atoms[atomii].mass
        particles_[atomii]['x']=frame.positions[atomii][0]/10
        particles_[atomii]['y']=frame.positions[atomii][1]/10
        particles_[atomii]['z']=frame.positions[atomii][2]/10

    bonds_=np.zeros(len(bonds),dtype=bond)
    for bondii in range(len(bonds)):
        bonds_[bondii]['atom_1']=bonds[bondii][0]
        bonds_[bondii]['atom_2']=bonds[bondii][1]

    angles_=np.zeros(len(angles),dtype=angle)
    for angleii in range(len(angles)):
        angles_[angleii]['atom_1']=angles[angleii][0]
        angles_[angleii]['atom_2']=angles[angleii][1]
        angles_[angleii]['atom_3']=angles[angleii][2]

    return particles_,bonds_,angles_
#
#chemfile_read('../PS.data')

def sample_normal_boxmuller(num,mu=0,sigma=1):
    """Sample from a normal distribution using Box-Muller method.
    See exercise sheet on sampling and motion models.
    """
    # Two uniform random variables
    u1 = np.random.uniform(0, 1, size=num)
    u2 = np.random.uniform(0, 1, size=num)
    # Box-Muller formula returns sample from STANDARD normal distribution
    x = np.cos(2*np.pi*u1) * np.sqrt(-2*np.log(u2))
    return mu+sigma * x

def create_vel_particles(particles, temp=0):
    '''
    Creates `n` velocities of particles with gaussian distribution of 
    reduced temperature: temp
    '''
    num_particles=particles.shape[0]
    vel_particles = np.zeros((num_particles), dtype=velocity_particle)
    vel_particles['vx'] = np.sqrt(temp/particles['m'])*sample_normal_boxmuller(num_particles)
    vel_particles['vy'] = np.sqrt(temp/particles['m'])*sample_normal_boxmuller(num_particles)
    vel_particles['vz'] = np.sqrt(temp/particles['m'])*sample_normal_boxmuller(num_particles)
    return vel_particles

def create_image_particles(particles):
    '''
    Creates `n` images of particles
    '''
    num_particles=particles.shape[0]
    image_particles = np.zeros(num_particles, dtype=image_particle)
    return image_particles

def create_force_particles(particles):
    '''
    Creates `n` forces
    '''
    num_particles=particles.shape[0]
    force_particles = np.zeros((num_particles), dtype=force_particle)
    return force_particles

def read_bondtable(file=None):
    bondtable_=np.loadtxt(file,dtype=bondtable,delimiter="\t")
    return bondtable_

def read_angletable(file=None):
    angletable_=np.loadtxt(file,dtype=angletable,delimiter=" ")
    return angletable_

def read_pairtable(file=None):
    pairtable_=np.loadtxt(file,dtype=pairtable,delimiter=" ")
    return pairtable_

@nb.jit(nopython=True)
def calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_d,thermostat):

    for bondii in bonds:
        dx=particles[bondii['atom_1']]['x']-particles[bondii['atom_2']]['x']
        dy=particles[bondii['atom_1']]['y']-particles[bondii['atom_2']]['y']
        dz=particles[bondii['atom_1']]['z']-particles[bondii['atom_2']]['z']
        
        dx-=box[0]*np.rint(dx/box[0])
        dy-=box[1]*np.rint(dy/box[1])
        dz-=box[2]*np.rint(dz/box[2])
        rsq=dx*dx+dy*dy+dz*dz
        r=np.sqrt(dx*dx+dy*dy+dz*dz)
        if r>bondtable['d'][199]:
            print(bondii,"ERROR: Bond Break")

       
        index_bond=int(r/delta_d)
        force_=bondtable['f'][index_bond]
        force_divr = force_/r
        thermostat[0]+=bondtable['v'][index_bond]

        thermostat[3]+=1/6*2*rsq*force_divr

        force_particles[bondii['atom_1']]['fx'] += dx * force_divr
        force_particles[bondii['atom_1']]['fy'] += dy * force_divr
        force_particles[bondii['atom_1']]['fz'] += dz * force_divr

        force_particles[bondii['atom_2']]['fx'] += -dx * force_divr
        force_particles[bondii['atom_2']]['fy'] += -dy * force_divr
        force_particles[bondii['atom_2']]['fz'] += -dz * force_divr

@nb.jit(nopython=True)
def calc_angle_table(table,force_particles,angles,box,particles,delta_angle,thermostat):
    for angleii in angles:
        atomi=angleii['atom_1']
        atomj=angleii['atom_2']
        atomk=angleii['atom_3']

        delx1=particles[atomi]['x']-particles[atomj]['x']
        dely1=particles[atomi]['y']-particles[atomj]['y']
        delz1=particles[atomi]['z']-particles[atomj]['z']

        delx2=particles[atomk]['x']-particles[atomj]['x']
        dely2=particles[atomk]['y']-particles[atomj]['y']
        delz2=particles[atomk]['z']-particles[atomj]['z']

        delx1-=box[0]*np.rint(delx1/box[0])
        dely1-=box[1]*np.rint(dely1/box[1])
        delz1-=box[2]*np.rint(delz1/box[2])

        rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1
        r1 = np.sqrt(rsq1)

        delx2-=box[0]*np.rint(delx2/box[0])
        dely2-=box[1]*np.rint(dely2/box[1])
        delz2-=box[2]*np.rint(delz2/box[2])

        rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2
        r2 = np.sqrt(rsq2)      

        #angle (cos and sin)

        c = delx1*delx2 + dely1*dely2 + delz1*delz2
        c /= r1*r2

        if c > 1.0:
            c = 1.0
        if c < -1.0:
            c = -1.0
        
        s = np.sqrt(1.0 - c*c)

        theta_ijk = np.arccos(c)

        point=int(theta_ijk/delta_angle)

        force=table['f'][point]

        thermostat[1]+=table['v'][point]

        a = force * s
        a11 = a*c / rsq1
        a12 = -a / (r1*r2)
        a22 = a*c / rsq2

        f1=np.zeros(3)
        f3=np.zeros(3)

        f1[0] = a11*delx1 + a12*delx2
        f1[1] = a11*dely1 + a12*dely2
        f1[2] = a11*delz1 + a12*delz2
        f3[0] = a22*delx2 + a12*delx1
        f3[1] = a22*dely2 + a12*dely1
        f3[2] = a22*delz2 + a12*delz1

        vx=delx1*f1[0]+delx2*f3[0]
        vy=dely1*f1[1]+dely2*f3[1]
        vz=delz1*f1[2]+delz2*f3[2]

        thermostat[3]+=3*1/6*(vx+vy+vz)

        force_particles[atomi]['fx'] += f1[0]
        force_particles[atomi]['fy'] += f1[1]
        force_particles[atomi]['fz'] += f1[2]

        force_particles[atomj]['fx'] -= f1[0]+f3[0]
        force_particles[atomj]['fy'] -= f1[1]+f3[1]
        force_particles[atomj]['fz'] -= f1[2]+f3[2]

        force_particles[atomk]['fx'] += f3[0]
        force_particles[atomk]['fy'] += f3[1]
        force_particles[atomk]['fz'] += f3[2]

@nb.jit(nopython=True)
def calc_pair_table(force_particles,box,neighborlist,particles,pairtable,deltad,thermostat):
    num_particles=particles.shape[0]
    for atomii in range(num_particles):
        neighlist_this=neighborlist[atomii]
        for atomjj in neighlist_this:
            if atomjj==-1.:
                continue
            atomjj=int(atomjj)

            dx=particles[atomii]['x']-particles[atomjj]['x']
            dy=particles[atomii]['y']-particles[atomjj]['y']
            dz=particles[atomii]['z']-particles[atomjj]['z']
            
            dx-=box[0]*np.rint(dx/box[0])
            dy-=box[1]*np.rint(dy/box[1])
            dz-=box[2]*np.rint(dz/box[2])
            force_divr=0
            rsq=dx*dx+dy*dy+dz*dz
            r=np.sqrt(dx*dx+dy*dy+dz*dz)

            point=int(r/deltad)
            
            if r>pairtable['d'][-1]:
                #print(r,point)
                force_divr=0
                thermostat[2]+=0
                thermostat[3] += 1.0/6.0 * rsq * force_divr

            else:

                force_divr= pairtable['f'][point]/r
                thermostat[2]+=pairtable['v'][point]
                thermostat[3] += 1.0/6.0 * rsq * force_divr
                #force_divr=0
            if r!=0:
                force_particles[atomii]['fx'] += dx * force_divr
                force_particles[atomii]['fy'] += dy * force_divr
                force_particles[atomii]['fz'] += dz * force_divr

@nb.jit(nopython=True)
def calc_TotalKineticEnergy(particles,vel_particles):
    num_particles=particles.shape[0]
    K_tot=0
    for atomii in range(num_particles):
        K_x=0.5*particles[atomii]['m']*np.power(vel_particles[atomii]['vx'],2)
        K_y=0.5*particles[atomii]['m']*np.power(vel_particles[atomii]['vy'],2)
        K_z=0.5*particles[atomii]['m']*np.power(vel_particles[atomii]['vz'],2)
        K_tot+=(K_x+K_y+K_z)
    return K_tot


def calc_InstaneousTmep(particles,TotalKineticEnergy,kB=1):
    num_particles=particles.shape[0]
    dim=3
    return TotalKineticEnergy*2/(kB*(dim*num_particles-dim))

def NVT_BerendsenDynamics_firststep(particle_images,box,particles,vel_particles,force_particles,dt=0,temp=0):
    
    particles['x'] += (vel_particles['vx']+0.5 * force_particles['fx']/particles['m'] * dt)*dt
    particles['y'] += (vel_particles['vy']+0.5 * force_particles['fy']/particles['m'] * dt)*dt
    particles['z'] += (vel_particles['vz']+0.5 * force_particles['fz']/particles['m'] * dt)*dt

    vel_particles['vx'] += 0.5*force_particles['fx']/particles['m']*dt
    vel_particles['vy'] += 0.5*force_particles['fy']/particles['m']*dt
    vel_particles['vz'] += 0.5*force_particles['fz']/particles['m']*dt 

    particle_images['ix'] += np.rint(particles['x']/box[0]).astype(int)
    particle_images['iy'] += np.rint(particles['y']/box[1]).astype(int)
    particle_images['iz'] += np.rint(particles['z']/box[2]).astype(int)

    particles['x'] -= box[0]*np.rint(particles['x']/box[0])
    particles['y'] -= box[1]*np.rint(particles['y']/box[1])
    particles['z'] -= box[2]*np.rint(particles['z']/box[2])
    
#@nb.jit
def NVT_BerendsenDynamics_secondstep(step,particles,vel_particles,force_particles,tau=0,dt=0,temp=0):
    
    curr_T=calc_InstaneousTmep(particles,calc_TotalKineticEnergy(particles,vel_particles))
    scalT  = np.sqrt(1.0+dt*(temp/curr_T-1.0)/tau)

    vel_particles['vx'] = ( vel_particles['vx']+0.5*force_particles['fx']/particles['m']*dt)*scalT
    vel_particles['vy'] = ( vel_particles['vy']+0.5*force_particles['fy']/particles['m']*dt)*scalT
    vel_particles['vz'] = ( vel_particles['vz']+0.5*force_particles['fz']/particles['m']*dt)*scalT

def output_Trj(f,particle_images,timestep,box,particles,type_=None):
    if type_=="xyz":
        np.savetxt(f, particles, fmt='%.4f',delimiter=' ', header=str(particles.shape[0])+'\n', comments='')
    if type_=="lammpscustom":
        f.write('ITEM: TIMESTEP\n')
        f.write(str(timestep)+'\n')
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write(str(particles.shape[0])+'\n')
        f.write('ITEM: BOX BOUNDS pp pp pp\n')
        f.write('0 '+str(box[0]*10)+'\n')
        f.write('0 '+str(box[1]*10)+'\n')
        f.write('0 '+str(box[2]*10)+'\n')
        f.write('ITEM: ATOMS id type x y z ix iy iz\n')
        for i in range(particles.shape[0]):
            f.write(str(i+1)+' 1 '+str(particles[i]['x']*10)+' '+str(particles[i]['y']*10)+' '+str(particles[i]['z']*10)+' '+str(particle_images[i]['ix'])+' '+str(particle_images[i]['iy'])+' '+str(particle_images[i]['iz'])+'\n')

def output_thermo(timestep,f,thermostat,kinetic_eng,curr_temp):
    f.write(str(int(timestep))+' '+str(curr_temp)+' '+str(thermostat[0])+' '+str(thermostat[1])+' '+str(thermostat[2])+' '+str(kinetic_eng)+"\n")

def create_angle_excludelist(angles,particles):
    num_particles=particles.shape[0]
    
    excludelist=[]
    for i in range(num_particles):
        excludelist.append([])
    for angleii in angles:
        atom_1=int(angleii['atom_1'])
        atom_2=int(angleii['atom_2'])
        atom_3=int(angleii['atom_3'])

        if atom_2 not in excludelist[atom_1]:
            excludelist[atom_1].append(atom_2)

        if atom_3 not in excludelist[atom_1]:
            excludelist[atom_1].append(atom_3)

        if atom_1 not in excludelist[atom_2]:
            excludelist[atom_2].append(atom_1)

        if atom_3 not in excludelist[atom_2]:
            excludelist[atom_2].append(atom_3)
        
        if atom_2 not in excludelist[atom_3]:
            excludelist[atom_3].append(atom_2)

        if atom_1 not in excludelist[atom_3]:
            excludelist[atom_3].append(atom_1)

    excludelist_array=[]
    for exii in excludelist:
        while len(exii)<4:
            exii.append(-1)
        excludelist_array.append(np.array(exii))

    return np.array(excludelist_array)

@nb.jit(nopython=True)
def neighbor_list_angle(neighborlist,particles,box,neigh_d,exlude):
    num_particles=particles.shape[0]
    #neighborlist=np.zeros([num_particles,100])
    for atomii in range(num_particles):
        x_this=particles['x'][atomii]
        y_this=particles['y'][atomii]
        z_this=particles['z'][atomii]
        #neighlist_this=np.zeros(100,dtype=np.int32)
        neighlist_this=neighborlist[atomii,:]
        index=0
        atomii_exclude=exlude[atomii]
        for atomjj in range(num_particles):
            
            if atomii==atomjj:
                continue
                
            if (atomii_exclude[0]==atomjj) or (atomii_exclude[1]==atomjj) or (atomii_exclude[2]==atomjj) or (atomii_exclude[3]==atomjj):
                continue

            x_that=particles['x'][atomjj]
            y_that=particles['y'][atomjj]
            z_that=particles['z'][atomjj]

            dx=x_this-x_that
            dy=y_this-y_that
            dz=z_this-z_that
            
            dx-=box[0]*np.rint(dx/box[0])
            dy-=box[1]*np.rint(dy/box[1])
            dz-=box[2]*np.rint(dz/box[2])
            
            r=np.sqrt(dx**2+dy**2+dz**2)

            if r<neigh_d:
                neighlist_this[index]=atomjj
                index=index+1
        neighborlist[atomii]=neighlist_this

@nb.jit(nopython=True)
def neighbor_list_virtual(neighborlist,particles,box,neigh_d,min_dist,max_dist):
    num_particles=particles.shape[0]
    #neighborlist=np.zeros([num_particles,100])
    for atomii in range(num_particles):
        x_this=particles['x'][atomii]
        y_this=particles['y'][atomii]
        z_this=particles['z'][atomii]
        #neighlist_this=np.zeros(100,dtype=np.int32)
        neighlist_this=neighborlist[atomii,:]
        index=0
        for atomjj in range(num_particles):
            
            if atomii==atomjj:
                continue

            x_that=particles['x'][atomjj]
            y_that=particles['y'][atomjj]
            z_that=particles['z'][atomjj]

            dx=x_this-x_that
            dy=y_this-y_that
            dz=z_this-z_that
            
            dx-=box[0]*np.rint(dx/box[0])
            dy-=box[1]*np.rint(dy/box[1])
            dz-=box[2]*np.rint(dz/box[2])
            
            r=np.sqrt(dx**2+dy**2+dz**2)

            if r>min_dist and r < max_dist: 
                neighlist_this[index]=atomjj
                index=index+1
        neighborlist[atomii]=neighlist_this

@nb.jit(nopython=True)
def calc_Forces_table_virtual(num_sites,cutoff,force_particles,box,neighborlist,particles,table,deltad,force_softer):
    num_particles=particles.shape[0]
    for atomii in range(num_particles):
        neighlist_this=neighborlist[atomii]
        length_neigh=int(neighlist_this.shape[0])
        site_index=0

        while site_index<num_sites:
            chosen_index=int(np.random.rand()*length_neigh)
            atomjj=int(neighlist_this[chosen_index])
            #print(atomii,atomjj)
            if atomjj==-1:
                continue
            #print(atomjj)
            atomjj=int(atomjj)
            dx=particles[atomii]['x']-particles[atomjj]['x']
            dy=particles[atomii]['y']-particles[atomjj]['y']
            dz=particles[atomii]['z']-particles[atomjj]['z']
            
            dx-=box[0]*np.rint(dx/box[0])
            dy-=box[1]*np.rint(dy/box[1])
            dz-=box[2]*np.rint(dz/box[2])

            r=np.sqrt(dx*dx+dy*dy+dz*dz)
            
            point=int((r-cutoff)/deltad)
            #print(r)
            if point>table['index'][-1]:
                force_divr=0
            else:
                force_divr= table['f'][point]*force_softer/r
            #force_divr=0
            force_particles[atomii]['fx'] += dx * force_divr
            force_particles[atomii]['fy'] += dy * force_divr
            force_particles[atomii]['fz'] += dz * force_divr

            force_particles[atomjj]['fx'] -= dx * force_divr
            force_particles[atomjj]['fy'] -= dy * force_divr
            force_particles[atomjj]['fz'] -= dz * force_divr

            site_index=site_index+1

@nb.jit(nopython=True)
def calc_Forces_table_virtual(num_sites,cutoff,force_particles,box,neighborlist,particles,table,deltad,force_softer):
    num_particles=particles.shape[0]
    for atomii in range(num_particles):
        neighlist_this=neighborlist[atomii]
        length_neigh=int(neighlist_this.shape[0])
        site_index=0

        while site_index<num_sites:
            chosen_index=int(np.random.rand()*length_neigh)
            atomjj=int(neighlist_this[chosen_index])
            #print(atomii,atomjj)
            if atomjj==-1:
                continue
            #print(atomjj)
            atomjj=int(atomjj)
            dx=particles[atomii]['x']-particles[atomjj]['x']
            dy=particles[atomii]['y']-particles[atomjj]['y']
            dz=particles[atomii]['z']-particles[atomjj]['z']
            
            dx-=box[0]*np.rint(dx/box[0])
            dy-=box[1]*np.rint(dy/box[1])
            dz-=box[2]*np.rint(dz/box[2])

            r=np.sqrt(dx*dx+dy*dy+dz*dz)
            
            point=int((r-cutoff)/deltad)
            #print(r)
            if point>table['index'][-1]:
                force_divr=0
            else:
                force_divr= table['f'][point]*force_softer/r
            #force_divr=0
            force_particles[atomii]['fx'] += dx * force_divr
            force_particles[atomii]['fy'] += dy * force_divr
            force_particles[atomii]['fz'] += dz * force_divr

            force_particles[atomjj]['fx'] -= dx * force_divr
            force_particles[atomjj]['fy'] -= dy * force_divr
            force_particles[atomjj]['fz'] -= dz * force_divr

            site_index=site_index+1

@nb.jit(nopython=True)
def calc_Forces_table_virtual_meanfield(num_sites,force_particles,box,particles,table,deltad,force_softer,L_max,L_cut,exlude,break_dist):
    num_particles=particles.shape[0]
    for atomii in range(num_particles):
        length_neigh=num_particles
        site_index=0
        atomii_exclude=exlude[atomii]
        while site_index<num_sites:
            chosen_index=int(np.random.rand()*length_neigh)
            atomjj=int(chosen_index)
            #print(atomii,atomjj)
            if atomjj==atomii:
                continue
            if (atomii_exclude[0]==atomjj) or (atomii_exclude[1]==atomjj) or (atomii_exclude[2]==atomjj) or (atomii_exclude[3]==atomjj):
                continue
            dx=particles[atomii]['x']-particles[atomjj]['x']
            dy=particles[atomii]['y']-particles[atomjj]['y']
            dz=particles[atomii]['z']-particles[atomjj]['z']
            
            dx-=box[0]*np.rint(dx/box[0])
            dy-=box[1]*np.rint(dy/box[1])
            dz-=box[2]*np.rint(dz/box[2])

            r=np.sqrt(dx*dx+dy*dy+dz*dz)    
            
            if r > L_cut:
                L_ij=(np.random.rand()*(L_max/L_cut-1)+1)*L_cut
                if np.abs(r-L_ij)>break_dist:
                    point=int(np.abs(r-L_ij)/deltad)
                    #print(r)
                    if point>table['index'][-1]:
                        force_divr=0
                    else:
                        force_divr= table['f'][point]*force_softer/r
                    #force_divr=0
                    force_particles[atomii]['fx'] += dx * force_divr
                    force_particles[atomii]['fy'] += dy * force_divr
                    force_particles[atomii]['fz'] += dz * force_divr

                    force_particles[atomjj]['fx'] -= dx * force_divr
                    force_particles[atomjj]['fy'] -= dy * force_divr
                    force_particles[atomjj]['fz'] -= dz * force_divr

                    site_index=site_index+1

def interpolation(start,end,deltad,table):
    f=open('nbAA_new.pot.table','w')
    num_point=int((end-start)/deltad)
    table_new=np.zeros([num_point,4])
    x=np.arange(start,end,deltad)
    for i in range(num_point):
        table_new[i][3]=interp(table['d'],pairtable['f'],x[i])
        table_new[i][2]=interp(table['d'],pairtable['v'],x[i])
        table_new[i][1]=x[i]
        table_new[i][0]=int(i)
    np.savetxt(f, table_new, fmt='%.8f',delimiter=' ', comments='')
    
#pairtable=read_pairtable('nbAA.pot.table')
#interpolation(0,1.5,0.00001,pairtable)

def main(datafile=None,force_softer=None,runstep=1,temp=1,num_virtualsite=None,themo_freq=None,trj_freq=None,trjfile=None,timestep=None,neigh_update=1,neighbor_distance=None,thermofile=None,box=None,neighborlist_length=300,BondTableFile=None,AngleTableFile=None,PairTableFile=None,init_vel=False):
    f_trj=open(trjfile,"w")
    f_thermo=open(thermofile,"w")

    particles,bonds,angles=chemfile_read(datafile)
    num_particles=particles.shape[0]

    bondtable=read_bondtable(BondTableFile)
    delta_bond=bondtable['d'][1]-bondtable['d'][0]
    angletable=read_angletable(AngleTableFile)
    delta_angle=angletable['rad'][1]-angletable['rad'][0]
    pairtable=read_pairtable(PairTableFile)
    delta_pair=pairtable['d'][1]-pairtable['d'][0]
    #pairforce_table=interpolation(0,pairtable['d'][-1],delta_pair*0.1,pairtable)
    

    if init_vel==True:
        vel_particles=create_vel_particles(particles,temp=temp)

    particle_images=create_image_particles(particles)

    TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)

    temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
    
    ### Initialize thermo data and forces
    thermostat_current=np.zeros(4)
    force_particles = np.zeros(num_particles, dtype=force_particle)

    ### Pair interactions ###
    excludelist=create_angle_excludelist(angles,particles)
    neighlist=np.ones([num_particles,neighborlist_length])*-1
    neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
    calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

    ### Bonded interactions ###
    calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
    calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

    ### Output thermo data
    output_thermo(0,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)

    for runii in tqdm(range(runstep-1)):
        ### First step of integration ###
        NVT_BerendsenDynamics_firststep(particle_images,box,particles,vel_particles,force_particles,dt=timestep,temp=temp)

        ### Initialize thermo data and forces
        thermostat_current=np.zeros(4)
        force_particles = np.zeros(num_particles, dtype=force_particle)

        ### Pair interactions ###
        if runii%neigh_update==0:
            neighlist=np.ones([num_particles,neighborlist_length])*-1
            neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
        calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

        ### Virtual sites ###
        min_dist=neighbor_distance+0.5
        max_dist=neighbor_distance*2

        #neighlist_virtual=np.ones([particles.shape[0],int(neighborlist_length*1.3)])*-1
        #neighbor_list_virtual(neighlist_virtual,particles,box,neighbor_distance,min_dist,max_dist)
        #print(neighlist_virtual)
        #calc_Forces_table_virtual(num_virtualsite,neighbor_distance,force_particles,box,neighlist_virtual,particles,pairtable,delta_pair,force_softer)
        L_max=3.0
        L_cut=1.5
        break_dist=0.45
        calc_Forces_table_virtual_meanfield(num_virtualsite,force_particles,box,particles,pairtable,delta_pair,force_softer,L_max,L_cut,excludelist,break_dist)
        ### Bonded interactions ###
        calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
        calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

        if runii%themo_freq==0:
            ### Output thermo data
            TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)
            temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
            output_thermo(runii,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)

        if runii%trj_freq==0:
            ### Output thermo data
            output_Trj(f_trj,particle_images,runii,box,particles,type_="lammpscustom")

        NVT_BerendsenDynamics_secondstep(runii,particles,vel_particles,force_particles,dt=timestep,tau=timestep*100,temp=temp)

    f_trj.close()
    f_thermo.close()

#main(datafile='./test_4000/PS.data',force_softer=1.0,num_virtualsite=28,runstep=100000,temp=373*kB,themo_freq=1000,trj_freq=100,trjfile='test.dump',thermofile='test.thermo',timestep=0.004,neigh_update=1,neighbor_distance=1.5+0.05,box=(9.14560,9.14560,9.14560),neighborlist_length=400,BondTableFile='bondAA.pot.tablenew',AngleTableFile='angleAAA.pot.tablenew',PairTableFile='nbAA_new.pot.table',init_vel=True)

def create_outtrjtimes(blocksize,blocknum):
    outtrjtimes=[]
    EXP=blocksize-17
    #variable s equal 0
    s=0
    #variable l loop 10  ## NUMBER OF BLOCKS
    #outerloop
    n_prd_blocks=blocknum
    exp_base=1.2
    for l in range(1,n_prd_blocks+1):
        #variable l loop 5
        # ----------------------------------------------------
        # ----------------------------
        ## 1. "PSEUDO-LINEAR"
        # ----------------------------
        for i in range(1,17):
            s=s+1
            outtrjtimes.append(s)
            #print(s)
        # ----------------------------
        ## 2. "BRIDGING"
        # ----------------------------
        m=int(math.floor(exp_base**16)-16)
        s=s+m
        outtrjtimes.append(s)
        #print(s)
        # ----------------------------
        ## 3. "EXPONENTIAL TIMESTEPS"
        # ----------------------------
        for c in range(1,EXP+1):
            m=int(math.floor(exp_base**(c+16))-math.floor(exp_base**(c+16-1)))
            s=s+m
            outtrjtimes.append(s)
            #print(s)
    return outtrjtimes



def main_exponentialscheme(datafile=None,force_softer=None,runstep=1,temp=1,num_virtualsite=None,themo_freq=None,num_trjblock=10,size_trjblock=10,trjfile=None,timestep=None,neigh_update=1,neighbor_distance=None,thermofile=None,box=None,neighborlist_length=300,BondTableFile=None,AngleTableFile=None,PairTableFile=None,init_vel=False):
    f_trj=open(trjfile,"w")
    f_thermo=open(thermofile,"w")
    
    #print(out_trjtimes[-1],out_trjtimes)
    particles,bonds,angles=chemfile_read(datafile)
    num_particles=particles.shape[0]

    bondtable=read_bondtable(BondTableFile)
    delta_bond=bondtable['d'][1]-bondtable['d'][0]
    angletable=read_angletable(AngleTableFile)
    delta_angle=angletable['rad'][1]-angletable['rad'][0]
    pairtable=read_pairtable(PairTableFile)
    delta_pair=pairtable['d'][1]-pairtable['d'][0]
    #pairforce_table=interpolation(0,pairtable['d'][-1],delta_pair*0.1,pairtable)
    

    if init_vel==True:
        vel_particles=create_vel_particles(particles,temp=temp)

    particle_images=create_image_particles(particles)

    TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)

    temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
    
    ### Initialize thermo data and forces
    thermostat_current=np.zeros(4)
    force_particles = np.zeros(num_particles, dtype=force_particle)

    ### Pair interactions ###
    excludelist=create_angle_excludelist(angles,particles)
    neighlist=np.ones([num_particles,neighborlist_length])*-1
    neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
    calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

    ### Bonded interactions ###
    calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
    calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

    ### Output thermo data
    out_trjtimes=create_outtrjtimes(size_trjblock,num_trjblock)
    output_thermo(0,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)
    output_Trj(f_trj,particle_images,0,box,particles,type_="lammpscustom")

    for runii in tqdm(range(out_trjtimes[-1]+1)):
        ### First step of integration ###
        NVT_BerendsenDynamics_firststep(particle_images,box,particles,vel_particles,force_particles,dt=timestep,temp=temp)

        ### Initialize thermo data and forces
        thermostat_current=np.zeros(4)
        force_particles = np.zeros(num_particles, dtype=force_particle)

        ### Pair interactions ###
        if runii%neigh_update==0:
            neighlist=np.ones([num_particles,neighborlist_length])*-1
            neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
        calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

        ### Virtual sites ###

        #min_dist=neighbor_distance+0.5
        #max_dist=neighbor_distance*2

        #neighlist_virtual=np.ones([particles.shape[0],int(neighborlist_length*1.3)])*-1
        #neighbor_list_virtual(neighlist_virtual,particles,box,neighbor_distance,min_dist,max_dist)
        #print(neighlist_virtual)
        #calc_Forces_table_virtual(num_virtualsite,neighbor_distance,force_particles,box,neighlist_virtual,particles,pairtable,delta_pair,force_softer)

        ### Bonded interactions ###
        calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
        calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

        if runii%themo_freq==0:
            ### Output thermo data
            TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)
            temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
            output_thermo(runii,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)

        if runii in out_trjtimes:
            ### Output thermo data
            output_Trj(f_trj,particle_images,runii,box,particles,type_="lammpscustom")

        NVT_BerendsenDynamics_secondstep(runii,particles,vel_particles,force_particles,dt=timestep,tau=timestep*100,temp=temp)

    f_trj.close()
    f_thermo.close()

#main_exponentialscheme(datafile='../PS.data',force_softer=0.5,num_virtualsite=0,runstep=100000,temp=373*kB,themo_freq=1000,num_trjblock=2,size_trjblock=20,trjfile='test.dump',thermofile='test.thermo',timestep=0.004,neigh_update=1,neighbor_distance=1.5+0.1,box=(5.4105963,5.4105963,5.4105963),neighborlist_length=300,BondTableFile='bondAA.pot.tablenew',AngleTableFile='angleAAA.pot.tablenew',PairTableFile='nbAA_new.pot.table',init_vel=True)

def main_equilibration_exponentialscheme(datafile=None,equilibration_steps=None,force_softer=None,runstep=1,temp=1,num_virtualsite=None,themo_freq=None,num_trjblock=10,size_trjblock=10,trjfile=None,timestep=None,neigh_update=1,neighbor_distance=None,thermofile=None,box=None,neighborlist_length=300,BondTableFile=None,AngleTableFile=None,PairTableFile=None,init_vel=False):
    f_trj=open(trjfile,"w")
    f_thermo=open(thermofile,"w")
    
    #print(out_trjtimes[-1],out_trjtimes)
    particles,bonds,angles=chemfile_read(datafile)
    num_particles=particles.shape[0]

    bondtable=read_bondtable(BondTableFile)
    delta_bond=bondtable['d'][1]-bondtable['d'][0]
    angletable=read_angletable(AngleTableFile)
    delta_angle=angletable['rad'][1]-angletable['rad'][0]
    pairtable=read_pairtable(PairTableFile)
    delta_pair=pairtable['d'][1]-pairtable['d'][0]
    #pairforce_table=interpolation(0,pairtable['d'][-1],delta_pair*0.1,pairtable)
    

    if init_vel==True:
        vel_particles=create_vel_particles(particles,temp=temp)

    particle_images=create_image_particles(particles)

    TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)

    temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
    
    ### Initialize thermo data and forces
    thermostat_current=np.zeros(4)
    force_particles = np.zeros(num_particles, dtype=force_particle)

    ### Pair interactions ###
    excludelist=create_angle_excludelist(angles,particles)
    neighlist=np.ones([num_particles,neighborlist_length])*-1
    neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
    calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

    ### Bonded interactions ###
    calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
    calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

    for runii in tqdm(range(equilibration_steps)):
        ### First step of integration ###
        NVT_BerendsenDynamics_firststep(particle_images,box,particles,vel_particles,force_particles,dt=timestep,temp=temp)

        ### Initialize thermo data and forces
        thermostat_current=np.zeros(4)
        force_particles = np.zeros(num_particles, dtype=force_particle)

        ### Pair interactions ###
        if runii%neigh_update==0:
            neighlist=np.ones([num_particles,neighborlist_length])*-1
            neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
        calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

        L_max=3.0
        L_cut=1.5
        break_dist=0.4
        calc_Forces_table_virtual_meanfield(num_virtualsite,force_particles,box,particles,pairtable,delta_pair,force_softer,L_max,L_cut,excludelist,break_dist)

        #neighlist_virtual=np.ones([particles.shape[0],int(neighborlist_length*1.3)])*-1
        #neighbor_list_virtual(neighlist_virtual,particles,box,neighbor_distance,min_dist,max_dist)
        #print(neighlist_virtual)
        #calc_Forces_table_virtual(num_virtualsite,neighbor_distance,force_particles,box,neighlist_virtual,particles,pairtable,delta_pair,force_softer)

        ### Bonded interactions ###
        calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
        calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

        if runii%themo_freq==0:
            ### Output thermo data
            TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)
            temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
            output_thermo(runii,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)

        NVT_BerendsenDynamics_secondstep(runii,particles,vel_particles,force_particles,dt=timestep,tau=timestep*100,temp=temp)

    ### Output thermo data
    out_trjtimes=create_outtrjtimes(size_trjblock,num_trjblock)
    output_thermo(0,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)
    output_Trj(f_trj,particle_images,0,box,particles,type_="lammpscustom")

    for runii in tqdm(range(out_trjtimes[-1]+1)):
        ### First step of integration ###
        NVT_BerendsenDynamics_firststep(particle_images,box,particles,vel_particles,force_particles,dt=timestep,temp=temp)

        ### Initialize thermo data and forces
        thermostat_current=np.zeros(4)
        force_particles = np.zeros(num_particles, dtype=force_particle)

        ### Pair interactions ###
        if runii%neigh_update==0:
            neighlist=np.ones([num_particles,neighborlist_length])*-1
            neighbor_list_angle(neighlist,particles,box,neighbor_distance,excludelist)
        calc_pair_table(force_particles,box,neighlist,particles,pairtable,delta_pair,thermostat_current)

        ### Virtual sites ###

        #min_dist=neighbor_distance+0.5
        #max_dist=neighbor_distance*2

        #neighlist_virtual=np.ones([particles.shape[0],int(neighborlist_length*1.3)])*-1
        #neighbor_list_virtual(neighlist_virtual,particles,box,neighbor_distance,min_dist,max_dist)
        #print(neighlist_virtual)
        #calc_Forces_table_virtual(num_virtualsite,neighbor_distance,force_particles,box,neighlist_virtual,particles,pairtable,delta_pair,force_softer)
        L_max=3.0
        L_cut=1.5
        break_dist=0.4
        calc_Forces_table_virtual_meanfield(num_virtualsite,force_particles,box,particles,pairtable,delta_pair,force_softer,L_max,L_cut,excludelist,break_dist)
        ### Bonded interactions ###
        calc_bond_table(bondtable,force_particles,bonds,box,particles,delta_bond,thermostat_current)
        calc_angle_table(angletable,force_particles,angles,box,particles,delta_angle,thermostat_current)

        if runii%themo_freq==0:
            ### Output thermo data
            TotalKineticEnergy=calc_TotalKineticEnergy(particles,vel_particles)
            temp_curr=calc_InstaneousTmep(particles,TotalKineticEnergy)
            output_thermo(runii,f_thermo,thermostat_current,TotalKineticEnergy,temp_curr)

        if runii in out_trjtimes:
            ### Output thermo data
            output_Trj(f_trj,particle_images,runii,box,particles,type_="lammpscustom")

        NVT_BerendsenDynamics_secondstep(runii,particles,vel_particles,force_particles,dt=timestep,tau=timestep*100,temp=temp)

    f_trj.close()
    f_thermo.close()

main_equilibration_exponentialscheme(datafile='./test_4000/PS.data',equilibration_steps=10000,force_softer=1.0,num_virtualsite=24,runstep=100000,temp=373*kB,themo_freq=1000,num_trjblock=10,size_trjblock=58,trjfile='test.dump',thermofile='test.thermo',timestep=0.004,neigh_update=1,neighbor_distance=1.5+0.1,box=(9.14560,9.14560,9.14560),neighborlist_length=300,BondTableFile='bondAA.pot.tablenew',AngleTableFile='angleAAA.pot.tablenew',PairTableFile='nbAA_new.pot.table',init_vel=True)

#print(create_outtrjtimes(58,10))