# -*- coding: utf-8 -*-
"""
__author__ = "Alexandra Filiatraut, Mehdi Zanjani"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Mehdi Zanjani"
__email__ = "zanjanm.miamioh.edu"
__status__ = "Production"
"""
maxneighcount=6
normfglobal=1109.6

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import numpy as np
import math
import matplotlib.pyplot as plt

#normalizatiion factors
mass0=1.0 
d0=1.0 
eps=1.0
tau=math.sqrt(mass0*d0*d0/eps)
print("tau=", tau)
vel0=d0/tau
print("vel0=",vel0)
f0=eps/d0
print("f0=",f0)
# READ particle coordinates from LAMMPS output file
f = open("outBinary", "r")
print(f.readline()) #this prints the first line
f.readline()
f.readline()#read line 3
myline=f.readline()
Ntotal=int(myline)
print(Ntotal, "atoms")
f.readline()#read line 5
myline=f.readline()
xlo=float(myline.split()[0])/d0#read line 6
xhi=float(myline.split()[1])/d0#read line 6

myline=f.readline()
ylo=float(myline.split()[0])/d0#read line 7
yhi=float(myline.split()[1])/d0#read line 7

myline=f.readline()
zlo=float(myline.split()[0])/d0#read line 8
zhi=float(myline.split()[1])/d0#read line 8

print(zlo,"zhi=", zhi)

boxx=(xhi-xlo)
boxy=(yhi-ylo)
boxz=(zhi-zlo)

print("bxo x=", boxx)
print("bxo y=", boxy)
print("bxo z=", boxz)


f.readline() #read line 9 

xpos = [0.00] * (Ntotal)
ypos = [0.00] * (Ntotal)
zpos = [0.00] * (Ntotal)
typepos = [0] * (Ntotal)
velx = [0.00] * (Ntotal)
vely = [0.00] * (Ntotal)
velz = [0.00] * (Ntotal)
frx = [0.00] * (Ntotal)
fry = [0.00] * (Ntotal)
frz = [0.00] * (Ntotal)

#initialize graph
masstype1=1.0  
masstype2=1.0 
diam_1=1.0 
diam_2=1.0 

mynodefeatures = []
forceoutput=[]


for i in range(Ntotal):
    myline=f.readline()
    typepos[i]=int(myline.split()[1])
    xpos[i]=float(myline.split()[2])/d0
    if(xpos[i]<xlo):
        print("xpos less than xlo for id=",i,"pos=",xpos[i])
    ypos[i]=float(myline.split()[3])/d0
    if(ypos[i]<ylo):
        print("ypos less than xlo for id=",i,"pos=",ypos[i])
    zpos[i]=float(myline.split()[4])/d0
    if(zpos[i]<zlo):
        print("zpos less than xlo for id=",i,"pos=",zpos[i])
    velx[i]=float(myline.split()[5])/vel0
    vely[i]=float(myline.split()[6])/vel0
    velz[i]=float(myline.split()[7])/vel0
    frx[i]=float(myline.split()[8])/f0
    fry[i]=float(myline.split()[9])/f0
    frz[i]=float(myline.split()[10])/f0

line000=f.readline()
timestepcount=1
timesteps=[]

print("boxx=", boxx)
print("boxy=", boxy)
print("boxz=", boxz)


rcut = 1.30
rcutsq = rcut * rcut

#define cells 
def icellnum(a,b,c,mm):
    cellind= (a%mm) + (b%mm) * mm + (c%mm) * mm * mm
    return cellind
#Lcell = 4.0 * rcut
#cellID = [0] * (Ntotal)
M=10 #math.floor(boxx/Lcell)
Lcell=boxx/M
ncell = M * M * M
print("ncell =", ncell, "Lcell=",Lcell)
# find cell neighbors first
Cellneigh=[0] * (13*ncell)
for iz in range(M):
    for iy in range(M):
        for ix in range(M):
            imap=icellnum(ix,iy,iz,M)*13
            Cellneigh[imap]=icellnum(ix+1,iy,iz,M)
            Cellneigh[imap+1]=icellnum(ix+1,iy+1,iz,M)
            Cellneigh[imap+2]=icellnum(ix,iy+1,iz,M)
            Cellneigh[imap+3]=icellnum(ix-1,iy+1,iz,M)
            Cellneigh[imap+4]=icellnum(ix+1,iy,iz-1,M)
            Cellneigh[imap+5]=icellnum(ix+1,iy+1,iz-1,M)
            Cellneigh[imap+6]=icellnum(ix,iy+1,iz-1,M)
            Cellneigh[imap+7]=icellnum(ix-1,iy+1,iz-1,M)
            Cellneigh[imap+8]=icellnum(ix+1,iy,iz+1,M)
            Cellneigh[imap+9]=icellnum(ix+1,iy+1,iz+1,M)
            Cellneigh[imap+10]=icellnum(ix,iy+1,iz+1,M)
            Cellneigh[imap+11]=icellnum(ix-1,iy+1,iz+1,M)
            Cellneigh[imap+12]=icellnum(ix,iy,iz+1,M)


head=[-1]*ncell
list_cells=[-1] * Ntotal

for ii in range(Ntotal):
    aofcell=int((xpos[ii]-xlo)/boxx*float(M))
    bofcell=int((ypos[ii]-ylo)/boxy*float(M))
    cofcell=int((zpos[ii]-zlo)/boxz*float(M))
    if aofcell > (M-1) or aofcell < 0:
        print("wrong cell x index =", aofcell)
    if bofcell > (M-1) or bofcell < 0:
        print("wrong cell y index =", bofcell)
    if cofcell > (M-1) or cofcell < 0:
        print("wrong cell z index =", cofcell)
    cellind=icellnum(aofcell,bofcell,cofcell,M)
    list_cells[ii]=head[cellind]
    head[cellind]= ii


num_edge = 0
edge_1=[]
edge_2=[]
edge_RL=[]
numneigh_of_i= [0] * Ntotal
index_neighofi=[-1] * Ntotal
Myadj=np.zeros((Ntotal,Ntotal),dtype=int)

neighbor_list = []
for i in range(Ntotal):
    neighbor_list.append([])
    
for ICELL in range(ncell):
    i= head[ICELL]
    while i > -1:
        j=list_cells[i]
        while j > -1:
            RXIJ = xpos[i]-xpos[j] 
            RYIJ = ypos[i]-ypos[j]
            RZIJ = zpos[i]-zpos[j]
            RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
            RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
            RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
            dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
            if dist <= rcutsq:
                indx1=i
                indx2=j
                num_edge += 1
                edge_1.append(i)
                edge_2.append(j)
                edge_1.append(j)
                edge_2.append(i)
                Myadj[i][j]=1
                Myadj[j][i]=1
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)
                numneigh_of_i[i] += 1
                numneigh_of_i[j] += 1
                if typepos[i]==1 and typepos[j]==1:
                    edgtp=0
                elif typepos[i]==2 and typepos[j]==2:
                    edgtp=2
                else:
                    edgtp=1
                edge_RL.append(edgtp)
                edge_RL.append(edgtp)
            j=list_cells[j]
        jcell0= 13*ICELL
        for nobar in range(13):
            jcell= Cellneigh[jcell0+nobar]
            j=head[jcell]
            while j > -1:
                RXIJ = xpos[i]-xpos[j] 
                RYIJ = ypos[i]-ypos[j]
                RZIJ = zpos[i]-zpos[j]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
                if dist <= rcutsq:
                    indx1=i
                    indx2=j
                    num_edge += 1
                    edge_1.append(i)
                    edge_2.append(j)
                    edge_1.append(j)
                    edge_2.append(i)
                    Myadj[i][j]=1
                    Myadj[j][i]=1
                    numneigh_of_i[i] += 1
                    numneigh_of_i[j] += 1
                    neighbor_list[i].append(j)
                    neighbor_list[j].append(i)
                    if typepos[i]==1 and typepos[j]==1:
                        edgtp=0
                    elif typepos[i]==2 and typepos[j]==2:
                        edgtp=2
                    else:
                        edgtp=1
                    edge_RL.append(edgtp)
                    edge_RL.append(edgtp)
                j=list_cells[j]
        i=list_cells[i]

print("num of edges in timestep ", timestepcount, " = ", num_edge)
print("length of neighborlist=", len(neighbor_list))
#print("length of neighborlistatomind=", len(neighlistatomindex))
#for i0 in range(Ntotal):
    #print("neighlist of atom ",i0," =  ", neighbor_list[i0])
    #print("numneigh of this atom =", numneigh_of_i[i0])
    
print("**************************************")
print("**************************************")
print("**************************************")
print("**************************************")

mydata0=[]
mydataout0=[]
datacount=0
totalnodefeatures=5 # one value for type of interaction, one value for average diameter, x-x0,y-y0,z-z0


print("*************************")
print("******Done with initial time step************")



totaltimesteps = 1000
stfreq=2
#count different catergory data points, i.e. the number of data points for 1 neighbors, 2 neighbors, 3...
ndatacount=[0]*(maxneighcount+1)
icount=[0]*(maxneighcount+1)

#normalization factor for force
normF=normfglobal
normX=rcut
while (timestepcount < totaltimesteps) and (datacount <= 10000):
    for i01 in range(4):
        line000=f.readline()
    myline=f.readline()
    xlo=float(myline.split()[0])/d0#read line 6
    xhi=float(myline.split()[1])/d0#read line 6
    myline=f.readline()
    ylo=float(myline.split()[0])/d0#read line 7
    yhi=float(myline.split()[1])/d0#read line 7
    myline=f.readline()
    zlo=float(myline.split()[0])/d0#read line 8
    zhi=float(myline.split()[1])/d0#read line 8
    boxx=(xhi-xlo)
    boxy=(yhi-ylo)
    boxz=(zhi-zlo)
    f.readline()
    timestepcount += 1
    #print("step==",timestepcount)
    edge_feat=[]
    forceoutput=[]
    mynodefeatures=[]
    for i in range(Ntotal):
        myline=f.readline()
        typepos[i]=int(myline.split()[1])
        xpos[i]=float(myline.split()[2])/d0
        #if(xpos[i]<xlo):
            #print("xpos less than xlo for id=",i,"pos=",xpos[i], "timstep===", timestepcount)
        ypos[i]=float(myline.split()[3])/d0
        #if(ypos[i]<ylo):
            #print("ypos less than xlo for id=",i,"pos=",ypos[i], "timstep===", timestepcount)
        zpos[i]=float(myline.split()[4])/d0
        #if(zpos[i]<zlo):
            #print("zpos less than xlo for id=",i,"pos=",zpos[i], "timstep===", timestepcount)
        velx[i]=float(myline.split()[5])/vel0
        vely[i]=float(myline.split()[6])/vel0
        velz[i]=float(myline.split()[7])/vel0
        frx[i]=float(myline.split()[8])/f0
        fry[i]=float(myline.split()[9])/f0
        frz[i]=float(myline.split()[10])/f0
    Lcell=boxx/M
    #print("ncell =", ncell, "Lcell=",Lcell)
    head=[-1] * ncell
    list_cells=[-1] * Ntotal
    for ii in range(Ntotal):
        aofcell=int((xpos[ii]-xlo)/boxx*float(M))
        bofcell=int((ypos[ii]-ylo)/boxy*float(M))
        cofcell=int((zpos[ii]-zlo)/boxz*float(M))
        if aofcell == M:
            aofcell=0
        if aofcell == -1:
            aofcell=M-1
        if bofcell == M:
            bofcell=0
        if bofcell == -1:
            bofcell=M-1
        if cofcell == M:
            cofcell=0
        if cofcell == -1:
            cofcell=M-1
        if aofcell > (M-1) or aofcell < 0:
            print("wrong cell x index =", aofcell)
        if bofcell > (M-1) or bofcell < 0:
            print("wrong cell y index =", bofcell)
        if cofcell > (M-1) or cofcell < 0:
            print("wrong cell z index =", cofcell)
        cellind=icellnum(aofcell,bofcell,cofcell,M)
        list_cells[ii]=head[cellind]
        head[cellind]= ii
    
    num_edge = 0
    edge_1=[]
    edge_2=[]
    edge_RL=[]
    numneigh_of_i= [0] * Ntotal
    index_neighofi=[-1] * Ntotal
    Myadj=np.zeros((Ntotal,Ntotal),dtype=int)
    
    neighbor_list = []
    for i in range(Ntotal):
        neighbor_list.append([])
    for ICELL in range(ncell):
        i= head[ICELL]
        while i > -1:
            j=list_cells[i]
            while j > -1:
                RXIJ = xpos[i]-xpos[j] 
                RYIJ = ypos[i]-ypos[j]
                RZIJ = zpos[i]-zpos[j]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
                if dist <= rcutsq:
                    indx1=i
                    indx2=j
                    num_edge += 1
                    edge_1.append(i)
                    edge_2.append(j)
                    edge_1.append(j)
                    edge_2.append(i)
                    Myadj[i][j]=1
                    Myadj[j][i]=1
                    neighbor_list[i].append(j)
                    neighbor_list[j].append(i)
                    numneigh_of_i[i] += 1
                    numneigh_of_i[j] += 1
                    if typepos[i]==1 and typepos[j]==1:
                        edgtp=0
                    elif typepos[i]==2 and typepos[j]==2:
                        edgtp=2
                    else:
                        edgtp=1
                    edge_RL.append(edgtp)
                    edge_RL.append(edgtp)
                j=list_cells[j]
            jcell0= 13*ICELL
            for nobar in range(13):
                jcell= Cellneigh[jcell0+nobar]
                j=head[jcell]
                while j > -1:
                    RXIJ = xpos[i]-xpos[j] 
                    RYIJ = ypos[i]-ypos[j]
                    RZIJ = zpos[i]-zpos[j]
                    RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                    RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                    RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                    dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
                    if dist <= rcutsq:
                        indx1=i
                        indx2=j
                        num_edge += 1
                        edge_1.append(i)
                        edge_2.append(j)
                        edge_1.append(j)
                        edge_2.append(i)
                        Myadj[i][j]=1
                        Myadj[j][i]=1
                        numneigh_of_i[i] += 1
                        numneigh_of_i[j] += 1
                        neighbor_list[i].append(j)
                        neighbor_list[j].append(i)
                        if typepos[i]==1 and typepos[j]==1:
                            edgtp=0
                        elif typepos[i]==2 and typepos[j]==2:
                            edgtp=2
                        else:
                            edgtp=1
                        edge_RL.append(edgtp)
                        edge_RL.append(edgtp)
                    j=list_cells[j]
            i=list_cells[i]
    
    if ((timestepcount%stfreq)>-1):
        print("num of edges in timestep ", timestepcount, " = ", num_edge)
    for i in range(Ntotal):
        if (numneigh_of_i[i] == maxneighcount) and ((timestepcount%stfreq)>-1):
            sub_nodefeatures=np.zeros((maxneighcount,totalnodefeatures),dtype=float)
            if typepos[i]==1:
                massi = masstype1/mass0
                diam_i=diam_1
            if typepos[i]==2:
                massi = masstype2/mass0
                diam_i=diam_2
            for indx in range(numneigh_of_i[i]):
                j=neighbor_list[i][indx]
                if typepos[i]==1 and typepos[j]==1:
                    sub_nodefeatures[indx][0]=0.0 
                if typepos[i]==2 and typepos[j]==2:
                    sub_nodefeatures[indx][0]=1.0
                if (typepos[i]==1 and typepos[j]==2) or (typepos[i]==2 and typepos[j]==1):
                    sub_nodefeatures[indx][0]=0.5
                if typepos[j]==1:
                    massj = masstype1/mass0
                    diam_j=diam_1
                if typepos[j]==2:
                    massj = masstype2/mass0
                    diam_j=diam_2
                sub_nodefeatures[indx][1]=0.5*(diam_i+diam_j)/(2.00*normX)
                RXIJ = xpos[j]-xpos[i] 
                RYIJ = ypos[j]-ypos[i]
                RZIJ = zpos[j]-zpos[i]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                sub_nodefeatures[indx][2]=(RXIJ+normX)/(2.00*normX)
                sub_nodefeatures[indx][3]=(RYIJ+normX)/(2.00*normX)
                sub_nodefeatures[indx][4]=(RZIJ+normX)/(2.00*normX)
            myx = np.array(sub_nodefeatures) #torch.tensor(np.array(sub_nodefeatures),dtype=torch.float)
            myy = np.array([frx[i], fry[i], frz[i]], dtype=float) #torch.tensor(np.array([frx[i], fry[i], frz[i]]),dtype=torch.float)
            mydata0.append(myx)
            mydataout0.append(myy)
            datacount += 1
            ndatacount[numneigh_of_i[i]] +=1
    line000=f.readline()
    if not line000:
        break
f.close()

for i in range(maxneighcount+1):
    icount[i]=i
plt.plot(icount,ndatacount)
plt.savefig ("neighborcountofdata_neigh3.png", dpi=300)

mydata=mydata0
mydata_Out=mydataout0
print("lenght of total dataset=", len(mydata0))
print("shape of total dataset=", np.array(mydata0).shape)
print("lenght of total output dataset=", len(mydataout0))
print("shape of total output dataset=", np.array(mydataout0).shape)
print("lenght of total dataset=", datacount)

        
print("length of train data=**********************", len(mydata))


print(" max Force= ", normF)
print("max X =", normX)

#normalize force data
for i0 in range(len(mydata)):
    mydata_Out[i0][0] = (mydata_Out[i0][0]+normF)/(2.00*normF)
    mydata_Out[i0][1] = (mydata_Out[i0][1]+normF)/(2.00*normF)
    mydata_Out[i0][2] = (mydata_Out[i0][2]+normF)/(2.00*normF)


mydata=np.array(mydata)
mydata_Out=np.array(mydata_Out)
mydata=np.reshape(np.array(mydata),(len(mydata),maxneighcount,totalnodefeatures,1))
print("mydata neew shape==*********", mydata.shape)

print("out train shape=", mydata_Out.shape)

print("np.shape(mydata)[1:]= ", np.shape(mydata)[1:])


print("max value of inputs", np.max([abs(np.max(mydata)),abs(np.min(mydata))]))
print("max value of force outputs", np.max([abs(np.max(mydata_Out)),abs(np.min(mydata_Out))]))

#np.save('NormOF_X.npy',normX)
print('NormOF_X ==',normX)

#np.save('NormOF_F.npy',normF)
print('NormOF_F ==',normF)

print("mydata new shape==*********", mydata.shape)
print("out train shape=", mydata_Out.shape)

#ML Model : CNN + dense layers
myfeaturesize0=totalnodefeatures

myoutsize=3
print("node featuresize=",myfeaturesize0)
print("outputsize=",myoutsize)
nconvfilters=10

model = models.Sequential()
model.add(layers.Conv2D(nconvfilters, (1,myfeaturesize0), activation='sigmoid', input_shape=np.shape(mydata)[1:]))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dense(maxneighcount*2, activation='sigmoid'))
model.add(layers.Dense(maxneighcount, activation='sigmoid'))
model.add(layers.Dense(myoutsize, activation='sigmoid'))

learning_rate=0.0002
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#model.compile(optimizer=optimizer, loss=['mean_squared_error'], metrics=['mean_absolute_percentage_error'])
#model.compile(optimizer=optimizer, loss=['mean_absolute_error'], metrics=['mean_absolute_percentage_error'])
model.compile(optimizer=optimizer, loss=['mean_absolute_error'], metrics=['mean_absolute_error'])
#model.compile(optimizer='sgd', loss="mean_absolute_error", metrics=["mean_absolute_error"])
model.summary()
history = model.fit(mydata, mydata_Out, batch_size=64, epochs=100, validation_split=0.10)

#MYmodelname='Model_'+str(maxneighcount)
#model.save(MYmodelname)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lossfigure_cnn_neigh_NEW'+str(maxneighcount)+'.png', dpi=300)

## test for prediction with totally new particle trajectories
#normalization factor from sweeping all the timesteps


f = open("outBinary_prediction", "r")
#f=open("outIPNstr_NEWsmallnew_10p0_nstep_10", "r")
print(f.readline()) #this prints the first line
f.readline()
f.readline()#read line 3
myline=f.readline()
Ntotal=int(myline)
print(Ntotal, "atoms")
f.readline()#read line 5
myline=f.readline()
xlo=float(myline.split()[0])/d0#read line 6
xhi=float(myline.split()[1])/d0#read line 6

myline=f.readline()
ylo=float(myline.split()[0])/d0#read line 7
yhi=float(myline.split()[1])/d0#read line 7

myline=f.readline()
zlo=float(myline.split()[0])/d0#read line 8
zhi=float(myline.split()[1])/d0#read line 8

print(zlo,"zhi=", zhi)

boxx=(xhi-xlo)/d0
boxy=(yhi-ylo)/d0
boxz=(zhi-zlo)/d0

print("bxo x=", boxx)
print("bxo y=", boxy)
print("bxo z=", boxz)


f.readline() #read line 9 

xpos = [0.00] * (Ntotal)
ypos = [0.00] * (Ntotal)
zpos = [0.00] * (Ntotal)
typepos = [0] * (Ntotal)
velx = [0.00] * (Ntotal)
vely = [0.00] * (Ntotal)
velz = [0.00] * (Ntotal)
frx = [0.00] * (Ntotal)
fry = [0.00] * (Ntotal)
frz = [0.00] * (Ntotal)


mynodefeatures = []
forceoutput=[]

for i in range(Ntotal):
    myline=f.readline()
    typepos[i]=int(myline.split()[1])
    xpos[i]=float(myline.split()[2])/d0
    if(xpos[i]<xlo):
        print("xpos less than xlo for id=",i,"pos=",xpos[i])
    ypos[i]=float(myline.split()[3])/d0
    if(ypos[i]<ylo):
        print("ypos less than xlo for id=",i,"pos=",ypos[i])
    zpos[i]=float(myline.split()[4])/d0
    if(zpos[i]<zlo):
        print("zpos less than xlo for id=",i,"pos=",zpos[i])
    velx[i]=float(myline.split()[5])/vel0
    vely[i]=float(myline.split()[6])/vel0
    velz[i]=float(myline.split()[7])/vel0
    frx[i]=float(myline.split()[8])/f0
    fry[i]=float(myline.split()[9])/f0
    frz[i]=float(myline.split()[10])/f0

line000=f.readline()
timestepcount=1
timesteps=[]

print("boxx=", boxx)
print("boxy=", boxy)
print("boxz=", boxz)

#define cells 
def icellnum(a,b,c,mm):
    cellind= (a%mm) + (b%mm) * mm + (c%mm) * mm * mm
    return cellind
#Lcell = 4.0 * rcut
#cellID = [0] * (Ntotal)
M=10 #math.floor(boxx/Lcell)
Lcell=boxx/M
ncell = M * M * M
print("ncell =", ncell, "Lcell=",Lcell)
# find cell neighbors first
Cellneigh=[0] * (13*ncell)
for iz in range(M):
    for iy in range(M):
        for ix in range(M):
            imap=icellnum(ix,iy,iz,M)*13
            Cellneigh[imap]=icellnum(ix+1,iy,iz,M)
            Cellneigh[imap+1]=icellnum(ix+1,iy+1,iz,M)
            Cellneigh[imap+2]=icellnum(ix,iy+1,iz,M)
            Cellneigh[imap+3]=icellnum(ix-1,iy+1,iz,M)
            Cellneigh[imap+4]=icellnum(ix+1,iy,iz-1,M)
            Cellneigh[imap+5]=icellnum(ix+1,iy+1,iz-1,M)
            Cellneigh[imap+6]=icellnum(ix,iy+1,iz-1,M)
            Cellneigh[imap+7]=icellnum(ix-1,iy+1,iz-1,M)
            Cellneigh[imap+8]=icellnum(ix+1,iy,iz+1,M)
            Cellneigh[imap+9]=icellnum(ix+1,iy+1,iz+1,M)
            Cellneigh[imap+10]=icellnum(ix,iy+1,iz+1,M)
            Cellneigh[imap+11]=icellnum(ix-1,iy+1,iz+1,M)
            Cellneigh[imap+12]=icellnum(ix,iy,iz+1,M)


head=[-1]*ncell
list_cells=[-1] * Ntotal

for ii in range(Ntotal):
    aofcell=int((xpos[ii]-xlo)/boxx*float(M))
    bofcell=int((ypos[ii]-ylo)/boxy*float(M))
    cofcell=int((zpos[ii]-zlo)/boxz*float(M))
    if aofcell > (M-1) or aofcell < 0:
        print("wrong cell x index =", aofcell)
    if bofcell > (M-1) or bofcell < 0:
        print("wrong cell y index =", bofcell)
    if cofcell > (M-1) or cofcell < 0:
        print("wrong cell z index =", cofcell)
    cellind=icellnum(aofcell,bofcell,cofcell,M)
    list_cells[ii]=head[cellind]
    head[cellind]= ii


num_edge = 0
edge_1=[]
edge_2=[]
edge_RL=[]
numneigh_of_i= [0] * Ntotal
index_neighofi=[-1] * Ntotal
Myadj=np.zeros((Ntotal,Ntotal),dtype=int)

neighbor_list = []
for i in range(Ntotal):
    neighbor_list.append([])
    
for ICELL in range(ncell):
    i= head[ICELL]
    while i > -1:
        j=list_cells[i]
        while j > -1:
            RXIJ = xpos[i]-xpos[j] 
            RYIJ = ypos[i]-ypos[j]
            RZIJ = zpos[i]-zpos[j]
            RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
            RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
            RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
            dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
            if dist <= rcutsq:
                indx1=i
                indx2=j
                num_edge += 1
                edge_1.append(i)
                edge_2.append(j)
                edge_1.append(j)
                edge_2.append(i)
                Myadj[i][j]=1
                Myadj[j][i]=1
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)
                numneigh_of_i[i] += 1
                numneigh_of_i[j] += 1
                if typepos[i]==1 and typepos[j]==1:
                    edgtp=0
                elif typepos[i]==2 and typepos[j]==2:
                    edgtp=2
                else:
                    edgtp=1
                edge_RL.append(edgtp)
                edge_RL.append(edgtp)
            j=list_cells[j]
        jcell0= 13*ICELL
        for nobar in range(13):
            jcell= Cellneigh[jcell0+nobar]
            j=head[jcell]
            while j > -1:
                RXIJ = xpos[i]-xpos[j] 
                RYIJ = ypos[i]-ypos[j]
                RZIJ = zpos[i]-zpos[j]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
                if dist <= rcutsq:
                    indx1=i
                    indx2=j
                    num_edge += 1
                    edge_1.append(i)
                    edge_2.append(j)
                    edge_1.append(j)
                    edge_2.append(i)
                    Myadj[i][j]=1
                    Myadj[j][i]=1
                    numneigh_of_i[i] += 1
                    numneigh_of_i[j] += 1
                    neighbor_list[i].append(j)
                    neighbor_list[j].append(i)
                    if typepos[i]==1 and typepos[j]==1:
                        edgtp=0
                    elif typepos[i]==2 and typepos[j]==2:
                        edgtp=2
                    else:
                        edgtp=1
                    edge_RL.append(edgtp)
                    edge_RL.append(edgtp)
                j=list_cells[j]
        i=list_cells[i]

print("num of edges in timestep ", timestepcount, " = ", num_edge)
print("length of neighborlist=", len(neighbor_list))
    
print("**************************************")
print("**************************************")
print("**************************************")
print("**************************************")

n_prediction=0
fxtrue=[]
fytrue=[]
fztrue=[]
fx_predicted=[]
fy_predicted=[]
fz_predicted=[]


f2x = open("Nscatterplot_xbinary.txt", "w")
f2y = open("Nscatterplot_ybinary.txt", "w")
f2z = open("Nscatterplot_zbinary.txt", "w")

normF=normfglobal
npredictioncount=500
mae_fx=0.0
mae_fy=0.0
mae_fz=0.0
for i in range(Ntotal):
    if numneigh_of_i[i] == maxneighcount:
        myx=[]
        sub_nodefeatures=np.zeros((maxneighcount,totalnodefeatures),dtype=float)
        if typepos[i]==1:
            massi = masstype1/mass0
            diam_i=diam_1
        if typepos[i]==2:
            massi = masstype2/mass0
            diam_i=diam_2
        for indx in range(numneigh_of_i[i]):
            j=neighbor_list[i][indx]
            if typepos[i]==1 and typepos[j]==1:
                sub_nodefeatures[indx][0]=0.0 
            if typepos[i]==2 and typepos[j]==2:
                sub_nodefeatures[indx][0]=1.0
            if (typepos[i]==1 and typepos[j]==2) or (typepos[i]==2 and typepos[j]==1):
                sub_nodefeatures[indx][0]=0.5
            if typepos[j]==1:
                massj = masstype1/mass0
                diam_j=diam_1
            if typepos[j]==2:
                massj = masstype2/mass0
                diam_j=diam_2
            sub_nodefeatures[indx][1]=0.5*(diam_i+diam_j)/(2.00*normX)
            RXIJ = xpos[j]-xpos[i] 
            RYIJ = ypos[j]-ypos[i]
            RZIJ = zpos[j]-zpos[i]
            RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
            RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
            RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
            sub_nodefeatures[indx][2]=(RXIJ+normX)/(2.00*normX)
            sub_nodefeatures[indx][3]=(RYIJ+normX)/(2.00*normX)
            sub_nodefeatures[indx][4]=(RZIJ+normX)/(2.00*normX)
        myx = np.array(sub_nodefeatures)
        fxtrue.append((frx[i]+normF)/(2.00*normF))
        fytrue.append((fry[i]+normF)/(2.00*normF))
        fztrue.append((frz[i]+normF)/(2.00*normF))
        myinput=np.reshape(myx,(1,maxneighcount,totalnodefeatures,1))
        myout_i=np.array(model.predict(myinput))
        fx_predicted.append(myout_i[0][0])
        fy_predicted.append(myout_i[0][1])
        fz_predicted.append(myout_i[0][2])
        #find 1000 new datapoints to evaluate average accuracy of new predictions
        if (n_prediction<npredictioncount): 
            n_prediction += 1
            mae_fx += abs((frx[i]+normF)/(2.00*normF)-myout_i[0][0])
            mae_fy += abs((fry[i]+normF)/(2.00*normF)-myout_i[0][1])
            mae_fz += abs((frz[i]+normF)/(2.00*normF)-myout_i[0][2])
            f2x.write("%f %f \r\n" % ((frx[i]+normF)/(2.00*normF), myout_i[0][0]))
            f2y.write("%f %f \r\n" % ((fry[i]+normF)/(2.00*normF), myout_i[0][1]))
            f2z.write("%f %f \r\n" % ((frz[i]+normF)/(2.00*normF), myout_i[0][2]))


while n_prediction<npredictioncount:
    for i01 in range(4):
        line000=f.readline()
    myline=f.readline()
    xlo=float(myline.split()[0])/d0#read line 6
    xhi=float(myline.split()[1])/d0#read line 6
    myline=f.readline()
    ylo=float(myline.split()[0])/d0#read line 7
    yhi=float(myline.split()[1])/d0#read line 7
    myline=f.readline()
    zlo=float(myline.split()[0])/d0#read line 8
    zhi=float(myline.split()[1])/d0#read line 8
    boxx=(xhi-xlo)
    boxy=(yhi-ylo)
    boxz=(zhi-zlo)
    f.readline()
    timestepcount += 1
    #print("step==",timestepcount)
    edge_feat=[]
    forceoutput=[]
    mynodefeatures=[]
    for i in range(Ntotal):
        myline=f.readline()
        typepos[i]=int(myline.split()[1])
        xpos[i]=float(myline.split()[2])/d0
        #if(xpos[i]<xlo):
            #print("xpos less than xlo for id=",i,"pos=",xpos[i], "timstep===", timestepcount)
        ypos[i]=float(myline.split()[3])/d0
        #if(ypos[i]<ylo):
            #print("ypos less than xlo for id=",i,"pos=",ypos[i], "timstep===", timestepcount)
        zpos[i]=float(myline.split()[4])/d0
        #if(zpos[i]<zlo):
            #print("zpos less than xlo for id=",i,"pos=",zpos[i], "timstep===", timestepcount)
        velx[i]=float(myline.split()[5])/vel0
        vely[i]=float(myline.split()[6])/vel0
        velz[i]=float(myline.split()[7])/vel0
        frx[i]=float(myline.split()[8])/f0
        fry[i]=float(myline.split()[9])/f0
        frz[i]=float(myline.split()[10])/f0
    Lcell=boxx/M
    #print("ncell =", ncell, "Lcell=",Lcell)
    head=[-1] * ncell
    list_cells=[-1] * Ntotal
    for ii in range(Ntotal):
        aofcell=int((xpos[ii]-xlo)/boxx*float(M))
        bofcell=int((ypos[ii]-ylo)/boxy*float(M))
        cofcell=int((zpos[ii]-zlo)/boxz*float(M))
        if aofcell == M:
            aofcell=0
        if aofcell == -1:
            aofcell=M-1
        if bofcell == M:
            bofcell=0
        if bofcell == -1:
            bofcell=M-1
        if cofcell == M:
            cofcell=0
        if cofcell == -1:
            cofcell=M-1
        if aofcell > (M-1) or aofcell < 0:
            print("wrong cell x index =", aofcell)
        if bofcell > (M-1) or bofcell < 0:
            print("wrong cell y index =", bofcell)
        if cofcell > (M-1) or cofcell < 0:
            print("wrong cell z index =", cofcell)
        cellind=icellnum(aofcell,bofcell,cofcell,M)
        list_cells[ii]=head[cellind]
        head[cellind]= ii
    
    num_edge = 0
    edge_1=[]
    edge_2=[]
    edge_RL=[]
    numneigh_of_i= [0] * Ntotal
    index_neighofi=[-1] * Ntotal
    Myadj=np.zeros((Ntotal,Ntotal),dtype=int)
    
    neighbor_list = []
    for i in range(Ntotal):
        neighbor_list.append([])
    for ICELL in range(ncell):
        i= head[ICELL]
        while i > -1:
            j=list_cells[i]
            while j > -1:
                RXIJ = xpos[i]-xpos[j] 
                RYIJ = ypos[i]-ypos[j]
                RZIJ = zpos[i]-zpos[j]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
                if dist <= rcutsq:
                    indx1=i
                    indx2=j
                    num_edge += 1
                    edge_1.append(i)
                    edge_2.append(j)
                    edge_1.append(j)
                    edge_2.append(i)
                    Myadj[i][j]=1
                    Myadj[j][i]=1
                    neighbor_list[i].append(j)
                    neighbor_list[j].append(i)
                    numneigh_of_i[i] += 1
                    numneigh_of_i[j] += 1
                    if typepos[i]==1 and typepos[j]==1:
                        edgtp=0
                    elif typepos[i]==2 and typepos[j]==2:
                        edgtp=2
                    else:
                        edgtp=1
                    edge_RL.append(edgtp)
                    edge_RL.append(edgtp)
                j=list_cells[j]
            jcell0= 13*ICELL
            for nobar in range(13):
                jcell= Cellneigh[jcell0+nobar]
                j=head[jcell]
                while j > -1:
                    RXIJ = xpos[i]-xpos[j] 
                    RYIJ = ypos[i]-ypos[j]
                    RZIJ = zpos[i]-zpos[j]
                    RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                    RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                    RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                    dist = RXIJ * RXIJ + RYIJ * RYIJ + RZIJ * RZIJ
                    if dist <= rcutsq:
                        indx1=i
                        indx2=j
                        num_edge += 1
                        edge_1.append(i)
                        edge_2.append(j)
                        edge_1.append(j)
                        edge_2.append(i)
                        Myadj[i][j]=1
                        Myadj[j][i]=1
                        numneigh_of_i[i] += 1
                        numneigh_of_i[j] += 1
                        neighbor_list[i].append(j)
                        neighbor_list[j].append(i)
                        if typepos[i]==1 and typepos[j]==1:
                            edgtp=0
                        elif typepos[i]==2 and typepos[j]==2:
                            edgtp=2
                        else:
                            edgtp=1
                        edge_RL.append(edgtp)
                        edge_RL.append(edgtp)
                    j=list_cells[j]
            i=list_cells[i]
    print("number of prediction points @ timestep ", timestepcount, " = ", n_prediction)
    for i in range(Ntotal):
        if (numneigh_of_i[i] == maxneighcount) and ((timestepcount%stfreq)>-1):
            sub_nodefeatures=np.zeros((maxneighcount,totalnodefeatures),dtype=float)
            if typepos[i]==1:
                massi = masstype1/mass0
                diam_i=diam_1
            if typepos[i]==2:
                massi = masstype2/mass0
                diam_i=diam_2
            for indx in range(numneigh_of_i[i]):
                j=neighbor_list[i][indx]
                if typepos[i]==1 and typepos[j]==1:
                    sub_nodefeatures[indx][0]=0.0 
                if typepos[i]==2 and typepos[j]==2:
                    sub_nodefeatures[indx][0]=1.0
                if (typepos[i]==1 and typepos[j]==2) or (typepos[i]==2 and typepos[j]==1):
                    sub_nodefeatures[indx][0]=0.5
                if typepos[j]==1:
                    massj = masstype1/mass0
                    diam_j=diam_1
                if typepos[j]==2:
                    massj = masstype2/mass0
                    diam_j=diam_2
                sub_nodefeatures[indx][1]=0.5*(diam_i+diam_j)/(2.00*normX)
                RXIJ = xpos[j]-xpos[i] 
                RYIJ = ypos[j]-ypos[i]
                RZIJ = zpos[j]-zpos[i]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                sub_nodefeatures[indx][2]=(RXIJ+normX)/(2.00*normX)
                sub_nodefeatures[indx][3]=(RYIJ+normX)/(2.00*normX)
                sub_nodefeatures[indx][4]=(RZIJ+normX)/(2.00*normX)
            myx = np.array(sub_nodefeatures)
            fxtrue.append(frx[i]/normF)
            fytrue.append(fry[i]/normF)
            fztrue.append(frz[i]/normF)
            myinput=np.reshape(myx,(1,maxneighcount,totalnodefeatures,1))
            myout_i=np.array(model.predict(myinput))
            fxtrue.append((frx[i]+normF)/(2.00*normF))
            fytrue.append((fry[i]+normF)/(2.00*normF))
            fztrue.append((frz[i]+normF)/(2.00*normF))
            #find 1000 new datapoints to evaluate average accuracy of new predictions
            if (n_prediction<npredictioncount): 
                n_prediction += 1
                mae_fx += abs((frx[i]+normF)/(2.00*normF)-myout_i[0][0])
                mae_fy += abs((fry[i]+normF)/(2.00*normF)-myout_i[0][1])
                mae_fz += abs((frz[i]+normF)/(2.00*normF)-myout_i[0][2])
                f2x.write("%f %f \r\n" % ((frx[i]+normF)/(2.00*normF), myout_i[0][0]))
                f2y.write("%f %f \r\n" % ((fry[i]+normF)/(2.00*normF), myout_i[0][1]))
                f2z.write("%f %f \r\n" % ((frz[i]+normF)/(2.00*normF), myout_i[0][2]))
    line000=f.readline()
    if not line000:
        break
f.close()       
        
print("number of prediction points= ",n_prediction)  
print("average MAE=", (mae_fx+mae_fy+mae_fz)/(3*n_prediction) )
