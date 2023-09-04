import os
import random
import time
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

################################################################################
############# DATA COLLECTION ################

object_dim = 5 # features: ID, avg diameter, x, y, z

# n_relations : TBD in each timestep--- number of edges in fully connected graph
# relation_dim = 1 #type of the relation, i.e. type of particles i & j

effect_dim = 100 #effect's vector size

#read in the lammps output file
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
masstype1=1.0 #6.872234e-14
masstype2=1.0 #6.872234e-14
masstype3=1.0
masstype4=1.0
diam_1=1.0 #500.0e-7
diam_2=1.0 #500.0e-7
diam_3=1.0
diam_4=1.0
mynodefeatures = []
forceoutput=[]

nwall=0

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

print("new NTOTAL=", Ntotal)
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
                if typepos[i]==1 and typepos[j]==1: #one type system
                    edgtp=0
                elif typepos[i]==2 and typepos[j]==2:
                    edgtp=1
                else:
                    edgtp=0.5
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

mydata_obj=[]
mydataout=[]

mydata_obj_val=[]
mydataout_val=[]

mydata_obj_predict=[]
mydataout_predict = []
predict_numneigh= []

datacount=0
totalnodefeatures=object_dim

print("*************************")
print("******Done with initial time step************")

totaltimesteps = 1000
stfreq=2

#count different catergory data points, i.e. the number of data points for 1 neighbors, 2 neighbors, 3...
maxneighcount=12
ndatacount=[0]*(maxneighcount)
icount=[0]*(maxneighcount)

#normalization factor for force
normF=1109.6
normV=0.0
normX=rcut

while (timestepcount < totaltimesteps) and (datacount <= 150000):
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
        normV= np.max([abs(normV),abs(velx[i]),abs(vely[i]),abs(velz[i])])
    Lcell=boxx/M
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
                        edgtp=1
                    else:
                        edgtp=0.5
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
        if (numneigh_of_i[i] > 0) and ((timestepcount%stfreq)>-1):
            sub_nodefeatures=np.zeros(((maxneighcount+1),totalnodefeatures),dtype=float)
            if typepos[i]==1:
                massi = masstype1/mass0
                diam_i=diam_1
            if typepos[i]==2:
                massi = masstype2/mass0
                diam_i=diam_2
            for indx in range(numneigh_of_i[i]):
              if indx <= 12:
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
                sub_nodefeatures[indx][1]=0.5*(diam_i+diam_j)/normX
                RXIJ = xpos[j]-xpos[i]
                RYIJ = ypos[j]-ypos[i]
                RZIJ = zpos[j]-zpos[i]
                RXIJ = RXIJ - math.floor ( RXIJ / boxx + 0.5 ) * boxx
                RYIJ = RYIJ - math.floor ( RYIJ / boxy + 0.5 ) * boxy
                RZIJ = RZIJ - math.floor ( RZIJ / boxz + 0.5 ) * boxz
                sub_nodefeatures[indx][2]=RXIJ/normX
                sub_nodefeatures[indx][3]=RYIJ/normX
                sub_nodefeatures[indx][4]=RZIJ/normX
            myx = np.array(sub_nodefeatures) #torch.tensor(np.array(sub_nodefeatures),dtype=torch.float)
            myy = np.array([frx[i], fry[i], frz[i]], dtype=float) #torch.tensor(np.array([frx[i], fry[i], frz[i]]),dtype=torch.float)
            if(numneigh_of_i[i]) <= 12:
              if (timestepcount > 100) and datacount <= 27000:
                mydata_obj.append(myx)
                mydataout.append(myy)
                datacount += 1
                ndatacount[numneigh_of_i[i]-1] +=1

              elif (timestepcount > 100) and datacount <= 30000:
                mydata_obj_val.append(myx)
                mydataout_val.append(myy)
                datacount += 1
                ndatacount[numneigh_of_i[i]-1] +=1

              elif (timestepcount > 100) and datacount <= 150000:
                mydata_obj_predict.append(myx)
                mydataout_predict.append(myy)
                predict_numneigh.append(numneigh_of_i[i])
                datacount += 1
                ndatacount[numneigh_of_i[i]-1] +=1

    line000=f.readline()
    if not line000:
        break
f.close()

for i in range(maxneighcount):
    icount[i]=i

print("length of total dataset=", len(mydata_obj))
print("shape of total dataset=", np.array(mydata_obj).shape)

print("length of total output dataset=", len(mydataout))
print("shape of total output dataset=", np.array(mydataout).shape)

normX=rcut
print("max Force= ", normF)
print("max X =", normX)

################################################################################
############# DATA NORMALIZATION ################
for ii, datai in enumerate(mydataout):
    for jj in range(3):
        datai[jj] /= normF

for ii, datai in enumerate(mydataout_val):
    for jj in range(3):
        datai[jj] /= normF

nmae_ave_p=0.0
nmae_max_p=0.0
for ii, datai in enumerate(mydataout_predict):
    for jj in range(3):
        datai[jj] /= normF

print("Data Point Num=", ndatacount)
print("*************************")
print("******Done with normalization************")


################################################################################
############# DATA TRAINING AND TESTING ################

MYOUTPUTSIZE=3 #fx,fy,fz

inputs_i = tf.keras.Input(shape=(object_dim,))
x1 = layers.Dense(effect_dim, activation='tanh')(inputs_i)
x2 = layers.Dense(effect_dim, activation='tanh')(x1)
outputs_i = layers.Dense(MYOUTPUTSIZE, activation='tanh')(x2)
model_SDN = tf.keras.Model(inputs=inputs_i, outputs=outputs_i)

learning_rate=0.0002
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.mean_squared_error
train_acc_metric = tf.keras.metrics.RootMeanSquaredError()
val_acc_metric = tf.keras.metrics.RootMeanSquaredError()

#speed up version
@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        mypred = model_SDN(x, training=True)
        fullpred=tf.reduce_sum(mypred, 0)
        loss_value = loss_fn(y,fullpred)
    grads = tape.gradient(loss_value, model_SDN.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_SDN.trainable_weights))
    train_acc_metric.update_state(y, fullpred)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits0 = model_SDN(x, training=False)
    val_logits=tf.reduce_sum(val_logits0, 0)
    val_acc_metric.update_state(y, val_logits)

n_epoch = 100
myloss = []
myloss_val = []

for epoch in range(n_epoch):
  print("\nStart of epoch %d" % (epoch,))
  loss_all = 0.0
  nls=0
  for ii, datai in enumerate(mydata_obj):
      myobj=np.array(datai)
      target=np.array(mydataout[ii])
      loss_value = train_step(myobj, target)
      nls +=1
      loss_all += loss_value
  print("loss at epoch ", epoch+1," = ", loss_all/float(nls))

  # Display metrics at the end of each epoch.
  train_acc = train_acc_metric.result()
  print("Training acc over epoch: %.4f" % (float(train_acc),))
  myloss.append(train_acc)

  # Reset training metrics at the end of each epoch
  train_acc_metric.reset_states()

  # Run a validation loop at the end of each epoch.
  for ii, datai in enumerate(mydata_obj_val):
      myobj=np.array(datai)
      target=np.array(mydataout_val[ii])
      # x_single_test=np.concatenate(myobj, axis=1)
      test_step(myobj, target)
  val_acc = val_acc_metric.result()
  myloss_val.append(val_acc)
  val_acc_metric.reset_states()
  print("Validation acc: %.4f" % (float(val_acc),))
