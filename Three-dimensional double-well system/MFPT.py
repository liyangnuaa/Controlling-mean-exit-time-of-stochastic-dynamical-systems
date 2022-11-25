# Compute controlled MFPT
h=0.001
nT=10000000000
Npath=1000
N=0
FPT=np.zeros(Npath,)
xmax=0
D=0.16
rou=0.3
cd=-0.0460
x1=-1.0*np.ones(Npath,)
x2=np.zeros(Npath,)
x3=np.zeros(Npath,)
for epoch in range(nT):
    n=x1.size
    x01=tf.constant(x1,shape=(n,),dtype=tf.float32)
    x02=tf.constant(x2,shape=(n,),dtype=tf.float32)
    x03=tf.constant(x3,shape=(n,),dtype=tf.float32)
    x0= tf.stack([x01,x02,x03],axis=1)
    p=model.predict(x0)
    p1=p.numpy()
    x4=x1 +h * (-2*(np.power(x1,3)-x1)- rou*(x2+x3)+cd*p1[:,0]) +np.sqrt(D*h)*np.random.normal(size=(n,))
    x5=x2 +h * (-x2+2*rou*(np.power(x1,3)-x1)+cd*p1[:,1]) +np.sqrt(D*h)*np.random.normal(size=(n,))
    x6=x3 +h * (-x3+2*rou*(np.power(x1,3)-x1)+cd*p1[:,2]) +np.sqrt(D*h)*np.random.normal(size=(n,))
    
    I=x4>xmax
    n0=x4[I].size
    if n0>0:
        FPT[N:N+n0]=epoch*h
        x4=x4[~I]
        x5=x5[~I]
        x6=x6[~I]
        N=N+n0
        print(epoch,x4.size)
        
    if x4.size==0:
        break
    x1=x4
    x2=x5
    x3=x6

MFPT=np.sum(FPT)/Npath
print(FPT)
print(MFPT)
scio.savemat('FPT.mat',{'FPT':FPT})
