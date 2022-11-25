# Compute controlled MFPT
h=0.001
nT=10000000000
Npath=1000
N=0
FPT=np.zeros(Npath,)
xmax=0
D=0.05
gamma=1.0
cd=0.2035
x1=-1.0*np.ones(Npath,)
x2=np.zeros(Npath,)
for epoch in range(nT):
    n=x1.size
    x01=tf.constant(x1,shape=(n,),dtype=tf.float32)
    x02=tf.constant(x2,shape=(n,),dtype=tf.float32)
    x0= tf.stack([x01,x02],axis=1)
    p=model.predict(x0)
    p1=p.numpy()
    x3=x1 +h * (x1-np.power(x1,3)-gamma*x1*np.power(x2,2)+cd*p1[:,0]) +np.sqrt(D*h)*np.random.normal(size=(n,))
    x4=x2 +h * (-(1+np.power(x1,2))*x2+cd*p1[:,1]) +np.sqrt(D*h)*np.random.normal(size=(n,))
    
    I=x3>xmax
    n0=x3[I].size
    if n0>0:
        FPT[N:N+n0]=epoch*h
        x3=x3[~I]
        x4=x4[~I]
        N=N+n0
        print(epoch,x3.size)
        
    if x3.size==0:
        break
    x1=x3
    x2=x4

MFPT=np.sum(FPT)/Npath
print(FPT)
print(MFPT)
scio.savemat('FPT.mat',{'FPT':FPT})
