# Compute MPEP
h=0.005
nT=10000
x1=np.zeros(nT,)
x2=np.zeros(nT,)
x1[1]=-0.01
x2[1]=0.01
for epoch in range(nT-2):
    x01=tf.constant(x1[epoch+1],shape=(1,),dtype=tf.float32)
    x02=tf.constant(x2[epoch+1],shape=(1,),dtype=tf.float32)
    x0= tf.stack([x01,x02],axis=1)
    p=model.predict(x0)
    p =tf.reduce_sum(p, axis=0)
    p1=p.numpy()
    gamma=1.0
    x1[epoch+2]=x1[epoch+1] -h * (x1[epoch+1]-np.power(x1[epoch+1],3)-gamma*x1[epoch+1]*np.power(x2[epoch+1],2)+p1[0])
    x2[epoch+2]=x2[epoch+1] -h * (-(1+np.power(x1[epoch+1],2))*x2[epoch+1]+p1[1])

MPEP=tf.stack([x1,x2], axis=1)
scio.savemat('MPEP.mat',{'MPEP':MPEP.numpy()})
