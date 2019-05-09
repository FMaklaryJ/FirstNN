import numpy as np
import copy



#------Defining step function and sigmoid function
def sigmoid(x):
    s=2*1/(1+np.exp(-x))-1
    return s

def step(x):
    s=0.0
    if x>0:
        s=1.0
    return s


#---------Defining neural network----------------
class NN2:
    def __init__(self,x,NL):
        #NL is a row vector containing the amount of neurons
        #in each corresponding layer
        #X is input, also a row vector
        
        NLS=np.zeros([1,NL.size])
        NLS[0,0]=NL[0]+x.shape[1]
        for i in range(NL.size-1):
            NLS[0,i+1]=NLS[0,i]+NL[i+1]
        self.input=x #input
        self.weights={}
        self.weights['0']=np.zeros([self.input.shape[1],NL[0]])
        for i in range(NL.size-1):
            self.weights[str(i+1)]=np.zeros([int(NLS[0,i]),int(NL[i+1])])
        
        self.out=np.zeros([1,NL[NL.size-1]])
        self.layercount=NL.size

        self.lay={}        
        self.lay['0']=self.input
    def ff(self):
        vstep=np.vectorize(step)

        for i in range(self.layercount-1):

            self.lay[str(i+1)]=np.append(self.lay[str(i)],vstep(np.dot(self.lay[str(i)],self.weights[str(i)])),axis=1)        
        
        
        j=self.layercount-1
        self.lay[str(j+1)]=vstep(np.dot(self.lay[str(j)],self.weights[str(j)]))
        self.out=self.lay[str(self.layercount)]

    def bp(self,thres):
        vevo=np.vectorize(evo)
        for i in range(len(self.weights)):
            self.weights[str(i)]=vevo(self.weights[str(i)],thres)

    def ffsig(self):
        vstep=np.vectorize(sigmoid)

        for i in range(self.layercount-1):

            self.lay[str(i+1)]=np.append(self.lay[str(i)],vstep(np.dot(self.lay[str(i)],self.weights[str(i)])),axis=1)        
        
        
        j=self.layercount-1
        self.lay[str(j+1)]=vstep(np.dot(self.lay[str(j)],self.weights[str(j)]))
        self.out=self.lay[str(self.layercount)]

            
def evo(w,thres):
    n=np.random.rand(1)
    n=n[0]
    x=w
    if n<=thres:
        x=np.random.rand(1)
        x=2*x[0]-1
    return x


#-------------Here we do stuff---------------------#
x=np.matrix([[1.0,0.0], [1.0,1.0],[0.0,0.0],[0.0,1.0]]) #input
bias=np.ones([x.shape[0],1])
#x=np.append(bias,x,axis=1)

NL=np.array([1,1])#Neurons (change this if you feel like it)

mut=float(input("Enter mutation rate:")) #mutation rate
c=int(input("Enter number of AI's per generation:")) #number of ai's per run
generations=int(input("Enter number of generations:"))


want=np.matrix([[1.0], [0.0],[0.0],[1.0]]) #XOR gate
fits0=100

N0=NN2(x,NL)
N0.bp(0.5)
N0.ff()
net={}


#training with stochastic variation:
for g in range(generations):
    for i in range(c):
        
        net[str(i)]=copy.deepcopy(N0) #initiation
        net[str(i)].bp(mut) #mutate
        net[str(i)].ff() #get result
    
    for i in range(c):
        fits=want-net[str(i)].out
        fits=sum(abs(fits))

        if fits<fits0:
            N0=net[str(i)]
            fits0=copy.copy(fits)

print("Output:")
print(N0.out)
print("Wanted output:")
print(want)
endstring=input("Click enter to exit.")





