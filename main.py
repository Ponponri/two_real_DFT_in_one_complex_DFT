import argparse
import numpy as np
import matplotlib.pyplot as plt

def DFT(f):
    N = len(f)
    F = np.zeros((N,),dtype=np.complex64)
    
    for i in range(N):
        num = 0
        for j in range(N):
            num += f[j]*np.exp(-2j*np.pi*i*j/N)
        num = np.around(num,6)
        F[i] = num

    return F

def IDFT(f):
    N = len(f)
    F = np.zeros((N,),dtype=np.complex64)
    
    for i in range(N):
        num = 0
        for j in range(N):
            num += f[j]*np.exp(2j*np.pi*i*j/N)
        num /= N
        num = np.around(num,6)
        F[i] = num
    
    return F

parser = argparse.ArgumentParser()
parser.add_argument('--x', default = [5,3,4,1,20,6], type=int, nargs='+') 
parser.add_argument('--y', default = [0,1,0,1,-1,3], type=int, nargs='+') 
args = parser.parse_args()

x = np.asarray(args.x,dtype=np.float32)
y = np.asarray(args.y,dtype=np.float32)
N = len(x)

f = x + 1j*y

F = DFT(f)

Fx = np.zeros((N,),dtype=np.complex64)
for i in range(N):
    Fx[i] = F[i].real - 1j*F[i].imag

X = np.zeros((N,),dtype=np.complex64)
Y = np.zeros((N,),dtype=np.complex64)

for i in range(N):
    m = i
    n = (N-i) % (N)
    X[i] = (F[m] + Fx[n])/2
X = np.around(X,6)

for i in range(N):
    m = i
    n = (N-i) % (N)
    Y[i] = (F[m] - Fx[n])/(2j)
Y = np.around(Y,6)


print(f'x:{x}')
print(f'y:{y}\n')
print(f'f:{f}')
print(f'F:{F}')
print(f'Fx:{Fx}\n')
print('One DFT:')
print(f'X with one DFT:{X}')
print(f'Y with one DFT:{Y}\n')
print('Two DFT:')
print(f'X with np DFT:{DFT(x)}')
print(f'Y with np DFT:{DFT(y)}\n')

print(f'IDFT(X):{IDFT(X)}')
print(f'IDFT(Y):{IDFT(Y)}\n')

print(f'Is X correct: {x==IDFT(X).real.astype(np.float32)}')
print(f'Is Y correct: {y==IDFT(Y).real.astype(np.float32)}\n')


t = np.linspace(0,N-1,N)
fig = plt.figure()
plt.suptitle(f'Implement two real DFT in one DFT\n\n \
             x: {x}\ny: {y}\n \
             X: {X}\nY: {Y}\n')
plt.subplot(221)
plt.title('X.real')
plt.bar(t,X.real)
plt.subplot(222)
plt.title('X.imag')
plt.bar(t,X.imag)
plt.subplot(223)
plt.title('Y.real')
plt.bar(t,Y.real)
plt.subplot(224)
plt.title('Y.imag')
plt.bar(t,Y.imag)
fig.tight_layout()
plt.savefig('result.jpg')
plt.show()
