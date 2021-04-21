from numpy import *
from numpy.linalg import inv
#time step of mobile movement
# dt = 0.1
# vx=1
# vy=1
# # Initialization of state matrices
# X = array([[0.0], [0.0]])
# P = diag((0.01, 0.01, 0.01, 0.01))
# B= array([[dt, 0], [0, dt]])
#
# A = eye(X.shape()[0])
# U = array([[vx], [vy]])



# Measurement matrices

# R = eye(X.shape()[0])
# Number of iterations in Kalman Filter
# N_iter = 50
# # Applying the Kalman Filter
# for i in arange(0, N_iter):
#     Y=over.somethig...
#     (X, P) = kf_predict(X, P, A, B, U)
#     (X, P, K, IM, IS, LH) = kf_update(X, P, Y, A, R)


from numpy import dot
def kf_predict(X, P, A, B, U,Q):
     X = dot(A, X) #+ dot(B, U)
     P = dot(A, dot(P, A.T)) +Q
     return(X,P)

from numpy import dot, sum, tile, linalg
from numpy.linalg import inv

def kf_update(X, P, Y, A, R):
     IM = dot(A, X)
     IS = R + dot(A, dot(P, A.T))
     K = dot(P, dot(A.T, inv(IS)))
     X = X + dot(K, (Y-IM))
     P = P - dot(K, dot(IS, K.T))
     LH = gauss_pdf(Y, IM, IS)
     return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):
     if M.shape[1] == 1:
         DX = X - tile(M, X.shape[1])
         E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
         E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
         P = exp(-E)
     elif X.shape[1] == 1:
         DX = tile(X, M.shape[1])- M
         E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
         E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
         P = exp(-E)
     else:
         DX = X-M
         E = 0.5 * dot(DX.T, dot(inv(S), DX))
         E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
         P = exp(-E)
     return (P[0],E[0])