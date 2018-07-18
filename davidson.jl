module Davidson
using parameters
#if basic.tensorops
  using TensorOperations
#else
  using contractions
#end

const realOrComplex = Union{Float64,Complex128}

function davidson(HamPsi::Array{T,4},AA::Array{T,4},tildeA::Array{Float64,2},ops::Array{T,6},HL::Array{T,3},HR::Array{T,3},basic::params) where T <: realOrComplex
  #could simplify to only one tensor in the above contractions for AA and HamPsi, then change this line to size(A,1)*size(A,2)*size(B,2)*size(B,3)
  veclength = size(AA,1) * size(AA,2) * size(AA,3) * size(AA,4)

  numeigs = 1 #need to change from 1 x 1 matrix if we want more eigenvalues
  savelam = 1000 #dummy for first iteration
  Ab = zeros(basic.maxiter,veclength) #stores all products A*b_i
  bi = zeros(basic.maxiter,veclength) #vectors found from process
  qm = zeros(veclength) #current update vector (m+1th vector)

  gr = 0 #result (dummy)

  #Part A
  almost = reshape(AA,veclength) #+ tempvec
  bi[numeigs,:] = almost/norm(almost) #set first elements
  Ab[numeigs,:] = reshape(HamPsi,veclength)

  #could simplify to one line...
  for k = 1:numeigs, w = 1:numeigs
    tildeA[k,w] = ((bi[k,:]')*Ab[w,:])  # =tilde(A), "[1]" converts to integer
  end

   for p = 1:basic.maxiter
    viewer = [p for p = 1:numeigs]
    lam,vecs = eig(tildeA[viewer,viewer])

    alpha = reshape(vecs[:,1]/norm(vecs[:,1]),size(vecs[:,1],1),1)

    #Part B
    qm = zeros(veclength) #current update vector (m+1th vector)
    for a = 1:numeigs
      qm =  (a==1 ? qm : 0.) + alpha[a] * (Ab[a,:] - lam[1] * bi[a,:])
    end
    # qm = sum(a->alpha[a] * (Ab[a,:] - lam[1] * bi[a,:]),1:numeigs)

    #Part C: Convergence
    #could also be another test here...
    if norm(qm) == 0. && p == 1
      gr = bi[1,:]
      #basic.verbose ? println("already in eigenvector in Davidson, exiting...") : 0
      savelam = deepcopy(lam[1])
      break
    end

    if (abs(lam[1] - savelam) < basic.errgoal) | (numeigs+1>size(lam,1) && p>1) | (norm(qm) < basic.effZero)
      #basic.verbose ? println("converged! sub-block energy = ",lam[1]) : 0
      gr = sum(a->alpha[a] * bi[a,:],1:numeigs) #sum(m->(alpha[1]' * bi)[m],1:size(alpha,1))
      savelam = deepcopy(lam[1])
      break
    else
      #basic.verbose ? println("continuing... ",p) : 0
      savelam = deepcopy(lam[1])
    end

  #Part D: Conditioner
  #set to identity now, but can help...
  #safety=1E-10
  #  xi = qm #[qm[r]/(lam - reshapeHam[r,r] + safety) for r = 1:size(qm,1)]

  #Part E: Graham-Schmidt orthonormalization
  #actually Modified Graham-Schmidt (since we reuse the normalized vector)
  #  dm = xi
  #note: two things really made this fast.  One was not forming the identity here.
  #the other way not forming the kron product
  #yet a third item was carefully constructing all the tensor contractions
    for k = 1:numeigs
      overlap = (conj(bi[k,:]') * qm)
      qm = qm - overlap * bi[k,:]/norm(bi[k,:])
      qm = qm/norm(qm)
    end

    bi[numeigs+1,:] = qm

  #Part I
    AA = reshape(qm,size(AA,1),size(AA,2),size(AA,3),size(AA,4))

    tempAA = contract(AA,[2 3],ops,[2 4])#,[1 4])
    Abnext = contract(HL,[2 3],tempAA,[1 3])
    Abnext = contract(Abnext,[2 5],HR,[1 2])

  #  push!(Ab,Abnext)
    Ab[numeigs+1,:] = reshape(Abnext,veclength)
    for k = 1:numeigs+1,w = 1:numeigs+1
      if !(k < numeigs+1 && w < numeigs+1)
        tildeA[k,w] = ((bi[k,:]') * Ab[w,:])
        k != w ? tildeA[w,k] = tildeA[k,w] : 0
      end
    end
    numeigs += 1
  end

  return gr,savelam

end

export davidson

end

using Davidson
