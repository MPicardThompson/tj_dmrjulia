module dmrg
using parameters
using MPSutil
using MPOutil
using Operators
# using PyPlot
#using functions
using Davidson
#if basic.tensorops
#  using TensorOperations
#else
  using contractions
#end

const realOrComplex = Union{Float64,Complex128}

#determines true/false for left/right sweeps
function condition(j::Int64,basic::params)
  return (j < 0 && j!= basic.Ns)
end

#       +-----------+
#>------|    DMRG   |-------<
#       +-----------+

function DMRG(psi::MPS,H::Any,basic::params)

QS = convert(Int64,basic.qstates)
MPO = convert2MPO(H,basic)
numsites = size(psi.A,1)
centerbond = floor(Int64,div(numsites,2))

move!(psi,1,basic) # if it isn't already...

 maxtrunc = 0
 maxmUsed = 0
 currEnergy = 10000

vecHL = Array{Array{Float64,3},1}(basic.Ns) #we can save time (but not memory) by allocating the left and right blocks to vectors
vecHL[1] = ones(1,1,1)
vecHR = Array{Array{Float64,3},1}(basic.Ns) #each site has a left and a right
vecHR[numsites] = ones(1,1,1)
tHR = vecHR[numsites]
for k = numsites:-1:3
    ket = psi.A[k]
    bra = conj(ket)
    W = MPO[k]
#      tHR = contract(bra,3,tHR,3)
#      tHR = contract(W,[2 4],tHR,[2 4])
#      tHR = contract(ket,[2 3],tHR,[1 4])
    mps2mpo = contract(ket,2,W,2)
    tHR = contract(mps2mpo,[2 5],tHR,[1 2])
    tHR = contract(tHR,[3 4],bra,[2 3])
    vecHR[k-1] = tHR
  end

  tildeA = zeros(basic.maxiter,basic.maxiter) #initializing the ~A matrix.
  twositeH = Array{Array{Float64,6},1}(basic.Ns-1)
  for i = 1:basic.Ns-1
    W1 = MPO[i]
    W2 = MPO[i+1]
    twositeH[i] = contract(W1,4,W2,1)
  end

for n = 1:basic.nsweeps
 @time for j = -numsites+1:numsites-1 #SRW trick for sweeping from left to right in one command
  j == 0 && continue #zero has no meaning here
  i = j < 0 ? numsites - abs(j) : numsites + 1 - j #site index

#       +------------------------+
#>------|   LR blocks (stored)   |-------<
#       +------------------------+
#constructs HL and HR which are the environment blocks but stores them for faster retrieval
#this incurs a memory penalty that scales with bond dimension
  limL = condition(j,basic) ? i-1 : i-2 #limits for left or right sweeps
  if limL != 0 && condition(j,basic) #since the first tensor is always fixed to be the identity
    tHL = vecHL[limL] #confusing! but this is the left block to the left of site limL
    ket = psi.A[limL]
    bra = conj(ket)
    W = MPO[limL]
#    mps2mpo = contract(W,3,bra,2)
#    tHL = contract(tHL,[2 3],mps2mpo,[1 4])
#    tHL = contract(ket,[1 2],tHL,[1 2])
    mps2mpo = contract(ket,2,W,2)
    tHL = contract(tHL,[2 3],mps2mpo,[1 3])
    tHL = contract(bra,[1 2],tHL,[1 3])
    vecHL[limL+1] = tHL
  end
  HL = vecHL[limL+1]

  limR = condition(j,basic) ? i+2 : i+1
  if limR <= basic.Ns && !condition(j,basic)
    tHR = vecHR[limR]
    ket = psi.A[limR]
    bra = conj(ket)
    W = MPO[limR]
#      tHR = contract(bra,3,tHR,3)
#      tHR = contract(W,[2 4],tHR,[2 4])
#      tHR = contract(ket,[2 3],tHR,[1 4])
    mps2mpo = contract(ket,2,W,2)
    tHR = contract(mps2mpo,[2 5],tHR,[1 2])
    tHR = contract(tHR,[3 4],bra,[2 3])
    vecHR[limR-1] = tHR
  end
  HR = vecHR[limR-1]

#       +-----------------+
#>------| Big Hamiltonian |-------<
#       +-----------------+
#contructs the Hamiltonian for diagonalization

A = condition(j,basic) ? psi.A[i] : psi.A[i-1] #the two tensors being updated
B = condition(j,basic) ? psi.A[i+1] : psi.A[i]

#       +-----------+
#>------|  Davidson |-------<
#       +-----------+
  ops = condition(j,basic) ? twositeH[i] : twositeH[i-1]
  AA = contract(A,3,B,1)
  AAops = contract(AA,[2 3],ops,[2 4])#,[1 4])
  HamPsi = contract(HL,[2 3],AAops,[1 3])
  HamPsi = contract(HamPsi,[2 5],HR,[1 2])

  bonddim = condition(j,basic) ? size(A,3) : size(B,1)
  #basic.verbose ? print("step ",n," ") : 0
  #basic.verbose ? (condition(j,basic) ? print("left") : print("right")) : 0
  #basic.verbose ? println(" sweep at site $i (m = $bonddim)") : 0

#println("davidson")
gr,currEnergy = davidson(HamPsi,AA,tildeA,ops,HL,HR,basic)


#       +-----------------+
#>------|  Orthogonality  |-------<
#       +-----------------+

#basic.verbose ? println("orthogonality restoration") : 0

 newAA = reshape(gr,size(AA,1),size(AA,2),size(AA,3),size(AA,4))

 cont = condition(j,basic) ? 2 : 1 #trace site i+1 (i) if we sweep to the left (right)
 rAA = reshape(newAA,size(AA,1)*size(AA,2),size(AA,3)*size(AA,4))
 densrho = contract(rAA,cont,conj(rAA),cont) #1: form density matrix on one site

 densrho += basic.noise*rand(size(densrho)) #2: add noise
 densrho *= 1/trace(densrho) #normalize so we don't loose anything
 densrho = (densrho + densrho')/2 #symmetrize to ensure sorted eigenvalues

# Dsq,U = eig(densrho)#doing "eig" gives U that is actually U' for U.D.U'
  if (basic.regSVD)
    Dsq,U = eig(densrho)
  else
    if (basic.maxm > minimum(size(AA)))
      svdobj = eig(AA)
      Dsq = svdobj[1]
      U = svdobj[2]
    else
      svdobj = eigs(AA;ncv = basic.maxm)
      Dsq = svdobj[1]
      U = svdobj[2]
    end
  end
  sizeDsq = size(Dsq,1)

 #to truncation of D and U
 truncerr = 0.
 if basic.cutoff > 0.
   compare = basic.maxm < sizeDsq ? 1-sum(i->Dsq[i],(sizeDsq-basic.maxm):sizeDsq) : 1
   p = 1
   if compare > basic.cutoff
   truncerr = 0.
   for q = 1:sizeDsq-1
     truncerr+Dsq[q] >= basic.cutoff ? break : p+=1
     truncerr += Dsq[q]
   end
   end
   m = min(basic.maxm-1,sizeDsq-p)
   else
   m = min(basic.maxm-1,sizeDsq)
 end
 lowtrunc = sizeDsq-m
 Dsq = Dsq[lowtrunc:sizeDsq] #because of how eig sorted the symmetric eigenvalues...least to greatest
 sizeDsq = size(Dsq,1)
 U = U[:,lowtrunc:size(U,2)]

 if i == centerbond && condition(j,basic)
   SvN = -sum(h->Dsq[h]*log(abs(Dsq[h])),sizeDsq:-1:1)#sometimes the zero value ~10^-25 is negative...implies absolute value sign
   println("SvN at center bond b=",centerbond," = ",SvN)
   println("Eigs at center bond b=",centerbond,": ",Float64[Dsq[y] for y = size(Dsq,1):-1:size(Dsq,1)-min(max(1,floor(Int64,size(Dsq,1)/5)),basic.eigsdisplay)])
 end

#err = 1-sum(h->Dsq[h],1:size(Dsq,1))
maxtrunc = truncerr > maxtrunc ? truncerr : maxtrunc
testMaxm = sizeDsq
maxmUsed = testMaxm > maxmUsed ? testMaxm : maxmUsed

 #basic.verbose ? println(condition(j,basic)," ",i) : 0
 if condition(j,basic)
   trunc = convert(Int64,div(size(U,1)*size(U,2),size(newAA,1)*QS))
   U = reshape(U,size(newAA,1),QS,trunc)
   psi.A[i] = U
   mult = contract(U,[1 2],newAA,[1 2])
   psi.A[i+1] = reshape(mult,trunc,QS,size(newAA,4))
 else
   trunc = convert(Int64,div(size(U,1)*size(U,2),QS*size(newAA,4)))
   U = reshape(U',trunc,QS,size(newAA,4))
   mult = contract(newAA,[4 3],U,[3 2])
   psi.A[i-1] = reshape(mult,size(newAA,1),QS,trunc)
   psi.A[i] = U
 end

psi.oc = condition(j,basic) ? i+1 : i-1

end
println("Sweep $n, Largest truncation = $maxtrunc, m = $maxmUsed")
println("Energy at sweep $n is $currEnergy")
println()

end

return currEnergy

end

export DMRG

end

using dmrg
