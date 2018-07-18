#makes several utilities for MPSs
module MPSutil
using parameters
#using functions
using MPOutil
#if basic.tensorops
  using TensorOperations
#else
  using contractions
#end

const realOrComplex = Union{Float64,Complex128}

type MPS
 A::Array{Array{T,3},1} where T <: realOrComplex
 oc::Int64
end
#=
function mSVD(AA::Array{T,2},basic::params) where T <: realOrComplex
  if ((basic.maxm > minimum(size(AA))) && basic.regSVD)
    svdobj = svd(AA)
    U = svdobj[1]
    D = svdobj[2]
    V = svdobj[3]'
  else
    svdobj = svds(AA;nsv = basic.maxm)
    U = svdobj[1][:U]
    D = svdobj[1][:S]
    V = svdobj[1][:Vt]
  end
  truncerr = 0. #1-sum(i->D[i]^2,1:newbond-1)
  newbond = size(D,1)
  while (truncerr < basic.cutoff)
    truncerr += D[newbond]^2
    newbond -= 1
  end
  newbond = min(newbond+1,basic.maxm)
  return U[:,1:newbond],D[1:newbond],V[1:newbond,:]
#    println(D[size(D,1)]," ",D[1])
#    println(size(D,1))
#    println(sum(i->D[i]^2,1:size(D,1)))
#
#    if (truncerr < basic.cutoff)# && (size(D,1) > basic.maxm)
#        newbond = 1
#        while (truncerr < basic.cutoff)
#         newbond += 1
#         truncerr += Dsq[newbond]
#        end
#        newbond = min(newbond-1,basic.maxm)
##      println("truncerr = ",truncerr)
#      return U[:,1:newbond],D[1:newbond],V[1:newbond,:]
#    else
#      return U,D,V
#    end
#
end
=#

function mSVD(AA::Array{T,2},basic::params) where T <: realOrComplex
    if (basic.regSVD)
      U,D,V = svd(AA)
    else
      if (basic.maxm > minimum(size(AA)))
        svdobj = svd(AA)
        U = svdobj[1]
        D = svdobj[2]
        V = svdobj[3]'
      else
        svdobj = svds(AA;nsv = basic.maxm)
        U = svdobj[1][:U]
        D = svdobj[1][:S]
        V = svdobj[1][:Vt]
      end
    end
    if size(D,1) > basic.maxm
     compare = basic.maxm < size(D,1) ? 1-sum(i->D[i]^2,1:basic.maxm) : 1
     p = size(D,1)
     if compare < basic.cutoff
       truncerr = 0.
       while (truncerr >= basic.cutoff)
        truncerr += D[p]^2
        p -= 1
       end
     end
    m = min(basic.maxm,max(p,1))
    else
    m = min(basic.maxm,size(D,1))
    end
    Utrunc = U[:,1:m]
    Dtrunc = D[1:m]
    Vtrunc = V[:,1:m]'
    return Utrunc,Dtrunc,Vtrunc
end

function move!(mps::MPS,pos::Int64,basic::params)
  currpos = mps.oc
  QS = convert(Int64,basic.qstates)
  while currpos != pos
  if currpos < pos
      A1 = mps.A[currpos]
      A2 = mps.A[currpos+1]
    else
      A1 = mps.A[currpos-1]
      A2 = mps.A[currpos]
  end
  if !basic.tensorops
    AA = contract(A1,3,A2,1)
  else
    @tensor begin
      AA[a,b,bp,d] := A1[a,b,c] * A2[c,bp,d]
    end
  end
    sizeL = size(AA,1)*size(AA,2)
    sizeR = size(AA,3)*size(AA,4)
    U,D,V = mSVD(reshape(AA,sizeL,sizeR),basic)
  if currpos < pos
    newdim = div(size(U,1)*size(U,2),size(AA,1)*QS)
    mps.A[currpos] = reshape(U,size(AA,1),QS,newdim)
    DV = Float64[D[p]*V[p,q] for p=1:size(V,1),q=1:size(V,2)]
    mps.A[currpos+1] = reshape(DV,newdim,QS,size(AA,4))
    currpos = currpos + 1
  else
    newdim = div(size(V,1)*size(V,2),size(AA,4)*QS)
    UD = Float64[U[p,q]*D[q] for p=1:size(U,1),q=1:size(U,2)]
    mps.A[currpos-1] = reshape(UD,size(AA,1),QS,newdim)
    mps.A[currpos] = reshape(V,newdim,QS,size(AA,4))
    currpos = currpos - 1
  end
  end
  mps.oc = pos
end

#applies MPO tensors to MPSs exactly from left to right.  This is then truncated
#on the return

#add approximate contraction scheme later

function MPO2MPS(psi::Array,MPO::Array,basic::params)
    retpsi = [zeros(Complex128,size(psi[p])) for p = 1:size(psi,1)]
    QS = convert(Int64,basic.qstates)
  for i = 1:basic.Ns
    phi = psi[i]
    W = MPO[i]
    if !basic.tensorops
      B = contract([2,4,1,3,5],W,1,phi,2)
    else
      @tensor begin
        B[a1,b1,s2,a2,b2] := W[s1,s2,a1,a2] * phi[b1,s1,b2]
      end
    end
    retpsi[i] = reshape(B,size(B,1)*size(B,2),QS,size(B,4)*size(B,5))
  end
  #orthogonality was destroyed in appying the full MPO.  We use SVDs to restore it.
  tempbasic = basic
  for i = 1:basic.Ns-1
    AA1 = retpsi[i]
    BB1 = retpsi[i+1]
    if !basic.tensorops
      AA = contract(AA1,3,BB1,1)
    else
      @tensor begin
        AA[a,s1,s2,b] := AA1[a,s1,c] * BB1[c,s2,b]
      end
    end
    tempbasic.maxm = basic.maxm * size(retpsi[i],3)
    left,right = size(AA,1)*size(AA,2),size(AA,3)*size(AA,4)
    U,D,V = mSVD(reshape(AA,left,right),tempbasic)
    left,bond = size(AA1,1),div(size(U,1)*size(U,2),size(AA1,1)*QS)
    retpsi[i] = reshape(U,left,QS,bond)
    mult = [(D[p]*V[p,q])[1] for p=1:size(V,1),q=1:size(V,2)]
    right = div(size(mult,1)*size(mult,2),bond*QS)
    retpsi[i+1] = reshape(mult,bond,QS,right)
  end
  #now to truncate properly on the way back
  for i = basic.Ns:-1:2
    AA1 = retpsi[i-1]
    BB1 = retpsi[i]
    if !basic.tensorops
      AA = contract(AA1,3,BB1,1)
    else
      @tensor begin
        AA[a,s1,s2,b] := AA1[a,s1,c] * BB1[c,s2,b]
      end
    end
    left,right = size(AA,1)*size(AA,2),size(AA,3)*size(AA,4)
    U,D,V = mSVD(reshape(AA,left,right),basic)
    bond,right = div(size(V,1)*size(V,2),size(BB1,3)*QS),size(BB1,3)
    retpsi[i] = reshape(V,bond,QS,right)
    mult = [(U[p,q]*D[q])[1] for p=1:size(U,1),q=1:size(U,2)]
    left = div(size(mult,1)*size(mult,2),bond*QS)
    retpsi[i-1] = reshape(mult,left,QS,bond)
  end
  return retpsi
end

#TODO: Expand to include two point and higher correlation functions
function corrfct(psi::MPS,H::Any,basic::params)
  MPO = convert2MPO(H,basic)
  E = ones(1,1,1,1,1,1)
  for i = 1:basic.Ns
  A = psi.A[i]
  Adag = conj(A)
  W = MPO[i]
  if !basic.tensorops
    E = contract(E,4,A,1)
    E = contract(E,[4 6],W,[3 1])
    E = contract(E,[4 6],Adag,[1 2])
  else
    @tensor begin
      E[a,b,c,y,z,s,ap] := E[a,b,c,x,y,z] * A[x,s,ap]
      E[a,b,c,z,ap,sp,bp] := E[a,b,c,y,z,s,ap] * W[s,sp,y,bp]
      E[a,b,c,ap,bp,cp] := E[a,b,c,z,ap,sp,bp] * Adag[z,sp,cp]
      #with contractions: E[a,b,c,ap,bp,cp] := E[a,b,c,z,ap,sp,bp] * Adag[z,sp,cp]
    end
  end
  end
  return E[1,1,1,1,1,1]
end

#
# Union({Float64,Complex128})
#

function applygates(gates::Array{Array{T,4},1},psi::Array{Array{T,3},1},basic::params) where T <: realOrComplex
  QS = convert(Int64,basic.qstates)
#  println("here in apply gates")
  #forward
  for i = 1:basic.Ns-1
    AA = contract(psi[i],3,psi[i+1],1)
    AA = contract([3 1 2 4],gates[i],[1 2],AA,[2 3])
    rAA = reshape(AA,size(AA,1)*size(AA,2),size(AA,3)*size(AA,4))
    U,D,V = mSVD(rAA,basic)
    psi[i] = reshape(U,size(psi[i],1),QS,size(U,2))
    psi[i+1] = reshape(Diagonal(D)*V,size(U,2),QS,size(psi[i+1],3))
  end
  #backward
  for i = basic.Ns:-1:2
    AA = contract(psi[i-1],3,psi[i],1)
    AA = contract([3 1 2 4],gates[i-1],[1 2],AA,[2 3])
    rAA = reshape(AA,size(AA,1)*size(AA,2),size(AA,3)*size(AA,4))
    U,D,V = mSVD(rAA,basic)
    psi[i-1] = reshape(U*Diagonal(D),size(psi[i-1],1),QS,size(U,2))
    psi[i] = reshape(V,size(U,2),QS,size(psi[i],3))
  end
  return psi
end

function overlapPsi(psiT::Array{Array{T,3},1},psi::Array{Array{T,3},1},basic::params) where T <: realOrComplex
  res = contract([1,3,2,4],conj(psiT[1]),2,psi[1],2)
  num = size(psi,1)
  for i = 2:num
    res = contract(res,4,psi[i],1)
    res = i != num ? contract([1 2 4 3],res,[3 4],conj(psiT[i]),[1 2]) : contract(res,[3 4],conj(psiT[i]),[1 2])
  end
  return res[1,1,1,1]
end

##################
#quantum number stuff

type qMPS
 A::Array{Array{Float64,3},1}
 oc::Int64
 LQN::Array#quantum numbers for the LEFT tensors
 SQN::Array #physical index quantum numbers
 RQN::Array#quantum numbers for the RIGHT tensors
end

export MPS,mSVD,move!,MPO2MPS,corrfct,applygates,overlapPsi

end

using MPSutil
