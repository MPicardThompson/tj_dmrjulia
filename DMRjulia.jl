


#1. Go through the code and come back with any notes
#2. Run a spin model (spin-1/2)
#3. Run the XX model
#4. Run the Hubbard model
#5. Run the t-J model and compre with literature anywhere (this might not work depending on what parameters you used, so talk to me!)



 include("parameters.jl")
 include("contractions.jl")
 include("MPOutil.jl")
 include("MPSutil.jl")
 include("Operators.jl")
 include("davidson.jl")
 include("dmrg.jl")

 const L = 100 ##total number of sites

 basic = params()
 basic.nsweeps = 75
 basic.cutoff = 1e-10
 basic.Ns = L
 basic.maxm = 25

#       +-------+
#>------|  MPS  |-------<
#       +-------+
#include("MPOutil.jl")
#include("MPSutil.jl")

S,S2,Sz,Sx,Sy,Sp,Sm = spinOps(basic)

initTensor = [zeros(1,basic.qstates,1) for i=1:basic.Ns]

for i = 1:basic.Ns
   initTensor[i][1,i%2==1 ? 1 : 2,1] = 1.0
end

#=
for i = 1:basic.Ns
   initTensor[i][1,1,1] = 1.0
end

for i = 1:basic.Nup
  initTensor[i] = contract([2,1,3],full(Cup'),2,initTensor[i],2)
  for j = 1:i-1
    initTensor[j] = contract([2,1,3],full(Fup),2,initTensor[j],2)
  end
end
for i = 1:basic.Ndn
  initTensor[i] = contract([2,1,3],full(Cdn'*Fdn),2,initTensor[i],2)
  for j = 1:i-1
    initTensor[j] = contract([2,1,3],full(Fdn),2,initTensor[j],2)
  end
end
=#

psi = MPS(initTensor,1) #oc on site 1 ok here?

#       +-------+
#>------|  MPO  |-------<
#       +-------+

onsite(i::Int64) = mu * Ndens + UU[i] * Nup * Ndn

function H(i::Int64)
    return full([Id  O O O O;
        Sp O O O O;
        Sm  O O O O;
        Sz O O O O;
        O Sm/2 Sp/2 Sz Id])
end

#       +-------+
#>------|  DMRG |-------<
#       +-------+

@time currenergy = DMRG(psi,H,basic)


#using contractions
#expMPO = CPLXconvert2MPO(tevolH,basic)
#psit = deepcopy(psi)

move!(psi,1,basic)
