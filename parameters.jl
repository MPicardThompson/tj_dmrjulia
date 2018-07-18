module parameters
#       +-------------------------+
#>------| Immutable params holder |--------<
#       +-------------------------+
mutable struct params
 Na::Any
 Nf::Any
 Z::Any
 R::Any
 Delta::Any
 c::Any
 L::Any
 Ns::Any
 Nup::Any
 Ndn::Any
 s::Any
 maxm::Any
 nsweeps::UInt
 cutoff::Float64
 qstates::UInt
 spins::Bool
 fermions::Bool
 lanczos::Bool
 maxiter::UInt
 verbose::Bool
 noise::Float64
 effZero::Float64
 errgoal::Float64
 deltaT::Float64
 ntime::Int64
 eigsdisplay::Int64
 subparallelizations::Bool
 A::Float64
 kappa::Float64

 regSVD::Bool
 tensorops::Bool
 memVtime::Bool
 singlesite::Bool
 monitor::Bool
 saveplot::Bool
end

export params


#       +-------------------+
#>------| System parameters |--------<
#       +-------------------+
 const spins = true
 const fermions = false #false is hardcore bosons

 const s = 1/2 # spin of the model
 const qstates = (spins | !fermions) ? convert(Int64,2s+1) : 2*convert(Int64,2s+1)
 const QNums = (spins | !fermions) ? sum(p->p,-2s:2s) : 4
               #spins: -l,-l+1,-l+2,...0,1,2,...l-1,l/Hubbard: fixed at four for now
               #only handles homogeneous quantum numbers
               #would need to make an interface that handles a vector of these...
               #add adjacent and increment in spins (think Clebsch-Gordon coefficients)

 #if particles...(i.e., !spins)
 const Na = 1 #number of atoms
 const Nf = 1 #number of electrons
 const Z = 1 #atomic charge
 const R = 0. #interatomic distance
 const Delta = 0.5 #grid spacing
 const c = 10. #distance to box edge
 const L = 2c+(Na-1)*R #length
 const kappa = 1/2.385345 #taken from T.E. Baker, et. al. Phys. Rev. B 95, 235141 (2015)
 const A = 1.071295

 Ns = 8 #number of sites
 Nup = 1
 Ndn = 1
#=
if Ns*Delta != L && !spins
 println("grid not commensurate with length!")
 #break
end
=#
#       +-------------------+
#>------|  DMRG parameters  |--------<
#       +-------------------+

 const lanczos = false
 const maxiter = 2 #maximum davidson iterations
 const maxm = 50
 const cutoff = 1E-12
 const nsweeps = 50
 const noise = 1E-12 #helps in davidson convergence for initial vector..not same as ITensor
 const effZero = 1E-10 #sets elements to zero in davidson if below this number
 const errgoal = 1E-10 #target in Davidson

 deltaT = 0.001
 ntime = 10

 const eigsdisplay = 8
 const verbose = false
 const subparallelizations = false

 const regSVD = true

 const tensorops = false #use tensor operations?
 const memVtime = false #true: memory is lower, false: time is faster
 const singlesite = true #false

 const monitor = false
 const saveplot = false

params() = params(Na,Nf,Z,R,Delta,c,L,Ns,Nup,Ndn,s,maxm,nsweeps,cutoff,qstates,spins,fermions,lanczos,maxiter,
               verbose,noise,effZero,errgoal,deltaT,ntime,eigsdisplay,subparallelizations,A,kappa,regSVD,tensorops,
               memVtime,singlesite,monitor,saveplot)

basic = params()

export basic, params

@show basic

end

using parameters
