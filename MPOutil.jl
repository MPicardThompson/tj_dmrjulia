module MPOutil
using parameters
#using functions

#An exact and simple way to apply an MPO to an MPS, described
#in Stoudenmire, White New J. Phys. (2010)
#works best for bond dimension k<~10.  Scales like N*m^3*k^3 (sites, maxm, MPO size)
#Other methods described in the article handle higher order MPOs.
#converts an array to an MPO so that it is instead of being represented in by an array,
#it is represented by a tensor diagrammatically as
#
#       s2
#       |
# a1 -- W -- a2       =    W[a1,s1,s2,a2]
#       |
#       s1
#
#The original Hamiltonian matrix H in the DMRjulia.jl file is of the form
#
# H = [ W_11^s1s2  W_12^s1s2 W_13^s1s2 ....
#       W_21^s1s2  W_22^s1s2 W_23^s1s2 ....
#       W_31^s1s2  W_32^s1s2 W_33^s1s2 ....
#       W_41^s1s2  W_42^s1s2 W_43^s1s2 ....]
#where each W occupies the equivalent of (basic.qstates X basic.qstates) sub-matrices in
#of the H matrix as recorded in each s1s2 pair.  These are each of the operators in H.

#TODO: add in different Hamiltonians on each site.  In other words, make a second function
#or add to this one so that it takes a vector of arrays (or simply, a larger array).
#OR
#could define a single MPO (most general case) and then use functions to fill in variables (in some cases zeros)

function convert2MPO(H::Array{Float64,2},basic::params)
  QS = basic.qstates
  a1size = div(size(H,1),QS) #represented in LEFT link indices
  a2size = div(size(H,2),QS) #represented in RIGHT link indices
  W = zeros(a1size,QS,QS,a2size)
  MPO = [zeros(a1size,QS,QS,a2size) for i=1:basic.Ns]
  MPO[1] = zeros(1,QS,QS,a2size)
  MPO[basic.Ns] = zeros(a1size,QS,QS,1)
  for j = 1:QS, k = 1:QS, l = 1:a1size, m = 1:a2size
      W[l,j,k,m] = H[j + (l-1)*QS, k + (m-1)*QS]
  end
  for i = 1:basic.Ns
  if i == 1
    MPO[i][1,:,:,:] = W[a1size,:,:,:] #put in bottom row on first site
  else
    if i == basic.Ns
      MPO[i][:,:,:,1] = W[:,:,:,1] #put in first column on last site
    else
      MPO[i] = W #to be expanded later to include MPOs that vary on each site.
    end
  end
  end
  return MPO
end

function convert2MPO(H::Function,basic::params)
  QS = basic.qstates
  MPO = [] #initialize everywhere like this?
  for i = 1:basic.Ns
    a1size = div(size(H(i),1),QS) #represented in LEFT link indices
    a2size = div(size(H(i),2),QS) #represented in RIGHT link indices
    W = zeros(a1size,QS,QS,a2size)
    if i == 1
      push!(MPO,zeros(1,QS,QS,a2size))
      #MPO[1] = zeros(QS,QS,1,a2size)
      else
      if i==basic.Ns
        push!(MPO,zeros(a1size,QS,QS,1))
        #MPO[basic.Ns] = zeros(QS,QS,a1size,1)
        else
        push!(MPO,zeros(a1size,QS,QS,a2size))
      end
    end
    for j = 1:QS, k = 1:QS, l = 1:a1size, m = 1:a2size
        W[l,j,k,m] = H(i)[j + (l-1)*QS, k + (m-1)*QS]
    end
    if i == 1
      MPO[i][1,:,:,:] = W[a1size,:,:,:] #put in bottom row on first site
    else
      if i == basic.Ns
        MPO[i][:,:,:,1] = W[:,:,:,1] #put in first column on last site
      else
        MPO[i] = W #to be expanded later to include MPOs that vary on each site.
      end
    end
  end
  return MPO
end

function convert2MPO(H::Function,HamParams::Array{Any,2},basic::params)
  QS = basic.qstates
  MPO = [] #initialize everywhere like this?
  for i = 1:basic.Ns
    a1size = div(size(H(i,HamParams),1),QS) #represented in LEFT link indices
    a2size = div(size(H(i,HamParams),2),QS) #represented in RIGHT link indices
    W = zeros(a1size,QS,QS,a2size)
    if i == 1
      push!(MPO,zeros(1,QS,QS,a2size))
      #MPO[1] = zeros(QS,QS,1,a2size)
      else
      if i==basic.Ns
        push!(MPO,zeros(a1size,QS,QS,1))
        #MPO[basic.Ns] = zeros(QS,QS,a1size,1)
        else
        push!(MPO,zeros(a1size,QS,QS,a2size))
      end
    end
    for j = 1:QS, k = 1:QS, l = 1:a1size, m = 1:a2size
        W[l,j,k,m] = H(i)[j + (l-1)*QS, k + (m-1)*QS]
    end
    if i == 1
      MPO[i][1,:,:,:] = W[a1size,:,:,:] #put in bottom row on first site
    else
      if i == basic.Ns
        MPO[i][:,:,:,1] = W[:,:,:,1] #put in first column on last site
      else
        MPO[i] = W #to be expanded later to include MPOs that vary on each site.
      end
    end
  end
  return MPO
end










function CPLXconvert2MPO(H::Array{Float64,2},basic::params)
  QS = basic.qstates
  a1size = div(size(H,1),QS) #represented in LEFT link indices
  a2size = div(size(H,2),QS) #represented in RIGHT link indices
  W = complex(zeros(a1size,QS,QS,a2size))
  MPO = [complex(zeros(a1size,QS,QS,a2size)) for i=1:basic.Ns]
  MPO[1] = complex(zeros(1,QS,QS,a2size))
  MPO[basic.Ns] = complex(zeros(a1size,QS,QS,1))
  for j = 1:QS, k = 1:QS, l = 1:a1size, m = 1:a2size
      W[l,j,k,m] = H[j + (l-1)*QS, k + (m-1)*QS]
  end
  for i = 1:basic.Ns
  if i == 1
    MPO[i][1,:,:,:] = W[a1size,:,:,:] #put in bottom row on first site
  else
    if i == basic.Ns
      MPO[i][:,:,:,1] = W[:,:,:,1] #put in first column on last site
    else
      MPO[i] = W #to be expanded later to include MPOs that vary on each site.
    end
  end
  end
  return MPO
end

function CPLXconvert2MPO(H::Function,basic::params)
  QS = basic.qstates
  MPO = [] #initialize everywhere like this?
  for i = 1:basic.Ns
    a1size = div(size(H(i),1),QS) #represented in LEFT link indices
    a2size = div(size(H(i),2),QS) #represented in RIGHT link indices
    W = complex(zeros(a1size,QS,QS,a2size))
    if i == 1
      push!(MPO,complex(zeros(1,QS,QS,a2size)))
      #MPO[1] = zeros(QS,QS,1,a2size)
      else
      if i==basic.Ns
        push!(MPO,complex(zeros(QS,QS,a1size,1)))
        #MPO[basic.Ns] = zeros(QS,QS,a1size,1)
        else
        push!(MPO,complex(zeros(QS,QS,a1size,a2size)))
      end
    end
    for j = 1:QS, k = 1:QS, l = 1:a1size, m = 1:a2size
        W[l,j,k,m] = H(i)[j + (l-1)*QS, k + (m-1)*QS]
    end
    if i == 1
      MPO[i][:,:,1,:] = W[a1size,:,:,:] #put in bottom row on first site
    else
      if i == basic.Ns
        MPO[i][:,:,:,1] = W[:,:,:,1] #put in first column on last site
      else
        MPO[i] = W #to be expanded later to include MPOs that vary on each site.
      end
    end
  end
  return MPO
end























#makes first order trotter approximation (in delta T) of the exponential of an MPO (given as a matrix here)
#must be lower left triangular
function expH(H::Function,basic::params)
  QS = convert(Int64,basic.qstates)
  reducedH = H(i)[1+QS:size(H(i),1),1+QS:size(H(i),2)]
  export convert2MPO(reducedH)
end

export convert2MPO,CPLXconvert2MPO,expH

end

using MPOutil
