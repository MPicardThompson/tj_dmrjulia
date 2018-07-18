module Operators
#generates spin operators from Pauli matrices. other operators are generated from these
using parameters

#=
#defined as a function for both spin and Hubbard model ease
function makeops(oz::Array{SparseMatrixCSC{Any,Int64},2},
                 op::Array{SparseMatrixCSC{Any,Int64},2},
                 om::Array{SparseMatrixCSC{Any,Int64},2})
 ox = (op+om)#x and y matrices
 oy = (op-om)/im
 o = [ox;oy;oz] #pauli vector
 o2 = oz*oz + oy*oy + ox*ox #sigma squared
 return o,o2,oz,ox,oy,op,om
end
=#

function spinOps(basic::params)

QS = convert(Int64,2basic.s+1)

 oz = spzeros(QS,QS) # z operator
 op = spzeros(QS,QS) # raising operator
 om = spzeros(QS,QS) # lowering operator

 q=1 #counts quantum states
 for m = basic.s:-1:-basic.s #counts from m to -m (all states)
   oz[q,q] = m
   if m+1 <= basic.s
     op[q-1,q] = sqrt(basic.s*(basic.s+1)-m*(m+1))
   end
   if m-1 >= -basic.s
     om[q+1,q] = sqrt(basic.s*(basic.s+1)-m*(m-1))
   end
   q += 1
 end
 ox = (op+om)#x and y matrices
 oy = (op-om)/im
 o = [ox;oy;4oz] #pauli vector
 o2 = 4oz*oz + oy*oy + ox*ox #sigma squared
 return o,o2,oz,ox,oy,op,om
 #return makeops(oz,op,om)
end

O = spzeros(convert(Int64,2basic.s+1),convert(Int64,2basic.s+1))
Id = sparse(diagm([1 for i = 1:convert(Int64,2basic.s+1)]))

function fermionOps(basic::params)

QS = convert(Int64,basic.qstates)
    Cup = spzeros(4, 4)
    Cup[1,2] = 1
    Cup[3,4] = 1
    Cdn = spzeros(4, 4)
    Cdn[1,3] = 1
    Cdn[2,4] = -1
bigO = spzeros(QS,QS)
bigId = sparse(diagm([1 for i = 1:basic.qstates]))
Nup = Cup' * Cup
Ndn = Cdn' * Cdn
Ndens = Nup + Ndn

F = spzeros(4, 4)
F[1,1] = 1
F[2,2] = -1
F[3,3] = -1
F[4,4] = 1


 return Cup,Cdn,bigO,bigId,Nup,Ndn,Ndens,F
 #return makeops(oz,op,om)
end

export  spinOps,O,Id,fermionOps

end

using Operators





#=
function levicivita(state::Vector{Bool,basic.qstates},basic::params)
  L = length(state)
  count = 0
  for i = 1:L
    for j = 1:L
      count += state[j]!=0 ? j < i : 0 #j>i for other vacuum convention, however the entries are defined
    end
  end
  return mod(perm,basic.qstates)
end

function phase(state::Vector{Bool,basic.qstates},basic::params)
  #generates the correct phase for a given particle with spin "basic.s"
  return exp(im * sum(state) * 2pi * basic.s) * levicivita(state)
end
=#
