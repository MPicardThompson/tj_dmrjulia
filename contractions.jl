module contractions

#=
#don't pass whole array, just pass dimension information?
#no time or allocation difference seen between the two...
function pos2index(AA::Array,loc::Vector)
  return pos2index(ndims(AA),size(AA),loc)
end

#takes a vector and returns index for array as assigned by eachindex
#rules:  starts at left most index and updates right (see "counter" operation below)
#function pos2index(ndimsAA::Int64,sizeAA::Tuple,loc::Vector)
function pos2index(sizeAA::Tuple,loc::Vector)
  index = loc[1]
  for p = 2:size(loc,1)
    num = 1
    for q = 1:p-1
      num *= sizeAA[q]
    end
    index += (loc[p]-1) * num
  end
  return index
end
=#

#rearranges the sizes of a vector, ex: (1,2,3) to (3,1,2) by rearrangevec(AA,[3,1,2])
function rearrangevec(origvec::Array{Int64,1},order::Array{Int64,1})
  size(origvec) != size(order) ? error("size of output vector and reordering are not equal in contractions") : 0
  returnvec = Int64[]
  for i = 1:size(origvec)[1]
    push!(returnvec,origvec[order[i]])
  end
  return returnvec
end

###################

#shorthand:  calls with only integer inputs for single index contraction
function contract(order::Array{Int64,1},A::Array,iA::Int64,B::Array,iB::Int64)
  vA = [iA] #converts integer to vector
  vB = [iB]
  return contract(order,A,vA,B,vB)
end

#if commas or semicolons are used, then you get a dimension 1 array...just spaces gives dimension 2
function contract(order::Array{Int64,2},A::Array,iA::Array{Int64,2},B::Array,iB::Array{Int64,2})
  return contract(order[:],A,iA[:],B,iB[:])
end

#if commas or semicolons are used, then you get a dimension 1 array...just spaces gives dimension 2
function contract(order::Array{Int64,1},A::Array,iA::Array{Int64,1},B::Array,iB::Array{Int64,1})
  return permutedims(contract(A,iA,B,iB),order)
end

#If we want the standard permutation
function contract(A::Array,iA::Int64,B::Array,iB::Int64)
  vA = [iA] #converts integer to vector
  vB = [iB]
  return contract(A,vA,B,vB)
end

function contract(A::Array,iA::Array{Int64,2},B::Array,iB::Array{Int64,2})
  return contract(A,iA[:],B,iB[:])
end

#contract to scalar
function contract(A::Array,B::Array)
  size(A) == size(B) ? 0. : error("A and B do not match in contract")
  numdims = ndims(A)
  indsvec = [i for i = 1:numdims]
  return contract(A,indsvec,B,indsvec)
end

#contracts A and B along indices in iA and iB, returns AA
#returns indices of AA in order they are given in A U B (without those contracted over)
#IMPORTANT NOTE: for this reason, this will not commute for A and B!
#added: can now permute dimensions with the first entry as a vector
function contract(A::Array,iA::Array{Int64,1},B::Array,iB::Array{Int64,1})
  #if basic.debug
  #=
  size(iA) == size(iB) ? 0. : error("Unequal number of contracted indices between tensors")
  for n = 1:size(iA)[1]
    size(A,iA[n]) == size(B,iB[n]) ? 0. : error("Unequal ranks in contraction")
  end
  =#
  #end
  vecA = Int64[]
  vecB = Int64[]
  AAsizes = Int64[]
  remsizeA = 1
  remsizeB = 1
  consize = 1

  tempA = 0
  tempB = 0


 for j = 1:2
   currInds = j == 1 ? iA : iB
   currTens = j == 1 ? A : B
   currVec = j == 1 ? vecA : vecB
#   remSize = j == 1 ? remsizeA : remsizeB
#   temp = j == 1 ? tempA : tempB
   if j == 2
     for i = 1:size(currInds,1)
       push!(currVec,currInds[i])
     end
   end
   for p = 1:ndims(currTens)
      docontract = false
      for q = 1:size(currInds,1)
        if p == currInds[q]
          docontract = true #contract this index, move onto next part of loop
          break
        end
      end
      if docontract == false
        push!(currVec,p)
        push!(AAsizes,size(currTens,p))
        j == 1 ? remsizeA *= size(currTens,p) : remsizeB *= size(currTens,p)
      end
   end
   if j == 1
     for i = 1:size(currInds,1)
       push!(currVec,currInds[i])
       consize *= size(currTens,currInds[i])
     end
   end
   if (currVec != [i for i = 1:ndims(currTens)])
     temp = permutedims(currTens,currVec)
     else
     temp = currTens
   end
   j == 1 ? tempA = temp : tempB = temp
 end

  if !(remsizeA == remsizeB == 1)
    tupleAA = (AAsizes[1]...) #initializes tuple for the new tensor to be returned
    for y = 2:size(AAsizes,1)
      tupleAA = (tupleAA...,AAsizes[y]...)
    end
  end

  C = reshape(tempA,remsizeA,consize)
  D = reshape(tempB,consize,remsizeB)
  if remsizeA == remsizeB == 1
    return   (C*D)[1]
  else
    return reshape(C*D,tupleAA)
  end
end

export contract

end

using contractions













#
# The following code is better for reading, but the above code was used to make qcontract
# (it only saves about 20 lines...)


#=
 for p = 1:ndims(A)
    docontract = false
    for q = 1:size(iA,1)
      if p == iA[q]
        docontract = true #contract this index, move onto next part of loop
        break
      end
    end
    if docontract == false
      push!(vecA,p)
      push!(AAsizes,size(A,p))
      remsizeA *= size(A,p)
    end
  end
  for i = 1:size(iA,1)
    push!(vecA,iA[i])
    consize *= size(A,iA[i])
  end

  for i = 1:size(iB,1)
    push!(vecB,iB[i])
  end
  for p = 1:ndims(B)
    docontract = false
    for q = 1:size(iB,1)
      if p == iB[q]
        docontract = true #contract this index, move onto next part of loop
        break
      end
    end
    if docontract == false
      push!(vecB,p)
      push!(AAsizes,size(B,p))
      remsizeB *= size(B,p)
    end
  end

  if !(remsizeA == remsizeB == 1)
    tupleAA = (AAsizes[1]...) #initializes tuple for the new tensor to be returned
    for y = 2:size(AAsizes,1)
      tupleAA = (tupleAA...,AAsizes[y]...)
    end
  end

  if (vecA != [i for i = 1:ndims(A)])
    tempA = permutedims(A,vecA)
  else
    tempA = A
  end

  if (vecB != [i for i = 1:ndims(B)])
    tempB = permutedims(B,vecB)
  else
    tempB = B
  end

  println("vecA ",vecA)
println("vecB ",vecB)
println("remsizeA: ",remsizeA,",remsizeB: ",remsizeB)
println("consize ",consize)
println("AAsizes ",AAsizes)
=#
