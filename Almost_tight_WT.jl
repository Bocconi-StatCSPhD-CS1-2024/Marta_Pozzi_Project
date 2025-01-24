#I'm absolutely sure that getChildren, getLeaf, findPath, IsOnPath


#this function should be changed in order to handle more general cases, but for now we construct the trees
#so that is works
function getChildren(A, u)
    # Get the children of node u in the tree represented by adjacency matrix A
    children = findall(x -> x == 1, A[u, :])  # Find non-zero elements in row u
    children = [x for x in children if x != u && x > u]  # Remove the node itself and nodes above u
    #children = [x for x in children if x != u] 
    return children
end

function getLeaf(A)
    leaf = []
    for i in 1:size(A, 2)
        if sum(A[i, :]) == 1
            push!(leaf, i)
        end
    end
    return leaf
end

function findPath(A, u, v)
    # Function to find the shortest path from node u to node v using BFS
    n = size(A, 1)  # Number of nodes
    visited = falses(n)  # Array to mark visited nodes
    parent = fill(-1, n)  # Array to store the parent of each node in the path
    queue = [u]  # Queue for BFS, start BFS from node u
    visited[u] = true  # Mark the start node as visited
    
    while !isempty(queue)
        current = popfirst!(queue)  # Get the current node
        
        if current == v
            # Found the target node, reconstruct the path
            path = []
            while current != -1
                push!(path, current)
                current = parent[current]
            end
            return reverse(path)  # Return the path in correct order
        end
        
        # Add all unvisited neighbors of the current node to the queue
        neighbors = findall(x -> x == 1, A[current, :])
        for neighbor in neighbors
            if !visited[neighbor]
                visited[neighbor] = true
                parent[neighbor] = current
                push!(queue, neighbor)
            end
        end
    end
    
    return []  # Return empty path if no path found
end

function isOnPath(A, u, v, w)
    # Function to check if node w is on the path from node u to node v
    path = findPath(A, u, v)  # Get the path from u to v
    if isempty(path)
        return false  # No path found
    else
        return w in path  # Check if w is in the path
    end
end
  

    function getWeight(A, v, F)
        # Compute the weight vector w(v) for node v in the witness tree
        # A is the adjacency matrix, F is the set of final Steiner nodes
        
        # Initialize the weight vector for node v
        w_v = 0
        
        # Traverse all pairs of nodes in F to count paths through node v
        for i in 1:length(F)
            for j in i+1:length(F)
                # Check if v belongs to the path between F[i] and F[j] in T[S*]
                if isOnPath(A, F[i], F[j], v)
                    w_v += 1  # Increment the weight if v belongs to the path
                end
            end
        end
        
        # Add the indicator for final node
        if v in F
            w_v += 1  # Final nodes get an additional weight
        end
        
        return w_v
    end

    function computeCost(A, u, F)#In particulr I'm just computing C1
        # Compute the cost C_{u_j}^1 for a given child and its subtree W_{u_j}
        cost = 0
        for i in 1:size(A, 2)
            for j in 1:length(F)
                if isOnPath(A, u, F[j], i)
                    w_v = getWeight(A, i, F)
                    cost += 1 / (w_v + 1 + 1)
                end
            end
        end
        return cost
    end

    function costVector(A, children)
        C = zeros(1, length(children))
        for i = 1:length(children)
            w_v = computeCost(A, children[i], F)
            C[i] = w_v
        end
        return C
    end


    function selectMarkedChild(children, F, C, phi, delta, H2)
  
        if all(x -> x ∉ F, children)
            # Select the child with the minimum cost
            minvalue, indexMarkedChild = findmin(C)
            #markedChild = markedChild[1]  # Convert CartesianIndex to integer index
            if typeof(indexMarkedChild) !=Float64
                #I just pick one of the two marked children, by semplicity the first one
                indexMarkedChild = indexMarkedChild[1]
            end
            markedChild = children[indexMarkedChild]
            return markedChild
        end
        # Select the marked child based on cost conditions
        if all(C .>= phi - delta - H2)
            minvalue, indexMarkedChild = findmin(C)  # Select the child with the minimum cost
            #markedChild = markedChild[1]  # Convert CartesianIndex to integer index
            if typeof(indexMarkedChild) !=Float64
                indexMarkedChild = indexMarkedChild[1]
            end
            markedChild = children[indexMarkedChild]
            return markedChild
        else
            C_use = []
            for j in 1:length(children)
                if C[j] < phi - delta - H2
                    C_use = [C_use, C[j]]
                end
                minvalue, indexMarkedChild = findmin(C_use)
                if typeof(indexMarkedChild) !=Float64
                    indexMarkedChild = indexMarkedChild[1]
                end
                markedChild = children[indexMarkedChild]
                return markedChild # Select the child that satisfies the condition
                
            end
        end
    end


    function getLeafDescendant(A, u)
        # Get the leaf descendant of node u in the tree represented by adjacency matrix A
        children = getChildren(A, u)
        if isempty(children)
            return u
        else

            return getLeafDescendant(A, children[1])  # Follow the marked child path
        end
    end

    function mergeAdjacencyMatrices(W_children, markedChild, A, children)
        # Function to merge the adjacency matrices of subtrees
        #here we already have the adjacency matrices computed 
        W_u = zeros(size(W_children[1],1), size(W_children[1],2))
        markedChildIndex = 0
        for i = 1:length(children)
            if children[i]==markedChild
                markedChildIndex = i
            end
        end

        # Now merge the adjacency matrices of the other subtrees
        for i in 1:length(W_children)
            if i != markedChildIndex
                # Merge the adjacency matrices
                for u in 1:size(W_u, 1)
                    for v in 1:size(W_u, 2)
                        if v != u # non voglio edges su singoli vertici
                            if W_children[i][u, v] == 1
                                W_u[u, v] = 1  # Add the connection if it exists in the other subtree
                            end
                        end
                    end
                end
            end
        end

        #here we have the actual merging of the matrices
        # Connect the leaf descendant of the marked child with all leaf descendants of other children
        #leaf_descendant_marked = getLeafDescendant(A, children[markedChild])

        for j in 1:length(W_children)
            leaf_descendant_marked = getLeafDescendant(A, children[markedChildIndex])
        #I have to loop on the other trees 
        #I still have to think about the logic here
            for i in 1:length(children) 
                if i != markedChild
                    leaf_descendant_other = getLeafDescendant(A, children[i])
                    
                    if leaf_descendant_other != leaf_descendant_marked
                        W_u[leaf_descendant_marked, leaf_descendant_other] = 1
                        W_u[leaf_descendant_other, leaf_descendant_marked] = 1
                    end
                    
                end
            end
        end
    

        return W_u
    end

function computeWitnessTree(u, A, F, phi, delta, H2, root)

    # Get the children of u
    children = getChildren(A, u)

    #Case in which the tree is just an edge
    #between the root and another node
    W_children = [zeros(Int, size(A, 1), size(A, 2)) for _ in 1:size(A,1)]

    if size(children,1)==1 && children[1]== root 
        #W_u = zeros(Int, size(A, 1), size(A, 2))
        W_children[children[1]][root, u] =1
        W_children[children[1]][u,root] =1
    end
    # Base case: if u is a leaf or T is a single edge
    if length(children) == 0
        W_u = zeros(Int, size(A, 1), size(A, 2))  # Return an adjacency matrix for node u
        #W_children[u] = zeros(Int, size(A, 1), size(A, 2))
       
        return W_u
    end

    # Step 1: Compute the witness tree for each child

    #W_children = [zeros(Int, size(A, 1), size(A, 2)) for _ in 1:size(A,1)]
       

    for i in 1:length(children)

            W_children[children[i]] = computeWitnessTree(children[i], A, F, phi, delta, H2, root)
    end

    # Step 2: Compute the cost function C_u for each child
    C = costVector(A, children)

    # Step 3: Select the marked child based on cost conditions
    markedChild = selectMarkedChild(children, F, C, phi, delta, H2)

    # Step 4: Construct the witness tree by merging adjacency matrices
    W_u = mergeAdjacencyMatrices(W_children, markedChild, A, children)
   
 
# Convert W_u to integer type

W_u = Int.(W_u) #Idk why but at some point the elemets became floeats and it was confusing    
return W_u
end


#oss: in our example we will take the final nodes as the leafs of the tree, assuming
 #that the oreprocessing was already done 
 #  Example usage


phi = 1.86
delta = 97/420
H2 = 1.5

#A = [0 1 1 0 0;
#1 0 0 1 1;
#1 0 0 0 0;
#0 1 0 0 0;
#0 1 0 0 0]
#F = getLeaf(A)
#W_u = computeWitnessTree(u, A, F, phi, delta, H2, visited)
# Example usage with a more complex adjacency matrix
u = 1



# Number of nodes in the tree
n = 20

# Initialize the adjacency matrix
A = zeros(Int, n, n)

# Create a connected tree (manually adding edges)
# In a tree with n nodes, there are exactly n-1 edges
edges = [1 2; 1 3; 2 4; 2 5; 3 6; 3 7; 4 8; 5 9; 6 10]
#edges = [1 2; 2 3; 2 4; 3 5; 3 6; 4 7; 4 8; 5 9; 6 10; 7 11]

#small thong to correct: I need to have leafes with higher enumeration than the parents
#I'd li ke to make it more general
edges = [1 2; 1 3; 2 4; 2 5; 3 6; 3 7;4 8; 5 9; 6 10;1 20; 11 19; 14 18; 15 14; 1 14; 11 12; 12 17; 12 13; 1 11; 15 16]
#edges = [1 2;2 3; 2 4; 3 5; 3 6; 4 7; 4 8; 5 9; 6 10; 7 11;2 21; 12 20; 15 19; 16 19; 2 16; 12 13; 13 18; 13 14; 2 12; 16 17]
# Add the edges to the adjacency matrix
for i in 1:size(edges, 1)
    A[edges[i, 1], edges[i, 2]] = 1
    A[edges[i, 2], edges[i, 1]] = 1
end
F = getLeaf(A)

root = F[1]

#F = getLeaf(A2)

#W_u = computeWitnessTree(u, A, F, phi, delta, H2, root)

# Print the final witness tree adjacency matrix
println("Final witness tree adjacency matrix:")
println(W_u)

u = 2

A = [0 1 1 0 0;
1 0 0 1 1;
1 0 0 0 0;
0 1 0 0 0;
0 1 0 0 0]
F = getLeaf(A)
root = F[1]
W_u = computeWitnessTree(u, A, F, phi, delta, H2, visited)



