# Search methods

import search


print "De A a B"
ab = search.GPSProblem('A', 'B', search.romania)


print search.breadth_first_graph_search(ab).path()
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print search.branch_graph_search(ab).path()
print search.branch_graph_search_heuristic(ab).path()

print "   "
#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450


print "De A a C"
ab = search.GPSProblem('A', 'C', search.romania)


print search.breadth_first_graph_search(ab).path()
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print search.branch_graph_search(ab).path()
print search.branch_graph_search_heuristic(ab).path()

print "   "
print "De F a N"
ab = search.GPSProblem('F', 'N', search.romania)


print search.breadth_first_graph_search(ab).path()
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print search.branch_graph_search(ab).path()
print search.branch_graph_search_heuristic(ab).path()

print "   "
print "De R a Z"
ab = search.GPSProblem('R', 'Z', search.romania)


print search.breadth_first_graph_search(ab).path()
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print search.branch_graph_search(ab).path()
print search.branch_graph_search_heuristic(ab).path()

print "   "
print "De E a O"
ab = search.GPSProblem('E', 'O', search.romania)


print search.breadth_first_graph_search(ab).path()
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print search.branch_graph_search(ab).path()
print search.branch_graph_search_heuristic(ab).path()