
import copy
import operator
import numpy as np
import networkx as nx

from .metric import compute_affinity


def solver_3x3(puzzle, graph):
    pieces = copy.deepcopy(puzzle.pieces)
    
    center = get_center(graph)
    center = pieces[center]
    
    solved, (rot_top, rot_right, rot_bottom, rot_left) = construct_cross(pieces, center)
    solved = add_corners(pieces, solved, (rot_top, rot_right, rot_bottom, rot_left))
    
    return solved


def solver_3x4(puzzle, graph):
    raise NotImplementedError


def solver_4x4(puzzle, graph):
    raise NotImplementedError


def get_center(graph):
    jaccard_coeffs = {node: 0 for node in graph.nodes()}
    for (u,v,p) in nx.jaccard_coefficient(graph):
        jaccard_coeffs[u] += p
        jaccard_coeffs[v] += p
    max_jaccard = max(jaccard_coeffs.items(), key=operator.itemgetter(1))[0]
    
    center = None
    center_score = np.inf
    scores = nx.get_node_attributes(graph, 'score')
    for node, degree in graph.degree():
        if (scores[node] < center_score) and (degree == 4):
            center = node
            center_score = scores[node]
    
    center_score = np.inf
    if center == None:
        if jaccard_coeffs[max_jaccard] > 1.8:
            center = max_jaccard
            return center
        else:
            for node, degree in graph.degree():
                if (scores[node] < center_score) and (degree == 3):
                    center = node
                    center_score = scores[node]
    return center


def construct_cross(pieces, center):
    # Assign neighbors
    top, rot_top = pieces[center.top[0]], center.top[1]
    right, rot_right = pieces[center.right[0]], center.right[1]
    left, rot_left = pieces[center.left[0]], center.left[1]
    bottom, rot_bottom = pieces[center.bottom[0]], center.bottom[1]
    
    # Apply correct rotations
    top.rotate(rot_top, update_neighbors=True)
    right.rotate(rot_right, update_neighbors=True)
    bottom.rotate(rot_bottom, update_neighbors=True)
    left.rotate(rot_left, update_neighbors=True)
    
    # Construct central cross
    solved = [None] * len(pieces)
    solved[4] = center
    solved[5] = right
    solved[3] = left
    solved[1] = top
    solved[7] = bottom
    
    return solved, (rot_top,rot_right, rot_bottom, rot_left)


def add_corners(pieces, solved, rotations):
    top, right, bottom, left = solved[1], solved[5], solved[7], solved[3]
    rot_top, rot_right, rot_bottom, rot_left = rotations[0], rotations[1], rotations[2], rotations[3]
    
    if (top.left[0] == left.top[0]) :
        bad_top_left = False
        top_left = pieces[top.left[0]]
        top_left.rotate(rot_top + top.left[1], update_neighbors=True)
        solved[0] = top_left
    else:
        bad_top_left = True

    if (top.right[0] == right.top[0]):
        bad_top_right = False
        top_right = pieces[top.right[0]]
        top_right.rotate(rot_top + top.right[1], update_neighbors=True)
        solved[2] = top_right
    else:
        bad_top_right = True
    
    if (bottom.left[0] == left.bottom[0]):
        bad_bottom_left = False
        bottom_left = pieces[bottom.left[0]]
        bottom_left.rotate(rot_bottom + bottom.left[1], update_neighbors=True)
        solved[6] = bottom_left
    else:
        bad_bottom_left = True

    if (bottom.right[0] == right.bottom[0]):
        bad_bottom_right = False
        bottom_right = pieces[bottom.right[0]]
        bottom_right.rotate(rot_bottom + bottom.right[1], update_neighbors=True)
        solved[8] = bottom_right
    else:
        bad_bottom_right = True
    
    if bad_top_left:
        solved = assign_bad_corner('top-left', pieces, solved, top, right, bottom, left, rot_top, rot_right, rot_bottom, rot_left)
        solved = find_best_rotation('top-left', solved)
    
    if bad_top_right:
        solved = assign_bad_corner('top-right', pieces, solved, top, right, bottom, left, rot_top, rot_right, rot_bottom, rot_left)
        solved = find_best_rotation('top-right', solved)
    
    if bad_bottom_left:
        solved = assign_bad_corner('bottom-left', pieces, solved, top, right, bottom, left, rot_top, rot_right, rot_bottom, rot_left)
        solved = find_best_rotation('bottom-left', solved)
    
    if bad_bottom_right:
        solved = assign_bad_corner('bottom-right', pieces, solved, top, right, bottom, left, rot_top, rot_right, rot_bottom, rot_left)
        solved = find_best_rotation('bottom-right', solved)
    
    return solved


def assign_bad_corner(position, pieces, solved, top, right, bottom, left, rot_top, rot_right, rot_bottom, rot_left):
    if position == 'top-left':
        candidate_1 = pieces[top.left[0]]
        candidate_2 = pieces[left.top[0]]
        
        affinity_1 = compute_affinity(top, candidate_1, 'left', rotation=rot_top+top.left[1])
        affinity_1 += compute_affinity(left, candidate_1, 'top', rotation=rot_top+top.left[1])
        
        affinity_2 = compute_affinity(top, candidate_2, 'left', rotation=rot_left + top.left[1])
        affinity_2 += compute_affinity(left, candidate_2, 'top', rotation=rot_left + top.left[1])
        
        solved[0] = choose_best_candidate(candidate_1, candidate_2, solved, affinity_1, affinity_2)
        return solved
    
    elif position == 'top-right':
        candidate_1 = pieces[top.right[0]]
        candidate_2 = pieces[right.top[0]]
        
        affinity_1 = compute_affinity(top, candidate_1, 'right', rotation=rot_top+top.right[1])
        affinity_1 += compute_affinity(right, candidate_1, 'top', rotation=rot_top+top.right[1])
        
        affinity_2 = compute_affinity(top, candidate_2, 'right', rotation=rot_right + top.right[1])
        affinity_2 += compute_affinity(right, candidate_2, 'top', rotation=rot_right + top.right[1])
        
        solved[2] = choose_best_candidate(candidate_1, candidate_2, solved, affinity_1, affinity_2)
        return solved
    
    elif position == 'bottom-left':
        candidate_1 = pieces[bottom.left[0]]
        candidate_2 = pieces[left.bottom[0]]
        
        affinity_1 = compute_affinity(left, candidate_1, 'bottom', rotation=rot_left+left.bottom[1])
        affinity_1 += compute_affinity(bottom, candidate_1, 'left', rotation=rot_left+left.bottom[1])
        
        affinity_2 = compute_affinity(left, candidate_2, 'bottom', rotation=rot_bottom + left.bottom[1])
        affinity_2 += compute_affinity(bottom, candidate_2, 'left', rotation=rot_bottom + left.bottom[1])
        
        solved[6] = choose_best_candidate(candidate_1, candidate_2, solved, affinity_1, affinity_2)
        return solved
    
    elif position == 'bottom-right':
        candidate_1 = pieces[bottom.right[0]]
        candidate_2 = pieces[right.bottom[0]]
        
        affinity_1 = compute_affinity(right, candidate_1, 'bottom', rotation=rot_bottom+bottom.right[1])
        affinity_1 += compute_affinity(bottom, candidate_1, 'right', rotation=rot_bottom+bottom.right[1])
        
        affinity_2 = compute_affinity(right, candidate_2, 'bottom', rotation=rot_right + right.bottom[1])
        affinity_2 += compute_affinity(bottom, candidate_2, 'right', rotation=rot_right + right.bottom[1])
        
        solved[8] = choose_best_candidate(candidate_1, candidate_2, solved, affinity_1, affinity_2)
        return solved


def choose_best_candidate(candidate_1, candidate_2, solved, affinity_1, affinity_2):
    if candidate_1 in solved:
        return candidate_2
    elif candidate_2 in solved:
        return candidate_1
    else:
        return candidate_1 if (affinity_1 < affinity_2) else candidate_2


def find_best_rotation(position, solved):
    best_rotation = 0
    best_score = np.inf
    if position == 'top-left':
        for rotation in [0, 1, 2, 3]:
            left_affinity = compute_affinity(solved[1], solved[0], 'left', rotation)
            top_affinity = compute_affinity(solved[3], solved[0], 'top', rotation)
            if left_affinity + top_affinity < best_score:
                best_score = left_affinity + top_affinity
                best_rotation = rotation
        solved[0].rotate(best_rotation)
        return solved
    
    elif position == 'top-right':
        for rotation in [0, 1, 2, 3]:
            right_affinity = compute_affinity(solved[1], solved[2], 'right', rotation)
            top_affinity = compute_affinity(solved[5], solved[2], 'top', rotation)
            if right_affinity + top_affinity < best_score:
                best_score = right_affinity + top_affinity
                best_rotation = rotation
        solved[2].rotate(best_rotation)
        return solved
    
    elif position == 'bottom-left':
        for rotation in [0, 1, 2, 3]:
            left_affinity = compute_affinity(solved[7], solved[6], 'left', rotation)
            bottom_affinity = compute_affinity(solved[3], solved[6], 'bottom', rotation)
            if left_affinity + bottom_affinity < best_score:
                best_score = left_affinity + bottom_affinity
                best_rotation = rotation
        solved[6].rotate(best_rotation)
        return solved
    
    elif position == 'bottom-right':
        for rotation in [0, 1, 2, 3]:
            right_affinity = compute_affinity(solved[7], solved[8], 'right', rotation)
            bottom_affinity = compute_affinity(solved[5], solved[8], 'bottom', rotation)
            if right_affinity + bottom_affinity < best_score:
                best_score = right_affinity + bottom_affinity
                best_rotation = rotation
        solved[8].rotate(best_rotation)
        return solved