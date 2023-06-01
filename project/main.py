import os
import copy
import argparse
import numpy as np

from PIL import Image
from modules.save_evaluation_files import export_solutions
from modules.segmentation import segment
from modules.utils import get_rectangles, get_piece
from modules.features_extraction import extract_features
from modules.clustering import cluster_features, clusters_check
from modules.puzzle_solving.puzzle import Puzzle


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tiling Puzzle Project full pipeline.'
    )
    parser.add_argument('data_dir', help='path to the images folder')
    parser.add_argument('out_dir', help='path to save the solutions')
    return parser.parse_args()


def main(data_dir, out_dir):
    
    files = os.listdir(data_dir)
    for file in files:
        img_idx = int(file.split('.')[0][-2:])
        
        # Load image
        print('\nLoading image: {}'.format(file))
        image = Image.open(os.path.join(data_dir, file)).convert('RGB')
        image = np.array(image)
        
        # Perform segmentation
        print('Performing segmentation ...')
        segmented = segment(image)
        rectangles, MASK = get_rectangles(image, segmented) 
        
        # Get the pieces
        pieces = []
        for rect in rectangles:
            pieces.append(get_piece(image, rect))
        
        # Perform feature extraction
        print('Performing feature extraction ...')
        features = []
        for piece in pieces:
            feature = extract_features(piece.data, use_gabor=True, use_lbp=True)
            features.append(feature)
        features = np.array(features) 
        FEATURES_MAP = copy.deepcopy(features)
        
        # Perform clustering
        print('Performing clustering ...')
        labels = cluster_features(features, max_clusters=4, use_dbscan=True, dbscan_eps=0.09, dbscan_min_samples=2)
        labels = clusters_check(features, labels)
        
        # Label and group the pieces
        for i, label in enumerate(labels):
            pieces[i].label = label
        
        clusters = [[] for _ in set(labels)]
        CLUSTERS_DATA = [[] for _ in set(labels)]
        for piece in pieces:
            clusters[piece.label].append(piece)
            CLUSTERS_DATA[piece.label].append(piece.data)
        
        # Solve the puzzles
        SOLVED_PUZZLES = []
        print('Solving the puzzles ...')
        for pieces in clusters[:-1]:
            try:
                puzzle = Puzzle(pieces)
                puzzle.solve(crop=4)
                puzzle.assemble()
                SOLVED_PUZZLES.append(puzzle.data)
            except:
                continue
        
        # Saving solution
        solution = [MASK, FEATURES_MAP, CLUSTERS_DATA, SOLVED_PUZZLES]
        export_solutions(img_idx,  solution, path = out_dir, group_id = "15")
        print('Solutions saved for image: {}'.format(file))


if __name__ == '__main__':
    args = parse_args()
    main(args.data_dir, args.out_dir)

