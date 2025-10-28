import os
import json
from collections import Counter

def parse_kitti_label_file(label_path):
    """Parse KITTI label file and return difficulty info"""
    difficulties = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != 'Car':
                continue
            truncated = float(parts[1])
            occluded = int(parts[2])
            
            # Determine difficulty based on KITTI criteria
            if occluded == 0 and truncated < 0.15:
                difficulty = "easy"
            elif occluded <= 1 and truncated < 0.30:
                difficulty = "moderate"
            elif occluded <= 2 and truncated < 0.50:
                difficulty = "hard"
            else:
                difficulty = "ignored"
            
            difficulties.append(difficulty)
    return difficulties

def analyze_difficulty_distribution():
    """Analyze the distribution of difficulty levels in the dataset"""
    label_dir = "../dataset/label_2"
    
    all_difficulties = []
    total_cars = 0
    
    # Process all label files
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            label_path = os.path.join(label_dir, filename)
            difficulties = parse_kitti_label_file(label_path)
            all_difficulties.extend(difficulties)
            total_cars += len(difficulties)
    
    # Count difficulties
    difficulty_counts = Counter(all_difficulties)
    
    print("=== KITTI Difficulty Distribution Analysis ===")
    print(f"Total Car objects: {total_cars}")
    print(f"Easy: {difficulty_counts['easy']} ({difficulty_counts['easy']/total_cars*100:.1f}%)")
    print(f"Moderate: {difficulty_counts['moderate']} ({difficulty_counts['moderate']/total_cars*100:.1f}%)")
    print(f"Hard: {difficulty_counts['hard']} ({difficulty_counts['hard']/total_cars*100:.1f}%)")
    print(f"Ignored: {difficulty_counts['ignored']} ({difficulty_counts['ignored']/total_cars*100:.1f}%)")
    
    # Check a few sample files
    print("\n=== Sample Analysis ===")
    sample_files = ["000000.txt", "000001.txt", "000002.txt", "000003.txt", "000004.txt"]
    for filename in sample_files:
        label_path = os.path.join(label_dir, filename)
        if os.path.exists(label_path):
            difficulties = parse_kitti_label_file(label_path)
            print(f"{filename}: {difficulties}")

if __name__ == "__main__":
    analyze_difficulty_distribution()
