#!/usr/bin/env python3
"""
Parser for Android APK opening book data (open_tree*.txt files).

These files contain opening book moves in format:
ID MOVE_NUMBER. MOVE (wins-losses, total: N)

Where MOVE is a 2-digit number representing the move.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple


def parse_move(move_str: str) -> int:
    """
    Parse move string like '78', '57x' to action (0-8).
    The first digit seems to be from position, second to position.
    For Togyzkumalak, we interpret as action = first_digit - 1 (0-8).
    """
    # Remove 'x' suffix if present (indicates capture)
    move_clean = move_str.replace('x', '').strip()
    if len(move_clean) >= 1:
        # First digit is the pit number (1-9), convert to action (0-8)
        first_digit = int(move_clean[0])
        return first_digit - 1 if 1 <= first_digit <= 9 else 0
    return 0


def parse_open_tree_file(filepath: str) -> List[Dict]:
    """
    Parse a single open_tree file and extract moves with win/loss stats.
    
    Returns list of dicts with: move_number, action, wins, losses, total
    """
    moves = []
    
    # Try different encodings
    encodings = ['utf-16', 'utf-16-le', 'utf-8', 'cp1251', 'latin-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if content is None:
        print(f"[ERROR] Could not decode file: {filepath}")
        return moves
    
    for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Pattern: ID MOVE_NUM. MOVE (wins-losses, total: N)
            # Example: 1 2. 78 (18-18, всего: 36)
            match = re.match(r'(\d+)\s+(\d+)\.?\s+(\d+x?)\s+\((\d+)-(\d+),\s*всего:\s*(\d+)\)', line)
            if match:
                move_id = int(match.group(1))
                move_num = int(match.group(2))
                move_str = match.group(3)
                wins = int(match.group(4))
                losses = int(match.group(5))
                total = int(match.group(6))
                
                action = parse_move(move_str)
                
                # Calculate reward based on win rate
                if total > 0:
                    win_rate = wins / total
                    # Map win_rate to reward: 0.5 -> 0, 1.0 -> 1, 0.0 -> -1
                    reward = (win_rate - 0.5) * 2
                else:
                    reward = 0.0
                
                moves.append({
                    'move_id': move_id,
                    'move_number': move_num,
                    'action': action,
                    'wins': wins,
                    'losses': losses,
                    'total': total,
                    'reward': reward,
                    'original_move': move_str
                })
    
    return moves


def generate_training_examples(moves: List[Dict], starting_player: int = 0) -> List[Dict]:
    """
    Convert parsed moves to training examples.
    
    For AlphaZero, we need: board state, action, reward
    Since we don't have full board state, we create normalized examples
    based on move number (early game = initial state approximation).
    """
    examples = []
    
    # Initial board state (normalized, 20 elements)
    # 18 pits with 9 kumalaks each, 2 kazans with 0
    initial_pits = [9/81] * 18  # Normalized
    initial_kazans = [0.0, 0.0]
    
    for move in moves:
        # Create approximate state based on move number
        # This is an approximation - real state would be better
        state = initial_pits.copy() + initial_kazans.copy()
        
        # Adjust state slightly based on move number (simulate game progress)
        move_num = move['move_number']
        if move_num > 1:
            # Reduce kumalaks proportionally to move number
            reduction = min(0.5, move_num * 0.02)
            state = [max(0, s - reduction * 0.1) for s in state[:18]] + state[18:]
        
        # Create training example in compact format
        example = {
            's': state,
            'a': move['action'],
            'r': move['reward'],
            'd': 0,  # not done
            'p': (move['move_number'] + starting_player) % 2  # alternating players
        }
        
        # Weight by total games (more games = more reliable)
        weight = min(1.0, move['total'] / 50)  # Normalize to 50 games
        
        # Add multiple copies based on weight for important moves
        copies = max(1, int(weight * 3))
        for _ in range(copies):
            examples.append(example)
    
    return examples


def parse_all_android_apk_data(apk_dirs: List[str], output_file: str) -> Dict:
    """
    Parse all open_tree files from Android APK directories.
    """
    all_examples = []
    stats = {
        'files_parsed': 0,
        'total_moves': 0,
        'total_examples': 0
    }
    
    for apk_dir in apk_dirs:
        internal_dir = os.path.join(apk_dir, 'assets', 'internal')
        if not os.path.exists(internal_dir):
            print(f"[WARNING] Directory not found: {internal_dir}")
            continue
        
        # Find all open_tree*.txt files
        for filename in os.listdir(internal_dir):
            if filename.startswith('open_tree') and filename.endswith('.txt'):
                filepath = os.path.join(internal_dir, filename)
                print(f"Parsing: {filepath}")
                
                moves = parse_open_tree_file(filepath)
                stats['total_moves'] += len(moves)
                stats['files_parsed'] += 1
                
                # Determine starting player from filename (open_tree2 = player 0, etc.)
                try:
                    tree_num = int(filename.replace('open_tree', '').replace('.txt', ''))
                    starting_player = tree_num % 2
                except:
                    starting_player = 0
                
                examples = generate_training_examples(moves, starting_player)
                all_examples.extend(examples)
    
    stats['total_examples'] = len(all_examples)
    
    # Write to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n=== Android APK Parsing Complete ===")
    print(f"Files parsed: {stats['files_parsed']}")
    print(f"Total moves: {stats['total_moves']}")
    print(f"Training examples generated: {stats['total_examples']}")
    print(f"Output saved to: {output_file}")
    
    return stats


def merge_with_existing_data(new_file: str, existing_file: str, output_file: str) -> int:
    """
    Merge new Android APK data with existing training data.
    """
    all_lines = []
    
    # Read existing data
    if os.path.exists(existing_file):
        with open(existing_file, 'r') as f:
            all_lines = f.readlines()
        print(f"Loaded {len(all_lines)} existing examples")
    
    # Read new data
    with open(new_file, 'r') as f:
        new_lines = f.readlines()
    print(f"Adding {len(new_lines)} new examples from Android APK")
    
    all_lines.extend(new_lines)
    
    # Write merged data
    with open(output_file, 'w') as f:
        f.writelines(all_lines)
    
    print(f"Total examples: {len(all_lines)}")
    return len(all_lines)


if __name__ == '__main__':
    # Default paths
    project_root = Path(__file__).parent.parent.parent.parent
    
    apk_dirs = [
        str(project_root / 'Android-APK'),
        str(project_root / 'Android-APK-Openings')
    ]
    
    output_dir = Path(__file__).parent.parent / 'training_data'
    apk_output = str(output_dir / 'android_apk_openings.jsonl')
    merged_output = str(output_dir / 'transitions_compact.jsonl')
    existing_file = str(output_dir / 'transitions_compact.jsonl')
    
    # Parse APK data
    stats = parse_all_android_apk_data(apk_dirs, apk_output)
    
    # Merge with existing data
    if os.path.exists(existing_file):
        total = merge_with_existing_data(apk_output, existing_file, merged_output)
        print(f"\nMerged data saved to: {merged_output}")
    else:
        # Just copy as the main file
        import shutil
        shutil.copy(apk_output, merged_output)
        print(f"\nData saved to: {merged_output}")
