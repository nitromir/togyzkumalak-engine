#!/usr/bin/env python3
"""
PROBS Tournament - Round-robin tournament between all PROBS checkpoints.
All checkpoints play against all other checkpoints.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import yaml
from cross_arena import Arena, load_probs_agent


def find_checkpoints(checkpoints_dir):
    """Find all .ckpt files in the checkpoints directory."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints = list(checkpoints_dir.glob('*.ckpt'))
    # Filter out temporary files
    checkpoints = [cp for cp in checkpoints if 'temp' not in cp.name.lower()]
    checkpoints.sort(key=lambda x: x.name)
    return checkpoints


def run_tournament(checkpoints_dir, config_path, num_games=20, device='cuda', output_file=None, verbose=False):
    """
    Run a round-robin tournament between all PROBS checkpoints.
    
    Args:
        checkpoints_dir: Directory containing .ckpt files
        config_path: Path to PROBS config YAML
        num_games: Number of games per pair (default: 20, 10 per side)
        device: Device to use ('cuda' or 'cpu')
        output_file: Path to save results JSON (default: tournament_results.json in checkpoints_dir)
        verbose: Show detailed game output
    """
    checkpoints_dir = Path(checkpoints_dir)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(checkpoints_dir)
    
    if len(checkpoints) < 2:
        print(f"❌ Need at least 2 checkpoints for tournament, found {len(checkpoints)}")
        return
    
    print(f"🏆 PROBS TOURNAMENT START")
    print(f"   Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"     - {cp.name}")
    
    # Load config
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    env_name = config["env"]["name"]
    
    # Create arena
    arena = Arena(env_name)
    
    # Generate all pairs (round-robin)
    pairs = list(itertools.combinations(checkpoints, 2))
    total_pairs = len(pairs)
    
    print(f"\n   Total pairs: {total_pairs}")
    print(f"   Games per pair: {num_games} ({num_games//2} per side)\n")
    
    # Initialize results table
    results = {cp.name: {'wins': 0, 'losses': 0, 'draws': 0, 'score': 0.0, 'games': 0} 
               for cp in checkpoints}
    
    # Run tournament
    start_time = datetime.now()
    
    for idx, (cp1, cp2) in enumerate(pairs):
        pair_num = idx + 1
        progress = (pair_num / total_pairs) * 100
        
        print(f"\n{'='*70}")
        print(f"⚔️  Pair {pair_num}/{total_pairs} ({progress:.1f}%): {cp1.name} vs {cp2.name}")
        print(f"{'='*70}")
        
        try:
            # Load agents
            agent1 = load_probs_agent(config_path, str(cp1))
            agent1_name = cp1.name
            
            agent2 = load_probs_agent(config_path, str(cp2))
            agent2_name = cp2.name
            
            # Play match
            match_results = arena.run_match(agent1, agent2, n_games=num_games, verbose=verbose)
            
            # Extract wins
            a1_wins = match_results["a1w"] + match_results["a1b"]
            a2_wins = match_results["a2w"] + match_results["a2b"]
            draws = match_results["draws"]
            
            # Update results
            results[agent1_name]['wins'] += a1_wins
            results[agent1_name]['losses'] += a2_wins
            results[agent1_name]['draws'] += draws
            results[agent1_name]['score'] += a1_wins + 0.5 * draws
            results[agent1_name]['games'] += num_games
            
            results[agent2_name]['wins'] += a2_wins
            results[agent2_name]['losses'] += a1_wins
            results[agent2_name]['draws'] += draws
            results[agent2_name]['score'] += a2_wins + 0.5 * draws
            results[agent2_name]['games'] += num_games
            
            print(f"✅ {agent1_name}: {a1_wins} wins, {agent2_name}: {a2_wins} wins, {draws} draws")
            
        except Exception as e:
            print(f"❌ Error in pair {pair_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # minutes
    
    # Sort by score (wins + 0.5*draws)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Print final leaderboard
    print(f"\n{'='*70}")
    print(f"🏆 TOURNAMENT COMPLETE! (Duration: {duration:.1f} minutes)")
    print(f"{'='*70}")
    print(f"\n📊 FINAL LEADERBOARD:\n")
    print(f"{'Rank':<6} {'Checkpoint':<50} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Score':<8} {'Win%':<8}")
    print(f"{'-'*6} {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for rank, (name, stats) in enumerate(sorted_results, 1):
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        win_pct = (stats['wins'] / total_games * 100) if total_games > 0 else 0.0
        print(f"{rank:<6} {name:<50} {stats['wins']:<8} {stats['losses']:<8} {stats['draws']:<8} {stats['score']:<8.1f} {win_pct:<7.1f}%")
    
    # Save results to JSON
    if output_file is None:
        output_file = checkpoints_dir / 'tournament_results.json'
    else:
        output_file = Path(output_file)
    
    tournament_data = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_minutes': duration,
        'num_checkpoints': len(checkpoints),
        'num_pairs': total_pairs,
        'games_per_pair': num_games,
        'device': device,
        'results': results,
        'leaderboard': [
            {
                'rank': rank,
                'checkpoint': name,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'draws': stats['draws'],
                'score': stats['score'],
                'win_percentage': (stats['wins'] / (stats['wins'] + stats['losses'] + stats['draws']) * 100) if (stats['wins'] + stats['losses'] + stats['draws']) > 0 else 0.0
            }
            for rank, (name, stats) in enumerate(sorted_results, 1)
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tournament_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n🏆 CHAMPION: {sorted_results[0][0]} (Score: {sorted_results[0][1]['score']:.1f})")


def main():
    parser = argparse.ArgumentParser(description='PROBS Round-Robin Tournament')
    parser.add_argument('--checkpoints-dir', 
                       default='../../models/probs/checkpoints',
                       help='Directory containing PROBS checkpoints (.ckpt files)')
    parser.add_argument('--config',
                       default='configs/train_togyzkumalak.yaml',
                       help='Path to PROBS config YAML file')
    parser.add_argument('--games', type=int, default=20,
                       help='Number of games per pair (default: 20, 10 per side)')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--output',
                       help='Output JSON file (default: tournament_results.json in checkpoints_dir)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed game output')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    checkpoints_dir = (script_dir / args.checkpoints_dir).resolve()
    config_path = (script_dir / args.config).resolve()
    
    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    run_tournament(
        checkpoints_dir=str(checkpoints_dir),
        config_path=str(config_path),
        num_games=args.games,
        device=args.device,
        output_file=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
