#!/usr/bin/env python3
"""
Data Parsers for Togyzkumalak Training Data

Converts various data formats to unified training format:
- Opening book files (open_tree*.txt)
- Championship game records (games.txt)
- PlayOK PGN format (all_results_combined.txt)

Output format compatible with gym training pipeline.
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Move:
    """Represents a single move in the game."""
    move_number: int
    player: str  # 'white' or 'black'
    action: int  # 0-8, pit index
    notation: str  # Original notation (e.g., '76', '91x')
    is_tuzduk: bool = False


@dataclass
class GameRecord:
    """Unified game record format."""
    game_id: str
    source: str  # 'opening_book', 'human_tournament', 'playok'
    timestamp: str
    white_player: Optional[str] = None
    black_player: Optional[str] = None
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: Optional[str] = None  # '1-0', '0-1', '0.5-0.5'
    moves: List[Dict] = None
    
    def __post_init__(self):
        if self.moves is None:
            self.moves = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


class OpeningBookParser:
    """
    Parses opening book files (open_tree*.txt).
    
    Format example:
    0   1 . . .   9 8   ( 2 4 - 2 8 ,   2A53>:   5 2 )
    1   2 .   7 8   ( 1 8 - 1 8 ,   всего:   3 6 )
    """
    
    def __init__(self):
        self.games: List[GameRecord] = []
        self.current_line: List[Move] = []
        
    def parse_file(self, filepath: str) -> List[GameRecord]:
        """Parse opening book file and extract game lines."""
        games = []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Clean up encoding artifacts
        content = content.replace('\x00', '').replace('��', '')
        
        lines = content.strip().split('\n')
        current_game_moves = []
        game_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse line: "index move_num. move (stats)"
            # Example: "1 2. 78 (18-18, всего: 36)"
            # Or: "0 1... 98 (24-28, всего: 52)"
            
            match = re.match(
                r'(\d+)\s+(\d+)\s*\.{1,3}\s*(\d{2})\s*x?\s*\(',
                line
            )
            
            if match:
                index = int(match.group(1))
                move_num = int(match.group(2))
                move_notation = match.group(3)
                
                is_tuzduk = 'x' in line.split('(')[0]
                
                # Determine player from move notation
                # "." means white's move, "..." means black's move
                is_black = '...' in line or '. . .' in line
                player = 'black' if is_black else 'white'
                
                # Convert notation to action (0-8)
                # Notation is like "76" meaning pit 7 to destination
                # First digit is the pit number (1-9), we convert to 0-8
                pit = int(move_notation[0]) - 1
                
                move = {
                    'move_number': move_num,
                    'player': player,
                    'action': pit,
                    'notation': move_notation,
                    'is_tuzduk': is_tuzduk
                }
                
                # Build game line
                if index == 0 or move_num == 1:
                    # New game line starting
                    if current_game_moves:
                        # Save previous game
                        game_count += 1
                        game = GameRecord(
                            game_id=f"opening_{Path(filepath).stem}_{game_count}",
                            source='opening_book',
                            timestamp=datetime.now().isoformat(),
                            moves=current_game_moves.copy()
                        )
                        games.append(game)
                    current_game_moves = [move]
                else:
                    current_game_moves.append(move)
        
        # Don't forget last game
        if current_game_moves:
            game_count += 1
            game = GameRecord(
                game_id=f"opening_{Path(filepath).stem}_{game_count}",
                source='opening_book',
                timestamp=datetime.now().isoformat(),
                moves=current_game_moves.copy()
            )
            games.append(game)
        
        return games


class ChampionshipGamesParser:
    """
    Parses championship game records (games.txt).
    
    Format varies but generally:
    - Game headers with player names and dates
    - Moves in format: move_number. white_move black_move
    - Results at end
    """
    
    def parse_file(self, filepath: str) -> List[GameRecord]:
        """Parse championship games file."""
        games = []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split into game sections
        # Look for patterns like "1. PlayerA - PlayerB" followed by moves
        
        # Pattern for game start with player names
        game_pattern = re.compile(
            r'(\d+)\.\s*([A-Za-z]+)\s+([A-Z]?\.?)\s*[-–]\s*([A-Za-z]+)\s+([A-Z]?\.?)',
            re.MULTILINE
        )
        
        # Find all move sequences
        # Moves look like: "1. 76 98" or "1.\t76\n98"
        move_line_pattern = re.compile(
            r'(\d+)\.\s*(\d{2})x?\s*[\n\t\s]*(\d{2})x?',
            re.MULTILINE
        )
        
        # Alternative format: moves on separate lines
        alt_move_pattern = re.compile(r'^(\d{2})x?$', re.MULTILINE)
        
        # Split content by result markers
        result_splits = re.split(r'Result\s+[\d\.,\s-]+', content)
        
        game_id = 0
        current_white = None
        current_black = None
        
        for section in result_splits:
            if not section.strip():
                continue
            
            # Extract player names if present
            name_match = re.search(
                r'([A-Z][a-z]+)\s+([A-Z])\.\s*\(.*?\)\s*\n\s*([A-Z][a-z]+)\s+([A-Z])\.',
                section
            )
            if name_match:
                current_white = f"{name_match.group(1)} {name_match.group(2)}."
                current_black = f"{name_match.group(3)} {name_match.group(4)}."
            
            # Try to extract moves
            moves = []
            
            # Standard format: 1. 76 98
            for match in move_line_pattern.finditer(section):
                move_num = int(match.group(1))
                white_move = match.group(2)
                black_move = match.group(3)
                
                # White move
                moves.append({
                    'move_number': move_num,
                    'player': 'white',
                    'action': int(white_move[0]) - 1,
                    'notation': white_move,
                    'is_tuzduk': 'x' in match.group(0).split()[1] if len(match.group(0).split()) > 1 else False
                })
                
                # Black move
                moves.append({
                    'move_number': move_num,
                    'player': 'black',
                    'action': int(black_move[0]) - 1,
                    'notation': black_move,
                    'is_tuzduk': 'x' in match.group(0).split()[-1] if len(match.group(0).split()) > 1 else False
                })
            
            if moves:
                game_id += 1
                
                # Determine result
                result = None
                if 'Result 1-0' in section or 'Result 1 - 0' in section:
                    result = '1-0'
                elif 'Result 0-1' in section or 'Result 0 - 1' in section:
                    result = '0-1'
                elif 'Result 0,5' in section or 'Result 0.5' in section:
                    result = '0.5-0.5'
                
                game = GameRecord(
                    game_id=f"championship_{game_id}",
                    source='human_tournament',
                    timestamp=datetime.now().isoformat(),
                    white_player=current_white,
                    black_player=current_black,
                    result=result,
                    moves=moves
                )
                games.append(game)
        
        return games


class PlayOKPGNParser:
    """
    Parses PlayOK PGN format (all_results_combined.txt).
    
    Format:
    [Event "9245757"]
    [Site "PlayOK"]
    [Date "2024.12.26"]
    [White "player1"]
    [Black "player2"]
    [Result "0-1"]
    [WhiteElo "1200"]
    [BlackElo "1172"]
    
    1. 87(10) 87(10) 2. 22 99(22) 3. 23 12(24) ...
    """
    
    def parse_file(self, filepath: str) -> List[GameRecord]:
        """Parse PlayOK PGN file."""
        games = []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split by double newlines (games are separated by empty lines)
        game_sections = re.split(r'\n\s*\n', content)
        
        current_headers = {}
        current_moves_text = ""
        
        for section in game_sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this is a header section
            if section.startswith('['):
                # Parse headers
                header_pattern = re.compile(r'\[(\w+)\s+"([^"]+)"\]')
                for match in header_pattern.finditer(section):
                    current_headers[match.group(1)] = match.group(2)
            else:
                # This is moves section
                current_moves_text = section
                
                # Process the game
                if current_headers:
                    game = self._process_game(current_headers, current_moves_text)
                    if game:
                        games.append(game)
                    current_headers = {}
                    current_moves_text = ""
        
        return games
    
    def _process_game(self, headers: Dict, moves_text: str) -> Optional[GameRecord]:
        """Process a single game from headers and moves text."""
        
        # Parse moves
        # Format: "1. 87(10) 87(10) 2. 22 99(22)"
        # The (N) indicates stones captured
        
        moves = []
        
        # Remove result at the end
        moves_text = re.sub(r'\s*[01]-[01]\s*\{.*?\}\s*$', '', moves_text)
        
        # Parse move pairs
        # Pattern: move_number. white_move black_move
        move_pattern = re.compile(
            r'(\d+)\.\s*(\d{2})(?:X|\([\d]+\))?\s*(\d{2})?(?:X|\([\d]+\))?'
        )
        
        for match in move_pattern.finditer(moves_text):
            move_num = int(match.group(1))
            white_move = match.group(2)
            black_move = match.group(3)
            
            # White move
            if white_move:
                is_tuzduk = 'X' in moves_text[match.start():match.end()].split()[1] if len(moves_text[match.start():match.end()].split()) > 1 else False
                moves.append({
                    'move_number': move_num,
                    'player': 'white',
                    'action': int(white_move[0]) - 1,
                    'notation': white_move,
                    'is_tuzduk': is_tuzduk
                })
            
            # Black move
            if black_move:
                moves.append({
                    'move_number': move_num,
                    'player': 'black',
                    'action': int(black_move[0]) - 1,
                    'notation': black_move,
                    'is_tuzduk': False
                })
        
        if not moves:
            return None
        
        game_id = headers.get('Event', 'unknown')
        
        return GameRecord(
            game_id=f"playok_{game_id}_{hashlib.md5(moves_text.encode()).hexdigest()[:8]}",
            source='playok',
            timestamp=f"{headers.get('Date', datetime.now().strftime('%Y.%m.%d'))}T{headers.get('Time', '00:00:00')}",
            white_player=headers.get('White'),
            black_player=headers.get('Black'),
            white_elo=int(headers.get('WhiteElo', 0)) if headers.get('WhiteElo') else None,
            black_elo=int(headers.get('BlackElo', 0)) if headers.get('BlackElo') else None,
            result=headers.get('Result'),
            moves=moves
        )


class TrainingDataConverter:
    """
    Converts GameRecord to neural network training format.
    
    Output format matches what gym_training.py expects:
    {
        "state": [...],  # 128-dim observation
        "action": 0-8,   # pit index
        "reward": float,
        "next_state": [...],
        "done": bool,
        "legal_moves": [0,1,1,1,1,1,1,1,1],
        "source": "human_tournament"
    }
    """
    
    def __init__(self):
        # Initial board state
        self.initial_pits = [9, 9, 9, 9, 9, 9, 9, 9, 9]
    
    def game_to_transitions(self, game: GameRecord) -> List[Dict]:
        """Convert a game to training transitions."""
        transitions = []
        
        # Simulate game state
        white_pits = self.initial_pits.copy()
        black_pits = self.initial_pits.copy()
        white_kazan = 0
        black_kazan = 0
        white_tuzduk = None
        black_tuzduk = None
        
        for i, move in enumerate(game.moves):
            is_last = i == len(game.moves) - 1
            
            # Generate observation (simplified - real implementation uses TogyzkumalakBoard)
            obs = self._make_observation(
                white_pits, black_pits, 
                white_kazan, black_kazan,
                white_tuzduk, black_tuzduk,
                move['player']
            )
            
            # Determine reward
            if is_last and game.result:
                if game.result == '1-0':
                    reward = 1.0 if move['player'] == 'white' else -1.0
                elif game.result == '0-1':
                    reward = -1.0 if move['player'] == 'white' else 1.0
                else:
                    reward = 0.0
            else:
                reward = 0.0
            
            # Legal moves (all pits with stones are legal)
            pits = white_pits if move['player'] == 'white' else black_pits
            legal_moves = [1 if p > 0 else 0 for p in pits]
            
            transition = {
                "state": obs,
                "action": move['action'],
                "reward": reward,
                "done": is_last,
                "legal_moves": legal_moves,
                "source": game.source,
                "game_id": game.game_id,
                "move_number": move['move_number'],
                "player": move['player'],
                "notation": move['notation']
            }
            
            transitions.append(transition)
            
            # Update state (simplified - doesn't fully simulate game)
            # In real use, you'd use TogyzkumalakBoard to simulate
        
        return transitions
    
    def _make_observation(
        self, 
        white_pits: List[int],
        black_pits: List[int],
        white_kazan: int,
        black_kazan: int,
        white_tuzduk: Optional[int],
        black_tuzduk: Optional[int],
        current_player: str
    ) -> List[float]:
        """Create 128-dim observation vector."""
        obs = []
        
        # White pits (normalized)
        for p in white_pits:
            obs.append(float(p) / 81.0)
        
        # Black pits (normalized)
        for p in black_pits:
            obs.append(float(p) / 81.0)
        
        # Kazans (normalized)
        obs.append(float(white_kazan) / 162.0)
        obs.append(float(black_kazan) / 162.0)
        
        # Tuzduk positions (one-hot, 9 each)
        for i in range(9):
            obs.append(1.0 if white_tuzduk == i else 0.0)
        for i in range(9):
            obs.append(1.0 if black_tuzduk == i else 0.0)
        
        # Current player (one-hot)
        obs.append(1.0 if current_player == 'white' else 0.0)
        obs.append(1.0 if current_player == 'black' else 0.0)
        
        # Pad to 128
        while len(obs) < 128:
            obs.append(0.0)
        
        return obs[:128]


def parse_all_data(
    opening_book_paths: List[str] = None,
    championship_path: str = None,
    playok_path: str = None,
    output_dir: str = "training_data"
) -> Dict:
    """
    Parse all available data sources and save to unified format.
    
    Args:
        opening_book_paths: List of paths to open_tree*.txt files
        championship_path: Path to games.txt
        playok_path: Path to all_results_combined.txt
        output_dir: Directory to save output
    
    Returns:
        Statistics about parsed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_games = []
    stats = {
        'opening_book': 0,
        'human_tournament': 0,
        'playok': 0,
        'total_games': 0,
        'total_moves': 0,
        'total_transitions': 0
    }
    
    # Parse opening books
    if opening_book_paths:
        parser = OpeningBookParser()
        for path in opening_book_paths:
            if os.path.exists(path):
                games = parser.parse_file(path)
                all_games.extend(games)
                stats['opening_book'] += len(games)
                print(f"Parsed {len(games)} games from {path}")
    
    # Parse championship games
    if championship_path and os.path.exists(championship_path):
        parser = ChampionshipGamesParser()
        games = parser.parse_file(championship_path)
        all_games.extend(games)
        stats['human_tournament'] += len(games)
        print(f"Parsed {len(games)} championship games from {championship_path}")
    
    # Parse PlayOK games
    if playok_path and os.path.exists(playok_path):
        parser = PlayOKPGNParser()
        games = parser.parse_file(playok_path)
        all_games.extend(games)
        stats['playok'] += len(games)
        print(f"Parsed {len(games)} PlayOK games from {playok_path}")
    
    stats['total_games'] = len(all_games)
    
    # Convert to training format
    converter = TrainingDataConverter()
    all_transitions = []
    
    for game in all_games:
        transitions = converter.game_to_transitions(game)
        all_transitions.extend(transitions)
        stats['total_moves'] += len(game.moves)
    
    stats['total_transitions'] = len(all_transitions)
    
    # Save games
    games_file = os.path.join(output_dir, 'human_games.jsonl')
    with open(games_file, 'w', encoding='utf-8') as f:
        for game in all_games:
            f.write(json.dumps(game.to_dict(), ensure_ascii=False) + '\n')
    
    # Save transitions
    transitions_file = os.path.join(output_dir, 'human_transitions.jsonl')
    with open(transitions_file, 'w', encoding='utf-8') as f:
        for t in all_transitions:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'parse_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Parsing Complete ===")
    print(f"Opening book games: {stats['opening_book']}")
    print(f"Tournament games: {stats['human_tournament']}")
    print(f"PlayOK games: {stats['playok']}")
    print(f"Total games: {stats['total_games']}")
    print(f"Total moves: {stats['total_moves']}")
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"\nSaved to: {output_dir}/")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse training data for Togyzkumalak')
    parser.add_argument('--opening-books', nargs='*', help='Paths to opening book files')
    parser.add_argument('--championship', help='Path to championship games file')
    parser.add_argument('--playok', help='Path to PlayOK PGN file')
    parser.add_argument('--output', default='training_data', help='Output directory')
    
    args = parser.parse_args()
    
    # Default paths if not specified
    if not any([args.opening_books, args.championship, args.playok]):
        # Try to find files in standard locations
        base_path = Path(__file__).parent.parent.parent.parent
        
        opening_books = list(base_path.glob('Android-APK/assets/internal/open_tree*.txt'))
        championship = base_path / 'games.txt'
        playok = base_path / 'all_results_combined.txt'
        
        args.opening_books = [str(p) for p in opening_books] if opening_books else None
        args.championship = str(championship) if championship.exists() else None
        args.playok = str(playok) if playok.exists() else None
    
    parse_all_data(
        opening_book_paths=args.opening_books,
        championship_path=args.championship,
        playok_path=args.playok,
        output_dir=args.output
    )

