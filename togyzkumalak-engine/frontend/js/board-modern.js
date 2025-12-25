/**
 * Togyzkumalak Engine - Modern Board Renderer
 * HTML/CSS-based board visualization.
 */

class ModernBoard {
    constructor(containerIds) {
        this.whitePitsContainer = document.getElementById(containerIds.whitePits);
        this.blackPitsContainer = document.getElementById(containerIds.blackPits);
        this.whiteKazan = document.getElementById(containerIds.kazanWhite);
        this.blackKazan = document.getElementById(containerIds.kazanBlack);
        this.pitLabelsContainer = document.getElementById(containerIds.pitLabels);
        
        this.pits = { white: [], black: [] };
        this.humanColor = 'white';
        this.legalMoves = [];
        this.onMoveClick = null;
        this.lastMove = null;
        
        this.pitNames = [
            'Арт', 'Тектұрмас', 'Ат өтпес', 'Атсыратар', 
            'Бел', 'Белбасар', 'Қандықақпан', 'Көкмойын', 'Маңдай'
        ];
        
        this.init();
    }

    /**
     * Initialize the board elements.
     */
    init() {
        this.whitePitsContainer.innerHTML = '';
        this.blackPitsContainer.innerHTML = '';
        this.pitLabelsContainer.innerHTML = '';
        
        // Create white pits (1-9, left to right)
        for (let i = 0; i < 9; i++) {
            const pit = this.createPit(i, 'white');
            this.whitePitsContainer.appendChild(pit);
            this.pits.white.push(pit);
        }
        
        // Create black pits (9-1, right to left when viewed from black's perspective)
        for (let i = 8; i >= 0; i--) {
            const pit = this.createPit(i, 'black');
            this.blackPitsContainer.appendChild(pit);
            this.pits.black.push(pit);
        }
        
        // Create pit labels
        for (let i = 0; i < 9; i++) {
            const label = document.createElement('div');
            label.className = 'pit-label';
            label.textContent = `${i + 1}`;
            this.pitLabelsContainer.appendChild(label);
        }
        
        // Initialize with starting position
        this.render({
            white_pits: [9, 9, 9, 9, 9, 9, 9, 9, 9],
            black_pits: [9, 9, 9, 9, 9, 9, 9, 9, 9],
            white_kazan: 0,
            black_kazan: 0,
            white_tuzduk: 0,
            black_tuzduk: 0,
            legal_moves: []
        });
    }

    /**
     * Create a pit element.
     */
    createPit(index, color) {
        const pit = document.createElement('div');
        pit.className = 'pit';
        pit.dataset.index = index;
        pit.dataset.color = color;
        pit.setAttribute('role', 'button');
        pit.setAttribute('tabindex', '0');
        pit.setAttribute('aria-label', `Pit ${index + 1} - ${this.pitNames[index]}`);
        
        // Kumalak count
        const count = document.createElement('span');
        count.className = 'pit-count';
        count.textContent = '9';
        pit.appendChild(count);
        
        // Pit number
        const number = document.createElement('span');
        number.className = 'pit-number';
        number.textContent = index + 1;
        pit.appendChild(number);
        
        // Pit name
        const name = document.createElement('span');
        name.className = 'pit-name';
        name.textContent = this.pitNames[index];
        pit.appendChild(name);
        
        // Click handler
        pit.addEventListener('click', () => {
            if (pit.classList.contains('playable') && this.onMoveClick) {
                this.onMoveClick(index + 1); // 1-based move
            }
        });
        
        return pit;
    }

    /**
     * Set the human player's color.
     */
    setHumanColor(color) {
        this.humanColor = color;
    }

    /**
     * Set the callback for move clicks.
     */
    setMoveCallback(callback) {
        this.onMoveClick = callback;
    }

    /**
     * Render the board state.
     */
    render(boardState) {
        const {
            white_pits,
            black_pits,
            white_kazan,
            black_kazan,
            white_tuzduk,
            black_tuzduk,
            legal_moves,
            current_player
        } = boardState;
        
        this.legalMoves = legal_moves || [];
        
        // Update white pits
        for (let i = 0; i < 9; i++) {
            const pit = this.pits.white[i];
            const value = white_pits[i];
            const count = pit.querySelector('.pit-count');
            
            if (value === -1 || black_tuzduk === i + 1) {
                // This is black's tuzduk
                count.textContent = 'X';
                pit.classList.add('tuzduk');
            } else {
                count.textContent = value;
                pit.classList.remove('tuzduk');
            }
            
            // Playable?
            const isHumanTurn = current_player === this.humanColor;
            const isPlayable = this.humanColor === 'white' && 
                              isHumanTurn && 
                              this.legalMoves.includes(i) &&
                              value > 0;
            
            pit.classList.toggle('playable', isPlayable);
        }
        
        // Update black pits (remember they're reversed in display)
        for (let i = 0; i < 9; i++) {
            const displayIndex = 8 - i;
            const pit = this.pits.black[displayIndex];
            const value = black_pits[i];
            const count = pit.querySelector('.pit-count');
            
            if (value === -1 || white_tuzduk === i + 1) {
                // This is white's tuzduk
                count.textContent = 'X';
                pit.classList.add('tuzduk');
            } else {
                count.textContent = value;
                pit.classList.remove('tuzduk');
            }
            
            // Playable?
            const isHumanTurn = current_player === this.humanColor;
            const isPlayable = this.humanColor === 'black' && 
                              isHumanTurn && 
                              this.legalMoves.includes(i) &&
                              value > 0;
            
            pit.classList.toggle('playable', isPlayable);
        }
        
        // Update kazans
        this.whiteKazan.querySelector('.kazan-value').textContent = white_kazan;
        this.blackKazan.querySelector('.kazan-value').textContent = black_kazan;
    }

    /**
     * Highlight the last move.
     */
    highlightLastMove(move, player) {
        // Clear previous highlight
        document.querySelectorAll('.pit.last-move').forEach(pit => {
            pit.classList.remove('last-move');
        });
        
        if (move === null || move === undefined) return;
        
        const index = move - 1; // Convert to 0-based
        const pits = player === 'white' ? this.pits.white : this.pits.black;
        
        if (player === 'black') {
            // Black pits are reversed in display
            pits[8 - index].classList.add('last-move');
        } else {
            pits[index].classList.add('last-move');
        }
        
        this.lastMove = { move, player };
    }

    /**
     * Show move probabilities from AI.
     */
    showProbabilities(probs, player) {
        const pits = player === 'white' ? this.pits.white : this.pits.black;
        
        for (let i = 0; i < 9; i++) {
            const pit = player === 'black' ? pits[8 - i] : pits[i];
            const prob = probs[i] || 0;
            
            if (prob > 0.1) {
                pit.style.outline = `3px solid rgba(212, 175, 55, ${prob})`;
            } else {
                pit.style.outline = 'none';
            }
        }
    }

    /**
     * Clear probability display.
     */
    clearProbabilities() {
        [...this.pits.white, ...this.pits.black].forEach(pit => {
            pit.style.outline = 'none';
        });
    }

    /**
     * Animate a move.
     */
    async animateMove(fromPit, player) {
        const pits = player === 'white' ? this.pits.white : this.pits.black;
        const pit = player === 'black' ? pits[8 - (fromPit - 1)] : pits[fromPit - 1];
        
        // Pulse animation
        pit.style.transform = 'scale(1.2)';
        await new Promise(resolve => setTimeout(resolve, 150));
        pit.style.transform = '';
    }
}

// Export
window.ModernBoard = ModernBoard;

