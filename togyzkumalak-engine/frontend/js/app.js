/**
 * Togyzkumalak Engine - Main Application
 * Handles game flow, UI interactions, and state management.
 */

class TogyzkumalakApp {
    constructor() {
        // Game state
        this.gameId = null;
        this.gameState = null;
        this.playerColor = 'white';
        this.aiLevel = 3;
        this.isClassicView = false;
        this.isMyTurn = false;
        
        // Board renderers
        this.modernBoard = null;
        this.classicBoard = null;
        
        // DOM elements
        this.elements = {};
        
        this.init();
    }

    /**
     * Initialize the application.
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        this.initBoards();
        this.loadEloStats();
    }

    /**
     * Cache DOM elements.
     */
    cacheElements() {
        this.elements = {
            // Panels
            setupPanel: document.getElementById('setupPanel'),
            gamePanel: document.getElementById('gamePanel'),
            
            // Setup
            colorBtns: document.querySelectorAll('.color-btn'),
            levelBtns: document.querySelectorAll('.level-btn'),
            btnStartGame: document.getElementById('btnStartGame'),
            
            // Game info
            turnIndicator: document.getElementById('turnIndicator'),
            moveCounter: document.getElementById('moveCounter'),
            aiElo: document.getElementById('aiElo'),
            playerElo: document.getElementById('playerElo'),
            lastMove: document.getElementById('lastMove'),
            
            // Boards
            modernBoard: document.getElementById('modernBoard'),
            classicBoard: document.getElementById('classicBoard'),
            btnViewToggle: document.getElementById('btnViewToggle'),
            viewIcon: document.getElementById('viewIcon'),
            
            // Controls
            btnUndo: document.getElementById('btnUndo'),
            btnNewGame: document.getElementById('btnNewGame'),
            btnResign: document.getElementById('btnResign'),
            
            // Move history
            moveList: document.getElementById('moveList'),
            
            // Analysis
            btnAnalyze: document.getElementById('btnAnalyze'),
            btnSuggest: document.getElementById('btnSuggest'),
            analysisContent: document.getElementById('analysisContent'),
            
            // ELO stats
            currentElo: document.getElementById('currentElo'),
            totalGames: document.getElementById('totalGames'),
            totalWins: document.getElementById('totalWins'),
            
            // Modal
            gameOverModal: document.getElementById('gameOverModal'),
            gameOverTitle: document.getElementById('gameOverTitle'),
            winnerText: document.getElementById('winnerText'),
            scoreText: document.getElementById('scoreText'),
            eloChangeText: document.getElementById('eloChangeText'),
            btnPlayAgain: document.getElementById('btnPlayAgain'),
            btnCloseModal: document.getElementById('btnCloseModal'),
            
            // AI thinking
            aiThinking: document.getElementById('aiThinking')
        };
    }

    /**
     * Bind event listeners.
     */
    bindEvents() {
        // Color selection
        this.elements.colorBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.elements.colorBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.playerColor = btn.dataset.color;
            });
        });
        
        // Level selection
        this.elements.levelBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.elements.levelBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.aiLevel = parseInt(btn.dataset.level);
            });
        });
        
        // Start game
        this.elements.btnStartGame.addEventListener('click', () => this.startGame());
        
        // View toggle
        this.elements.btnViewToggle.addEventListener('click', () => this.toggleView());
        
        // Game controls
        this.elements.btnNewGame.addEventListener('click', () => this.showSetup());
        this.elements.btnResign.addEventListener('click', () => this.resign());
        
        // Analysis
        this.elements.btnAnalyze.addEventListener('click', () => this.analyzePosition());
        this.elements.btnSuggest.addEventListener('click', () => this.suggestMove());
        
        // Modal
        this.elements.btnPlayAgain.addEventListener('click', () => {
            this.hideModal();
            this.startGame();
        });
        this.elements.btnCloseModal.addEventListener('click', () => {
            this.hideModal();
            this.showSetup();
        });
    }

    /**
     * Initialize board renderers.
     */
    initBoards() {
        this.modernBoard = new ModernBoard({
            whitePits: 'whitePits',
            blackPits: 'blackPits',
            kazanWhite: 'kazanWhite',
            kazanBlack: 'kazanBlack',
            pitLabels: 'pitLabels'
        });
        
        this.classicBoard = new ClassicBoard('classicBoard');
        
        // Set move callbacks
        this.modernBoard.setMoveCallback((move) => this.makeMove(move));
        this.classicBoard.setMoveCallback((move) => this.makeMove(move));
    }

    /**
     * Load ELO statistics.
     */
    async loadEloStats() {
        try {
            const stats = await api.getEloStats();
            if (stats.human) {
                this.elements.currentElo.textContent = stats.human.current_elo;
                this.elements.totalGames.textContent = stats.human.games_played;
                this.elements.totalWins.textContent = stats.human.wins;
            }
        } catch (error) {
            console.error('Failed to load ELO stats:', error);
        }
    }

    /**
     * Show setup panel.
     */
    showSetup() {
        this.elements.setupPanel.classList.remove('hidden');
        this.elements.gamePanel.classList.add('hidden');
        this.gameId = null;
        this.gameState = null;
    }

    /**
     * Start a new game.
     */
    async startGame() {
        try {
            this.elements.btnStartGame.disabled = true;
            this.elements.btnStartGame.textContent = 'Starting...';
            
            const response = await api.createGame(this.playerColor, this.aiLevel);
            
            this.gameId = response.game_id;
            this.gameState = response;
            
            // Configure boards
            this.modernBoard.setHumanColor(this.playerColor);
            this.classicBoard.setHumanColor(this.playerColor);
            
            // Update UI
            this.elements.setupPanel.classList.add('hidden');
            this.elements.gamePanel.classList.remove('hidden');
            
            // Update player labels
            if (this.playerColor === 'white') {
                document.querySelector('.player-white .player-label').textContent = '⚪ АҚ (You)';
                document.querySelector('.player-black .player-label').textContent = '⚫ ҚАРА (AI)';
            } else {
                document.querySelector('.player-white .player-label').textContent = '⚪ АҚ (AI)';
                document.querySelector('.player-black .player-label').textContent = '⚫ ҚАРА (You)';
            }
            
            // Update ELO displays
            this.elements.aiElo.textContent = `ELO: ${response.ai_elo || 1500}`;
            this.elements.playerElo.textContent = `ELO: ${response.player_elo || 1500}`;
            
            // Clear move history
            this.elements.moveList.innerHTML = '';
            this.elements.analysisContent.innerHTML = '<p class="placeholder">Click "Analyze Position" to get Gemini\'s analysis.</p>';
            
            // Render board
            this.updateBoard(response);
            
        } catch (error) {
            console.error('Failed to start game:', error);
            alert('Failed to start game: ' + error.message);
        } finally {
            this.elements.btnStartGame.disabled = false;
            this.elements.btnStartGame.textContent = '▶️ Start Game';
        }
    }

    /**
     * Make a move.
     */
    async makeMove(move) {
        if (!this.gameId) return;
        
        try {
            // Disable board during move
            this.isMyTurn = false;
            
            const response = await api.makeMove(move);
            
            // Update state
            this.gameState = response;
            this.updateBoard(response);
            
            // Add to move history
            await this.updateMoveHistory();
            
            // Check for AI move
            if (response.ai_move) {
                this.showAIThinking();
                
                // Small delay to show thinking animation
                await new Promise(resolve => setTimeout(resolve, 300));
                
                this.hideAIThinking();
                
                // Highlight AI move
                const aiPlayer = this.playerColor === 'white' ? 'black' : 'white';
                this.modernBoard.highlightLastMove(response.ai_move.move, aiPlayer);
                this.classicBoard.highlightLastMove(response.ai_move.move, aiPlayer);
                
                // Update history again after AI move
                await this.updateMoveHistory();
            }
            
            // Check for game over
            if (response.status === 'finished') {
                this.handleGameOver(response);
            }
            
        } catch (error) {
            console.error('Failed to make move:', error);
            alert('Invalid move: ' + error.message);
        }
    }

    /**
     * Update board display.
     */
    updateBoard(state) {
        const board = state.board;
        
        // Update boards
        this.modernBoard.render(board);
        this.classicBoard.render(board);
        
        // Update turn indicator
        this.isMyTurn = board.current_player === this.playerColor;
        this.elements.turnIndicator.textContent = this.isMyTurn ? 'Your Turn' : 'AI\'s Turn';
        this.elements.turnIndicator.style.color = this.isMyTurn ? '#2ecc71' : '#e94560';
        
        // Update move counter
        this.elements.moveCounter.textContent = `Move: ${state.move_count || 0}`;
        
        // Update last move display
        if (state.last_move) {
            this.elements.lastMove.textContent = `Last move: ${state.last_move}`;
        }
    }

    /**
     * Update move history.
     */
    async updateMoveHistory() {
        try {
            const response = await api.getMoveHistory();
            const moves = response.moves || [];
            
            this.elements.moveList.innerHTML = '';
            
            for (let i = 0; i < moves.length; i += 2) {
                const moveNum = Math.floor(i / 2) + 1;
                const whiteMove = moves[i]?.notation || '';
                const blackMove = moves[i + 1]?.notation || '';
                
                const entry = document.createElement('div');
                entry.className = 'move-entry';
                entry.innerHTML = `
                    <span class="move-number">${moveNum}.</span>
                    <span class="move-white">${whiteMove}</span>
                    <span class="move-black">${blackMove}</span>
                `;
                this.elements.moveList.appendChild(entry);
            }
            
            // Scroll to bottom
            this.elements.moveList.scrollTop = this.elements.moveList.scrollHeight;
            
        } catch (error) {
            console.error('Failed to update move history:', error);
        }
    }

    /**
     * Handle game over.
     */
    handleGameOver(state) {
        const isWinner = state.winner === this.playerColor;
        const isDraw = state.winner === 'draw';
        
        // Update modal
        if (isDraw) {
            this.elements.gameOverTitle.textContent = '🤝 Draw!';
            this.elements.winnerText.textContent = 'The game ended in a draw.';
        } else if (isWinner) {
            this.elements.gameOverTitle.textContent = '🎉 Victory!';
            this.elements.winnerText.textContent = 'Congratulations! You win!';
        } else {
            this.elements.gameOverTitle.textContent = '😔 Defeat';
            this.elements.winnerText.textContent = 'The AI wins this time.';
        }
        
        // Score
        const whiteScore = state.board?.white_kazan || 0;
        const blackScore = state.board?.black_kazan || 0;
        this.elements.scoreText.textContent = `Final Score: ${whiteScore} - ${blackScore}`;
        
        // ELO change
        if (state.elo_change !== undefined) {
            const change = state.elo_change;
            const sign = change >= 0 ? '+' : '';
            this.elements.eloChangeText.textContent = `ELO Change: ${sign}${change}`;
            this.elements.eloChangeText.className = `elo-change ${change >= 0 ? 'positive' : 'negative'}`;
            
            // Update current ELO display
            if (state.new_elo) {
                this.elements.currentElo.textContent = state.new_elo;
            }
        }
        
        // Show modal
        this.showModal();
        
        // Refresh stats
        this.loadEloStats();
    }

    /**
     * Resign the game.
     */
    async resign() {
        if (!this.gameId) return;
        
        if (!confirm('Are you sure you want to resign?')) return;
        
        try {
            const response = await api.resign();
            this.handleGameOver(response);
        } catch (error) {
            console.error('Failed to resign:', error);
        }
    }

    /**
     * Analyze current position.
     */
    async analyzePosition() {
        if (!this.gameId) return;
        
        this.elements.analysisContent.innerHTML = '<p>Analyzing position...</p>';
        this.elements.btnAnalyze.disabled = true;
        
        try {
            const response = await api.analyzePosition();
            
            if (response.available) {
                this.elements.analysisContent.innerHTML = `
                    <div class="analysis-result">
                        ${this.formatAnalysis(response.analysis)}
                    </div>
                `;
            } else {
                this.elements.analysisContent.innerHTML = `
                    <p class="error">${response.error || 'Analysis not available'}</p>
                    <p class="placeholder">Configure GEMINI_API_KEY to enable AI analysis.</p>
                `;
            }
        } catch (error) {
            this.elements.analysisContent.innerHTML = `
                <p class="error">Failed to analyze: ${error.message}</p>
            `;
        } finally {
            this.elements.btnAnalyze.disabled = false;
        }
    }

    /**
     * Get move suggestion.
     */
    async suggestMove() {
        if (!this.gameId) return;
        
        this.elements.analysisContent.innerHTML = '<p>Getting suggestion...</p>';
        this.elements.btnSuggest.disabled = true;
        
        try {
            const response = await api.suggestMove();
            
            if (response.available) {
                this.elements.analysisContent.innerHTML = `
                    <div class="suggestion-result">
                        <p><strong>Suggested Move: ${response.suggested_move}</strong></p>
                        <p>${response.explanation}</p>
                    </div>
                `;
                
                // Highlight suggested move on board
                if (response.suggested_move) {
                    // Could add visual highlight here
                }
            } else {
                this.elements.analysisContent.innerHTML = `
                    <p class="error">${response.error || 'Suggestion not available'}</p>
                `;
            }
        } catch (error) {
            this.elements.analysisContent.innerHTML = `
                <p class="error">Failed to get suggestion: ${error.message}</p>
            `;
        } finally {
            this.elements.btnSuggest.disabled = false;
        }
    }

    /**
     * Format analysis text for display.
     */
    formatAnalysis(text) {
        if (!text) return '';
        
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/EVALUATION:/g, '<strong>📊 EVALUATION:</strong>')
            .replace(/BEST MOVE:/g, '<strong>🎯 BEST MOVE:</strong>')
            .replace(/KEY FACTORS:/g, '<strong>🔑 KEY FACTORS:</strong>')
            .replace(/WARNING:/g, '<strong>⚠️ WARNING:</strong>');
    }

    /**
     * Toggle between modern and classic view.
     */
    toggleView() {
        this.isClassicView = !this.isClassicView;
        const boardWrapper = document.querySelector('.board-wrapper');
        
        if (this.isClassicView) {
            this.elements.modernBoard.classList.add('hidden');
            this.elements.classicBoard.classList.remove('hidden');
            boardWrapper.classList.add('classic-active');
            this.elements.viewIcon.textContent = '🎨';
            this.elements.btnViewToggle.innerHTML = '<span>🎨</span> Modern View';
            
            // Re-render classic board
            if (this.gameState) {
                this.classicBoard.render(this.gameState.board);
            }
        } else {
            this.elements.modernBoard.classList.remove('hidden');
            this.elements.classicBoard.classList.add('hidden');
            boardWrapper.classList.remove('classic-active');
            this.elements.viewIcon.textContent = '🎨';
            this.elements.btnViewToggle.innerHTML = '<span>🪵</span> Classic View';
        }
    }

    /**
     * Show AI thinking indicator.
     */
    showAIThinking() {
        this.elements.aiThinking.classList.remove('hidden');
    }

    /**
     * Hide AI thinking indicator.
     */
    hideAIThinking() {
        this.elements.aiThinking.classList.add('hidden');
    }

    /**
     * Show game over modal.
     */
    showModal() {
        this.elements.gameOverModal.classList.remove('hidden');
    }

    /**
     * Hide game over modal.
     */
    hideModal() {
        this.elements.gameOverModal.classList.add('hidden');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TogyzkumalakApp();
});

