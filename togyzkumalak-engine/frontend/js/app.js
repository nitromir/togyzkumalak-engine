/**
 * Togyzkumalak Engine - Main Application
 * Simplified: Classic Board only
 */

class TogyzkumalakApp {
    constructor() {
        // Game state
        this.gameId = null;
        this.gameState = null;
        this.playerColor = 'white';
        this.aiLevel = 3;
        this.isMyTurn = false;
        this.confidenceEnabled = true;
        
        // Board renderer (Classic only)
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
        this.initBoard();
        this.loadEloStats();
        this.loadConfidenceSetting();  // Apply saved preference at startup
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
            
            // Board
            classicBoard: document.getElementById('classicBoard'),
            boardWrapper: document.querySelector('#gamePanel .board-wrapper'),
            
            // Controls
            btnUndo: document.getElementById('btnUndo'),
            btnNewGame: document.getElementById('btnNewGame'),
            btnResign: document.getElementById('btnResign'),
            btnToggleConfidence: document.getElementById('btnToggleConfidence'),
            
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
            aiThinking: document.getElementById('aiThinking'),
            
            // Score panel elements
            whiteKazan: document.getElementById('whiteKazan'),
            blackKazan: document.getElementById('blackKazan'),
            whiteBarFill: document.getElementById('whiteBarFill'),
            blackBarFill: document.getElementById('blackBarFill'),
            whiteKumalaks: document.getElementById('whiteKumalaks'),
            blackKumalaks: document.getElementById('blackKumalaks'),
            leftAvatar: document.getElementById('leftAvatar'),
            rightAvatar: document.getElementById('rightAvatar')
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
        
        // Game controls
        this.elements.btnNewGame.addEventListener('click', () => this.showSetup());
        this.elements.btnResign.addEventListener('click', () => this.resign());
        this.elements.btnToggleConfidence?.addEventListener('click', () => this.toggleConfidence());
        
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
     * Initialize Classic Board.
     */
    initBoard() {
        this.classicBoard = new ClassicBoard('classicBoard');
        this.classicBoard.setMoveCallback((move) => this.makeMove(move));
    }

    loadConfidenceSetting() {
        try {
            const saved = localStorage.getItem('confidenceEnabled');
            if (saved === '0' || saved === '1') {
                this.confidenceEnabled = saved === '1';
            }
        } catch (e) {
            // ignore
        }
        this.applyConfidenceSetting();
    }

    applyConfidenceSetting() {
        this.classicBoard?.setShowProbabilities(this.confidenceEnabled);
        if (this.elements.btnToggleConfidence) {
            this.elements.btnToggleConfidence.textContent = this.confidenceEnabled
                ? '📊 Уверенность: ON'
                : '📊 Уверенность: OFF';
            this.elements.btnToggleConfidence.classList.toggle('active', this.confidenceEnabled);
        }
        if (!this.confidenceEnabled) {
            this.classicBoard?.setProbabilities(null);
        }
    }

    toggleConfidence() {
        this.confidenceEnabled = !this.confidenceEnabled;
        try {
            localStorage.setItem('confidenceEnabled', this.confidenceEnabled ? '1' : '0');
        } catch (e) {
            // ignore
        }
        this.applyConfidenceSetting();
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
            
            // Store AI model info
            this.currentAiModel = response.ai_model || null;
            
            // Configure board
            this.classicBoard.setHumanColor(this.playerColor);
            this.loadConfidenceSetting();
            
            // Update UI
            this.elements.setupPanel.classList.add('hidden');
            this.elements.gamePanel.classList.remove('hidden');
            
            // Update player labels with model info
            this.updatePlayerLabels();
            
            // Update ELO displays
            this.elements.aiElo.textContent = `ELO: ${response.ai_elo || 1500}`;
            this.elements.playerElo.textContent = `ELO: ${response.player_elo || 1500}`;
            
            // Update score panel avatars
            this.updateScoreAvatars();
            
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
     * Update player labels with model info
     */
    updatePlayerLabels() {
        const modelInfo = this.currentAiModel || {};
        const modelName = modelInfo.name || 'default';
        const modelType = modelInfo.type || 'default';
        const useMcts = modelInfo.use_mcts || false;
        
        // Format AI label
        let aiLabel;
        if (modelType === 'alphazero') {
            aiLabel = `🦾 ${modelName}${useMcts ? ' +MCTS' : ''}`;
        } else if (modelType === 'gym') {
            aiLabel = `🧠 ${modelName}`;
        } else {
            aiLabel = `🤖 AI Lv${this.aiLevel}`;
        }
        
        // Truncate long names
        if (aiLabel.length > 25) {
            aiLabel = aiLabel.substring(0, 22) + '...';
        }
        
        if (this.playerColor === 'white') {
            document.querySelector('.player-white .player-label').textContent = '⚪ АҚ (You)';
            document.querySelector('.player-black .player-label').textContent = `⚫ ${aiLabel}`;
        } else {
            document.querySelector('.player-white .player-label').textContent = `⚪ ${aiLabel}`;
            document.querySelector('.player-black .player-label').textContent = '⚫ ҚАРА (You)';
        }
        
        // Update model info display if element exists
        const modelInfoEl = document.getElementById('currentModelInfo');
        if (modelInfoEl) {
            modelInfoEl.textContent = aiLabel;
            modelInfoEl.title = `Model: ${modelName}\nType: ${modelType}\nMCTS: ${useMcts ? 'On' : 'Off'}`;
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
            this.triggerBoardGlow();
            
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
    async updateBoard(state) {
        const board = state.board;
        
        // Update board
        this.classicBoard.render(board);
        
        // Update score panel (kazans)
        this.updateScorePanel(board);
        
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

        // Fetch and show move probabilities if enabled and it's user turn
        if (this.confidenceEnabled && this.isMyTurn && state.status === 'in_progress') {
            try {
                const probResponse = await api.getMoveProbabilities(this.aiLevel);
                if (probResponse && probResponse.probabilities) {
                    this.classicBoard.setProbabilities(probResponse.probabilities);
                }
            } catch (e) {
                console.warn('Could not fetch probabilities:', e);
            }
        } else {
            this.classicBoard.setProbabilities(null);
        }
    }

    /**
     * Update the score panel showing kazans for both players.
     */
    updateScorePanel(board) {
        const whiteKazan = board.white_kazan || 0;
        const blackKazan = board.black_kazan || 0;
        const victoryTarget = 82;
        
        // Update numbers
        if (this.elements.whiteKazan) {
            this.elements.whiteKazan.textContent = whiteKazan;
        }
        if (this.elements.blackKazan) {
            this.elements.blackKazan.textContent = blackKazan;
        }
        
        // Update progress bars
        const whiteProgress = Math.min((whiteKazan / victoryTarget) * 100, 100);
        const blackProgress = Math.min((blackKazan / victoryTarget) * 100, 100);
        
        if (this.elements.whiteBarFill) {
            this.elements.whiteBarFill.style.width = `${whiteProgress}%`;
        }
        if (this.elements.blackBarFill) {
            this.elements.blackBarFill.style.width = `${blackProgress}%`;
        }
        
        // Update kumalak icons (1 icon per 20 kumalaks, max 5)
        this.updateKumalakIcons(this.elements.whiteKumalaks, whiteKazan);
        this.updateKumalakIcons(this.elements.blackKumalaks, blackKazan);
    }

    /**
     * Update kumalak icons in score panel.
     */
    updateKumalakIcons(container, count) {
        if (!container) return;
        
        const maxIcons = 5;
        const iconCount = Math.min(Math.ceil(count / 20), maxIcons);
        
        container.innerHTML = '';
        for (let i = 0; i < iconCount; i++) {
            const icon = document.createElement('div');
            icon.className = 'score-kumalak-icon';
            container.appendChild(icon);
        }
    }

    /**
     * Update score panel avatars based on player color and AI level.
     * Left avatar = Black side, Right avatar = White side
     */
    updateScoreAvatars() {
        const leftAvatar = this.elements.leftAvatar;
        const rightAvatar = this.elements.rightAvatar;
        
        if (!leftAvatar || !rightAvatar) return;
        
        // Score panel layout: [Left Avatar] [BLACK] [VS] [WHITE] [Right Avatar]
        // If player is BLACK: Left = Player, Right = AI
        // If player is WHITE: Left = AI, Right = Player
        
        const isPlayerBlack = this.playerColor === 'black';
        
        if (isPlayerBlack) {
            // Player is Black (left side)
            leftAvatar.classList.remove('ai-avatar', 'gemini-active');
            leftAvatar.classList.add('player-avatar');
            leftAvatar.querySelector('.avatar-icon').textContent = '👤';
            leftAvatar.querySelector('.avatar-label').textContent = 'ВЫ';
            
            // AI is White (right side)
            rightAvatar.classList.remove('player-avatar');
            rightAvatar.classList.add('ai-avatar');
            if (this.aiLevel === 6) {
                rightAvatar.classList.add('gemini-active');
                rightAvatar.querySelector('.avatar-icon').textContent = '✨';
                rightAvatar.querySelector('.avatar-label').textContent = 'GEMINI';
            } else {
                rightAvatar.classList.remove('gemini-active');
                rightAvatar.querySelector('.avatar-icon').textContent = '🤖';
                rightAvatar.querySelector('.avatar-label').textContent = `AI L${this.aiLevel}`;
            }
        } else {
            // AI is Black (left side)
            leftAvatar.classList.remove('player-avatar');
            leftAvatar.classList.add('ai-avatar');
            if (this.aiLevel === 6) {
                leftAvatar.classList.add('gemini-active');
                leftAvatar.querySelector('.avatar-icon').textContent = '✨';
                leftAvatar.querySelector('.avatar-label').textContent = 'GEMINI';
            } else {
                leftAvatar.classList.remove('gemini-active');
                leftAvatar.querySelector('.avatar-icon').textContent = '🤖';
                leftAvatar.querySelector('.avatar-label').textContent = `AI L${this.aiLevel}`;
            }
            
            // Player is White (right side)
            rightAvatar.classList.remove('ai-avatar', 'gemini-active');
            rightAvatar.classList.add('player-avatar');
            rightAvatar.querySelector('.avatar-icon').textContent = '👤';
            rightAvatar.querySelector('.avatar-label').textContent = 'ВЫ';
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
        
        // Get model info
        const modelInfo = state.ai_model || this.currentAiModel || {};
        const modelName = modelInfo.name || 'default';
        const modelType = modelInfo.type || 'default';
        const useMcts = modelInfo.use_mcts || false;
        
        // Format model display name
        let modelDisplay = modelName;
        if (modelType === 'alphazero') {
            modelDisplay = `🦾 ${modelName}${useMcts ? ' (MCTS)' : ''}`;
        } else if (modelType === 'gym') {
            modelDisplay = `🧠 ${modelName}`;
        } else {
            modelDisplay = `🤖 Level ${this.aiLevel}`;
        }
        
        // Update modal
        if (isDraw) {
            this.elements.gameOverTitle.textContent = '🤝 Draw!';
            this.elements.winnerText.textContent = 'The game ended in a draw.';
        } else if (isWinner) {
            this.elements.gameOverTitle.textContent = '🎉 Victory!';
            this.elements.winnerText.textContent = `Congratulations! You beat ${modelDisplay}!`;
        } else {
            this.elements.gameOverTitle.textContent = '😔 Defeat';
            this.elements.winnerText.textContent = `${modelDisplay} wins this time.`;
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
                const moveLabel = response.suggested_move ? response.suggested_move : '?';
                
                this.elements.analysisContent.innerHTML = `
                    <div class="suggestion-result">
                        <p><strong>Рекомендуемый ход: ${moveLabel}</strong></p>
                        <p>${this.formatAnalysis(response.explanation)}</p>
                    </div>
                `;
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

        // IMPORTANT: Escape HTML from Gemini output to avoid accidental tag interpretation
        // (which can visually "truncate" the output if Gemini returns "<...>").
        const escaped = this.escapeHtml(text);

        return escaped
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/EVALUATION:/g, '<strong>📊 EVALUATION:</strong>')
            .replace(/ОЦЕНКА:/g, '<strong>📊 ОЦЕНКА:</strong>')
            .replace(/BEST MOVE:/g, '<strong>🎯 BEST MOVE:</strong>')
            .replace(/ЛУЧШИЙ ХОД:/g, '<strong>🎯 ЛУЧШИЙ ХОД:</strong>')
            .replace(/РЕКОМЕНДУЕМЫЙ ХОД:/g, '<strong>🎯 РЕКОМЕНДУЕМЫЙ ХОД:</strong>')
            .replace(/ОБОСНОВАНИЕ:/g, '<strong>💡 ОБОСНОВАНИЕ:</strong>')
            .replace(/KEY FACTORS:/g, '<strong>🔑 KEY FACTORS:</strong>')
            .replace(/КЛЮЧЕВЫЕ ФАКТОРЫ:/g, '<strong>🔑 КЛЮЧЕВЫЕ ФАКТОРЫ:</strong>')
            .replace(/WARNING:/g, '<strong>⚠️ WARNING:</strong>')
            .replace(/УГРОЗЫ:/g, '<strong>⚠️ УГРОЗЫ:</strong>')
            .replace(/АНАЛИЗ ХОДОВ:/g, '<strong>📝 АНАЛИЗ ХОДОВ:</strong>');
    }

    escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    triggerBoardGlow() {
        const el = this.elements.boardWrapper;
        if (!el) return;
        el.classList.remove('board-glow');
        // force reflow to restart animation
        void el.offsetWidth;
        el.classList.add('board-glow');
        window.clearTimeout(this._glowTimer);
        this._glowTimer = window.setTimeout(() => {
            el.classList.remove('board-glow');
        }, 900);
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
    
    // Initialize panel expand buttons
    initPanelExpandButtons();
});

/**
 * ============================================
 * FULLSCREEN PANEL FUNCTIONALITY
 * ============================================
 */

/**
 * Initialize expand buttons for all panels
 */
function initPanelExpandButtons() {
    // Select all panels that should have expand functionality
    const panels = document.querySelectorAll('.panel');
    
    panels.forEach((panel, index) => {
        // Get panel title for modal header
        const titleEl = panel.querySelector('h2, h3');
        const title = titleEl ? titleEl.textContent : `Panel ${index + 1}`;
        
        // Create expand button
        const expandBtn = document.createElement('button');
        expandBtn.className = 'panel-expand-btn';
        expandBtn.innerHTML = '⛶';
        expandBtn.title = 'Развернуть на весь экран';
        expandBtn.setAttribute('data-panel-title', title);
        
        expandBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openFullscreenModal(panel, title);
        });
        
        panel.appendChild(expandBtn);
    });
}

/**
 * Open fullscreen modal with panel content
 */
function openFullscreenModal(panel, title) {
    const modal = document.getElementById('fullscreenModal');
    const modalTitle = document.getElementById('fullscreenModalTitle');
    const modalBody = document.getElementById('fullscreenModalBody');
    
    if (!modal || !modalBody) return;
    
    // Set title
    modalTitle.textContent = title;
    
    // Clone panel content (excluding the expand button itself)
    const panelClone = panel.cloneNode(true);
    
    // Remove expand button from clone
    const clonedExpandBtn = panelClone.querySelector('.panel-expand-btn');
    if (clonedExpandBtn) {
        clonedExpandBtn.remove();
    }
    
    // Clear and set content
    modalBody.innerHTML = '';
    modalBody.appendChild(panelClone);
    
    // Show modal
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    
    // Close on Escape key
    document.addEventListener('keydown', handleEscapeKey);
    
    // Close on backdrop click
    modal.addEventListener('click', handleBackdropClick);
}

/**
 * Close fullscreen modal
 */
function closeFullscreenModal() {
    const modal = document.getElementById('fullscreenModal');
    if (!modal) return;
    
    modal.classList.add('hidden');
    document.body.style.overflow = '';
    
    // Remove event listeners
    document.removeEventListener('keydown', handleEscapeKey);
    modal.removeEventListener('click', handleBackdropClick);
}

/**
 * Handle Escape key to close modal
 */
function handleEscapeKey(e) {
    if (e.key === 'Escape') {
        closeFullscreenModal();
    }
}

/**
 * Handle backdrop click to close modal
 */
function handleBackdropClick(e) {
    if (e.target.id === 'fullscreenModal') {
        closeFullscreenModal();
    }
}

// Expose to global scope for inline onclick handlers
window.openFullscreenModal = openFullscreenModal;
window.closeFullscreenModal = closeFullscreenModal;
