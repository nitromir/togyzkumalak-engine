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
        this.confidenceModel = 'all'; // Default: Show all three
        this.autoAnalyzeEnabled = true;
        this.isStreaming = false;
        this.userScrolling = false;
        
        // Board ready state - prevents misclicks while board is loading
        this.isBoardReady = false;
        this.isProcessingMove = false;
        
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
        this.loadAutoAnalyzeSetting(); // Load auto-analyze preference
        this.loadActiveModelInfo();    // Load active model info for setup panel
        this.initAnimatedBackground(); // Initialize animated background
    }
    
    /**
     * Initialize multi-layered animated background elements for setup panel
     */
    initAnimatedBackground() {
        const bgContainer = document.querySelector('.animated-bg-container');
        if (!bgContainer) return;

        const setupPanel = document.getElementById('setupPanel');
        if (!setupPanel) return;

        // Define layers with different symbols and counts
        const layers = [
            {
                id: 'animatedBgLayer1',
                symbols: ['→', '↑', '↗'],
                count: 8,
                baseDelay: 0
            },
            {
                id: 'animatedBgLayer2',
                symbols: ['+', '/', '\\', '×'],
                count: 6,
                baseDelay: -5
            },
            {
                id: 'animatedBgLayer3',
                symbols: ['→', '↑', '↗', '+', '/', '\\', '×'],
                count: 7,
                baseDelay: -10
            }
        ];

        // Create elements for each layer
        layers.forEach(layer => {
            const layerElement = document.getElementById(layer.id);
            if (!layerElement) return;

            for (let i = 0; i < layer.count; i++) {
                const el = document.createElement('div');
                el.className = 'animated-bg-symbol';
                el.textContent = layer.symbols[Math.floor(Math.random() * layer.symbols.length)];
                el.style.left = `${Math.random() * 100}%`;
                el.style.top = `${Math.random() * 100}%`;
                el.style.animationDelay = `${layer.baseDelay + Math.random() * 10}s`;
                layerElement.appendChild(el);
            }
        });

        // Stop animation when game starts
        const observer = new MutationObserver((mutations) => {
            if (setupPanel.classList.contains('hidden')) {
                bgContainer.style.display = 'none';
                document.body.classList.add('game-active');
            } else {
                bgContainer.style.display = 'block';
                document.body.classList.remove('game-active');
            }
        });

        observer.observe(setupPanel, { attributes: true, attributeFilter: ['class'] });
    }

    /**
     * Load and display active model info in setup panel.
     */
    async loadActiveModelInfo() {
        try {
            const response = await fetch('/api/ai/model-info?level=5');
            if (!response.ok) return;
            
            const data = await response.json();
            const modelNameEl = document.getElementById('activeModelName');
            
            if (modelNameEl) {
                const typeIcon = data.type === 'alphazero' ? '🦾' : '🧠';
                const name = data.name || 'default (встроенная)';
                modelNameEl.textContent = `${typeIcon} ${name}`;
                
                // Add styling based on type
                modelNameEl.className = 'model-name ' + (data.type === 'alphazero' ? 'alphazero' : 'gym');
            }
        } catch (e) {
            console.error('Error loading active model info:', e);
            const modelNameEl = document.getElementById('activeModelName');
            if (modelNameEl) modelNameEl.textContent = '🧠 default';
        }
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
            selectConfidenceModel: document.getElementById('selectConfidenceModel'),
            
            // Move history
            moveList: document.getElementById('moveList'),
            
            // Analysis
            btnAnalyze: document.getElementById('btnAnalyze'),
            btnSuggest: document.getElementById('btnSuggest'),
            analysisContent: document.getElementById('analysisContent'),
            checkAutoAnalyze: document.getElementById('checkAutoAnalyze'),
            
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
            rightAvatar: document.getElementById('rightAvatar'),
            
            // Voice controls
            voiceToggle: document.getElementById('btnVoiceToggle'),
            voicePause: document.getElementById('btnVoicePause'),
            voiceRecord: document.getElementById('btnVoiceRecord'),
            voiceStatus: document.getElementById('voiceStatus')
        };
        
        // Initialize voice service
        this.initVoiceService();
    }
    
    /**
     * Initialize voice service with UI elements
     */
    initVoiceService() {
        if (typeof voiceService !== 'undefined') {
            voiceService.init({
                toggleBtn: this.elements.voiceToggle,
                pauseBtn: this.elements.voicePause,
                recordBtn: this.elements.voiceRecord,
                statusEl: this.elements.voiceStatus
            });
            
            // Set callback for voice input
            voiceService.setVoiceInputCallback((userText, context) => {
                this.handleVoiceInput(userText, context);
            });
        }
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
        
        this.elements.selectConfidenceModel?.addEventListener('change', (e) => {
            this.confidenceModel = e.target.value;
            localStorage.setItem('confidenceModel', this.confidenceModel);
            this.updateBoard(this.gameState); // Refresh probabilities immediately
        });
        
        // Analysis
        this.elements.btnAnalyze.addEventListener('click', () => this.analyzePosition());
        this.elements.btnSuggest.addEventListener('click', () => this.suggestMove());
        this.elements.checkAutoAnalyze?.addEventListener('change', (e) => {
            this.autoAnalyzeEnabled = e.target.checked;
            localStorage.setItem('autoAnalyzeEnabled', this.autoAnalyzeEnabled ? '1' : '0');
        });

        // User scroll detection for analysis
        this.elements.analysisContent?.addEventListener('scroll', () => {
            const el = this.elements.analysisContent;
            // If user scrolled up, set userScrolling to true
            if (el.scrollHeight - el.scrollTop - el.clientHeight > 20) {
                this.userScrolling = true;
            } else {
                this.userScrolling = false;
            }
        });
        
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
            
            const savedModel = localStorage.getItem('confidenceModel');
            if (savedModel) {
                this.confidenceModel = savedModel;
                if (this.elements.selectConfidenceModel) {
                    this.elements.selectConfidenceModel.value = savedModel;
                }
            }
        } catch (e) {
            // ignore
        }
        this.applyConfidenceSetting();
    }

    loadAutoAnalyzeSetting() {
        try {
            const saved = localStorage.getItem('autoAnalyzeEnabled');
            if (saved === '0' || saved === '1') {
                this.autoAnalyzeEnabled = saved === '1';
                if (this.elements.checkAutoAnalyze) {
                    this.elements.checkAutoAnalyze.checked = this.autoAnalyzeEnabled;
                }
            }
        } catch (e) {
            // ignore
        }
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
    /**
     * Start a new game.
     */
    async startGame() {
        let loader = null;
        try {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:330',message:'startGame entry',data:{playerColor:this.playerColor,aiLevel:this.aiLevel},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'4'})}).catch(()=>{});
            // #endregion

            this.elements.btnStartGame.disabled = true;
            this.elements.btnStartGame.textContent = 'Starting...';
            
            // Block board interactions while loading
            this.isBoardReady = false;
            this.isProcessingMove = false;
            
            // Show setup panel hiding and game panel showing
            this.elements.setupPanel.classList.add('hidden');
            this.elements.gamePanel.classList.remove('hidden');
            
            // Show mesmerizing loader over the board (after showing panel)
            // We store the reference to manually remove it if needed
            loader = this.showBoardLoader(4000);

            // RESTORED API CALL WITH CORRECT RESPONSE HANDLING
            const response = await api.createGame(this.playerColor, this.aiLevel);
            
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:345',message:'game created',data:{gameId:response?.game_id},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'4'})}).catch(()=>{});
            // #endregion

            if (!response || !response.game_id) {
                throw new Error('Server failed to create game. Check console logs.');
            }

            this.gameId = response.game_id;
            this.gameState = response;
            
            // Connect WebSocket
            api.connectWebSocket(this.gameId, (data) => this.handleWSMessage(data));
            
            // Store AI model info
            this.currentAiModel = response.ai_model || null;
            
            // Configure board
            this.classicBoard.setHumanColor(this.playerColor);
            this.loadConfidenceSetting();
            
            // Update player labels with model info
            this.updatePlayerLabels();
            
            // Update ELO displays
            this.elements.aiElo.textContent = `ELO: ${response.ai_elo || 1500}`;
            this.elements.playerElo.textContent = `ELO: ${response.player_elo || 1500}`;
            
            // Update score panel avatars
            this.updateScoreAvatars();
            
            // Clear move history
            this.elements.moveList.innerHTML = '';
            this.elements.analysisContent.innerHTML = '<p class="placeholder">Нажмите "Анализировать" для получения анализа от Gemini.</p>';
            
            // Render board
            await this.updateBoard(response);
            
            // Board is now ready for interaction
            this.isBoardReady = true;
            
        } catch (error) {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:390',message:'startGame error',data:{err:error.message,stack:error.stack},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'4'})}).catch(()=>{});
            // #endregion
            console.error('Failed to start game:', error);
            alert('Failed to start game: ' + error.message);
            
            // Manual loader removal on error
            if (loader) {
                loader.style.opacity = '0';
                setTimeout(() => loader.remove(), 500);
            }
            
            // Show setup panel again if game creation failed
            this.showSetup();
        } finally {
            this.elements.btnStartGame.disabled = false;
            this.elements.btnStartGame.textContent = '▶️ Start Game';
        }
    }

    /**
     * Handle WebSocket messages.
     */
    async handleWSMessage(data) {
        console.log('[WS] Message received:', data.type);
        
        switch (data.type) {
            case 'game_state':
                this.gameState = data.data;
                this.updateBoard(this.gameState);
                break;
                
            case 'move_made':
                this.gameState = data.data;
                this.isProcessingMove = false; // Reset processing flag
                this.updateBoard(this.gameState);
                this.triggerBoardGlow();
                await this.updateMoveHistory();
                break;
                
            case 'ai_thinking':
                this.showAIThinking();
                break;
                
            case 'ai_move':
                this.hideAIThinking();
                this.gameState = data.data;
                this.isProcessingMove = false; // Reset processing flag
                this.updateBoard(this.gameState);
                this.triggerBoardGlow();
                
                // Highlight AI move
                const aiPlayer = this.playerColor === 'white' ? 'black' : 'white';
                this.classicBoard.highlightLastMove(data.move, aiPlayer);
                
                await this.updateMoveHistory();
                
                // Automatic analysis if enabled
                if (this.autoAnalyzeEnabled && this.gameState.status === 'in_progress') {
                    this.analyzePosition();
                }
                break;
                
            case 'analysis_start':
                this.isStreaming = true;
                this.userScrolling = false;
                this.ttsBuffer = '';  // Buffer for progressive TTS
                this.ttsSentIndex = 0;  // Track what we've sent to TTS
                this.accumulatedText = ''; // Hidden buffer for raw text
                this.firstSentenceSent = false; // For faster first sentence detection
                
                // Stop current speech when new analysis starts
                if (typeof voiceService !== 'undefined') voiceService.stopPlayback();
                
                // Show mesmerizing preloader
                this.elements.analysisContent.innerHTML = `
                    <div class="analysis-result">
                        <div class="analysis-preloader">
                            ${this.getMesmerizingLoaderHtml('AI анализирует позицию...')}
                        </div>
                    </div>`;
                this.elements.btnAnalyze.disabled = true;
                break;
                
            case 'analysis_chunk':
                // Accumulate text silently
                this.accumulatedText += data.chunk;
                console.log(`[Frontend] Received analysis chunk: '${data.chunk}' (accumulated: ${this.accumulatedText.length} chars)`);
                // Progressive TTS still works in background
                this.processProgressiveTTS(data.chunk);
                break;
                
            case 'analysis_end':
                this.isStreaming = false;
                this.elements.btnAnalyze.disabled = false;

                console.log(`[Frontend] Analysis ended. Full text length: ${data.full_text?.length || 0}, accumulated: ${this.accumulatedText.length}`);

                // Now show the BEAUTIFULLY formatted text all at once
                const resEl = this.elements.analysisContent.querySelector('.analysis-result');
                const fullTxt = data.full_text || this.accumulatedText;
                console.log(`[Frontend] Final text to display: '${fullTxt.substring(0, 200)}...'`);
                if (resEl) {
                    resEl.innerHTML = this.formatAnalysis(fullTxt);
                }
                
                // Flush remaining buffer to TTS (progressive system handles the rest)
                this.flushTTSBuffer();
                break;
                
            case 'suggestion_start':
                this.isStreaming = true;
                this.userScrolling = false;
                this.ttsBuffer = '';  // Buffer for progressive TTS
                this.ttsSentIndex = 0;  // Track what we've sent to TTS
                this.accumulatedText = '';
                this.firstSentenceSent = false;
                
                // Stop current speech
                if (typeof voiceService !== 'undefined') voiceService.stopPlayback();
                
                this.elements.analysisContent.innerHTML = `
                    <div class="suggestion-result">
                        <div class="analysis-preloader">
                            ${this.getMesmerizingLoaderHtml('Ищу лучший ход...')}
                        </div>
                    </div>`;
                this.elements.btnSuggest.disabled = true;
                break;
                
            case 'suggestion_chunk':
                this.accumulatedText += data.chunk;
                // Progressive TTS: accumulate and send sentences
                this.processProgressiveTTS(data.chunk);
                break;
                
            case 'suggestion_end':
                this.isStreaming = false;
                this.elements.btnSuggest.disabled = false;
                
                const suggestEl = this.elements.analysisContent.querySelector('.suggestion-result');
                const fullText = data.full_text || this.accumulatedText;
                if (suggestEl) {
                    suggestEl.innerHTML = this.formatAnalysis(fullText);
                }
                
                // Flush remaining buffer to TTS (progressive system handles the rest)
                this.flushTTSBuffer();
                break;
                
            case 'game_over':
                this.handleGameOver(data);
                break;
                
            case 'error':
                console.error('[WS] Server error:', data.message);
                alert('Error: ' + data.message);
                break;
        }
    }

    /**
     * Append chunk to analysis content and scroll.
     */
    appendAnalysisChunk(chunk) {
        const resultEl = this.elements.analysisContent.querySelector('.analysis-result, .suggestion-result');
        if (!resultEl) return;
        
        // Remove existing cursor if any
        const cursor = resultEl.querySelector('.chunk-loading');
        if (cursor) cursor.remove();
        
        // Add chunk
        const textNode = document.createTextNode(chunk);
        resultEl.appendChild(textNode);
        
        // Re-add cursor
        const newCursor = document.createElement('span');
        newCursor.className = 'chunk-loading';
        resultEl.appendChild(newCursor);
        
        // Auto-scroll if user hasn't intervened
        if (!this.userScrolling) {
            this.elements.analysisContent.scrollTop = this.elements.analysisContent.scrollHeight;
        }
    }

    /**
     * Process progressive TTS - send sentences to TTS as they complete.
     */
    processProgressiveTTS(chunk) {
        if (typeof voiceService === 'undefined' || !voiceService.isVoiceEnabled) return;
        
        this.ttsBuffer = (this.ttsBuffer || '') + chunk;
        
        // Update UI if we have enough text but haven't switched from preloader yet
        // Show text as soon as we start speaking
        const shouldUpdateUI = !this.firstSentenceSent && this.ttsBuffer.length > 20;

        // Fast sentence detection:
        // Match . ! ? followed by space, newline, or even just end of string if enough text
        const sentenceEndings = /([.!?。！？])(\s+|\n+|$)/g;
        let match;
        
        // Very first sentence optimization: 
        // If we have > 30 chars and NO sentence yet, just take what we have
        if (!this.firstSentenceSent && this.ttsBuffer.length > 40) {
            const lastSpace = this.ttsBuffer.lastIndexOf(' ');
            if (lastSpace > 20) {
                const fragment = this.ttsBuffer.substring(0, lastSpace).trim();
                voiceService.queueSpeak(fragment);
                this.ttsSentIndex = lastSpace + 1;
                this.firstSentenceSent = true;
                this.updateAnalysisUIWithAccumulated();
                return;
            }
        }

        while ((match = sentenceEndings.exec(this.ttsBuffer)) !== null) {
            const sentenceEnd = match.index + match[1].length;
            
            if (sentenceEnd > this.ttsSentIndex) {
                const sentence = this.ttsBuffer.substring(this.ttsSentIndex, sentenceEnd).trim();
                
                if (sentence.length > 5) {
                    if (sentence.length > 15 || (sentence.length > 2 && !/\d$/.test(sentence))) {
                        voiceService.queueSpeak(sentence);
                        this.ttsSentIndex = match.index + match[0].length;
                        this.firstSentenceSent = true;
                        this.updateAnalysisUIWithAccumulated();
                    }
                }
            }
        }
        
        // Safety break for long texts
        const unsentLen = this.ttsBuffer.length - this.ttsSentIndex;
        if (unsentLen > 150) {
            const lastSpace = this.ttsBuffer.lastIndexOf(' ');
            if (lastSpace > this.ttsSentIndex + 40) {
                const fragment = this.ttsBuffer.substring(this.ttsSentIndex, lastSpace).trim();
                voiceService.queueSpeak(fragment);
                this.ttsSentIndex = lastSpace + 1;
                this.firstSentenceSent = true;
                this.updateAnalysisUIWithAccumulated();
            }
        }
    }

    /**
     * Update Analysis UI with current accumulated text (partial).
     */
    updateAnalysisUIWithAccumulated() {
        if (!this.accumulatedText) return;
        
        const resultEl = this.elements.analysisContent.querySelector('.analysis-result') || 
                         this.elements.analysisContent.querySelector('.suggestion-result');
        
        if (resultEl) {
            // Remove preloader if it exists
            const preloader = resultEl.querySelector('.analysis-preloader');
            if (preloader) preloader.remove();
            
            // Format and set content
            resultEl.innerHTML = this.formatAnalysis(this.accumulatedText);
            
            // Add a temporary cursor to show it's still streaming
            const cursor = document.createElement('span');
            cursor.className = 'chunk-loading';
            resultEl.appendChild(cursor);
            
            // Auto-scroll
            if (!this.userScrolling) {
                this.elements.analysisContent.scrollTop = this.elements.analysisContent.scrollHeight;
            }
        }
    }

    /**
     * Flush remaining TTS buffer at end of stream.
     */
    flushTTSBuffer() {
        if (typeof voiceService === 'undefined' || !voiceService.isVoiceEnabled) return;
        
        // Send any remaining text that wasn't a complete sentence
        const remaining = (this.ttsBuffer || '').substring(this.ttsSentIndex || 0).trim();
        if (remaining.length > 2) {
            voiceService.queueSpeak(remaining);
        }
        
        // Reset
        this.ttsBuffer = '';
        this.ttsSentIndex = 0;
    }

    /**
     * Make a move.
     */
    async makeMove(move) {
        if (!this.gameId) return;
        
        // Stop speech when making a new move
        if (typeof voiceService !== 'undefined') voiceService.stopPlayback();
        
        // Block moves if board is not ready (still loading)
        if (!this.isBoardReady) {
            console.warn('[App] Ignoring move: board not ready yet');
            return;
        }
        
        // Block if already processing a move
        if (this.isProcessingMove) {
            console.warn('[App] Ignoring move: already processing a move');
            return;
        }
        
        // Block moves if not our turn or game is finished
        if (!this.isMyTurn || this.gameState?.status !== 'in_progress') {
            console.warn('[App] Ignoring move: not our turn or game finished');
            return;
        }
        
        // Set processing flag to prevent double clicks
        this.isProcessingMove = true;
        this.isMyTurn = false;
        
        // Use WebSocket for move
        api.makeMoveWS(move);
        
        // Reset processing flag after a short delay (will be properly reset on WS response)
        setTimeout(() => {
            this.isProcessingMove = false;
        }, 500);
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
                // Clear old ones first
                this.classicBoard.clearProbabilities();
                
                let modelsToFetch = [];
                if (this.confidenceModel === 'all') {
                    modelsToFetch = ['polynet', 'alphazero', 'probs'];
                } else {
                    modelsToFetch = [this.confidenceModel];
                }

                const requests = modelsToFetch.map(m => api.getMoveProbabilities(this.aiLevel, m));
                const results = await Promise.all(requests);

                results.forEach((res, idx) => {
                    if (res && res.probabilities) {
                        const modelKey = modelsToFetch[idx] === 'auto' ? 'polynet' : modelsToFetch[idx];
                        this.classicBoard.setProbabilities(modelKey, res.probabilities);
                    }
                });
            } catch (e) {
                console.warn('Could not fetch multi-probabilities:', e);
            }
        } else {
            this.classicBoard.clearProbabilities();
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
     * Update player labels in info bar based on player color and AI model.
     */
    updatePlayerLabels() {
        const playerBlackLabel = document.querySelector('.player-black .player-label');
        const playerWhiteLabel = document.querySelector('.player-white .player-label');
        
        if (!playerBlackLabel || !playerWhiteLabel) return;
        
        // Get AI model display name
        let aiName = 'AI';
        if (this.aiLevel === 6) {
            aiName = 'Gemini';
        } else if (this.aiLevel === 7) {
            aiName = 'PROBS';
        } else if (this.currentAiModel && this.currentAiModel.name) {
            aiName = this.currentAiModel.name;
        } else {
            aiName = `AI L${this.aiLevel}`;
        }
        
        if (this.playerColor === 'white') {
            playerWhiteLabel.textContent = '⚪ АҚ (Вы)';
            playerBlackLabel.textContent = `⚫ ҚАРА (${aiName})`;
        } else {
            playerBlackLabel.textContent = '⚫ ҚАРА (Вы)';
            playerWhiteLabel.textContent = `⚪ АҚ (${aiName})`;
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
        
        // Reset scroll position and state
        this.userScrolling = false;
        this.elements.analysisContent.innerHTML = `
            <div class="analysis-result">
                <div class="analysis-preloader">
                    ${this.getMesmerizingLoaderHtml('Анализирую позицию...')}
                </div>
            </div>`;
        this.elements.btnAnalyze.disabled = true;
        
        // Use WebSocket for streaming analysis
        api.requestAnalysisWS();
    }

    /**
     * Get move suggestion.
     */
    async suggestMove() {
        if (!this.gameId) return;
        
        // Reset scroll position and state
        this.userScrolling = false;
        this.elements.analysisContent.innerHTML = `
            <div class="suggestion-result">
                <div class="analysis-preloader">
                    ${this.getMesmerizingLoaderHtml('Ищу лучший ход...')}
                </div>
            </div>`;
        this.elements.btnSuggest.disabled = true;
        
        // Use WebSocket for streaming suggestion
        api.requestSuggestionWS();
    }

    /**
     * Format analysis text for display.
     */
    formatAnalysis(text) {
        if (!text) return '';

        // IMPORTANT: Escape HTML from Gemini output to avoid accidental tag interpretation
        // (which can visually "truncate" the output if Gemini returns "<...>").
        const escaped = this.escapeHtml(text);

        // Split into paragraphs first
        const paragraphs = escaped.split(/\n\s*\n/).filter(p => p.trim());

        return paragraphs.map(paragraph => {
            // Process each paragraph individually
            const processed = paragraph
                // Handle markdown ## headings first (new format)
                .replace(/^## (📊.*?)$/gm, '<h4 class="analysis-heading">$1</h4>')
                .replace(/^## (🔍.*?)$/gm, '<h4 class="analysis-heading">$1</h4>')
                .replace(/^## (🎯.*?)$/gm, '<h4 class="analysis-heading highlight">$1</h4>')
                .replace(/^## (⚠️.*?)$/gm, '<h4 class="analysis-heading warning">$1</h4>')
                .replace(/^## (💡.*?)$/gm, '<h4 class="analysis-heading">$1</h4>')
                .replace(/^## (📋.*?)$/gm, '<h4 class="analysis-heading">$1</h4>')
                // Newlines to breaks within paragraph
                .replace(/\n/g, '<br>');

            // Wrap each paragraph in a styled container
            return `<div class="analysis-paragraph">${processed}</div>`;
        }).join('')
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Legacy format support
            .replace(/EVALUATION:/g, '<strong>📊 ОЦЕНКА:</strong>')
            .replace(/ОЦЕНКА:/g, '<strong>📊 ОЦЕНКА:</strong>')
            .replace(/BEST MOVE:/g, '<strong>🎯 РЕКОМЕНДУЕМЫЙ ХОД:</strong>')
            .replace(/ЛУЧШИЙ ХОД:/g, '<strong>🎯 РЕКОМЕНДУЕМЫЙ ХОД:</strong>')
            .replace(/РЕКОМЕНДУЕМЫЙ ХОД:/g, '<strong>🎯 РЕКОМЕНДУЕМЫЙ ХОД:</strong>')
            .replace(/ОБОСНОВАНИЕ:/g, '<strong>💡 ОБОСНОВАНИЕ:</strong>')
            .replace(/ПОЧЕМУ ЭТОТ ХОД\?/g, '<strong>💡 ПОЧЕМУ ЭТОТ ХОД?</strong>')
            .replace(/KEY FACTORS:/g, '<strong>🔑 КЛЮЧЕВЫЕ ФАКТОРЫ:</strong>')
            .replace(/КЛЮЧЕВЫЕ ФАКТОРЫ:/g, '<strong>🔑 КЛЮЧЕВЫЕ ФАКТОРЫ:</strong>')
            .replace(/STRATEGIC ANALYSIS:/g, '<strong>🔍 СТРАТЕГИЧЕСКИЙ РАЗБОР:</strong>')
            .replace(/СТРАТЕГИЧЕСКИЙ АНАЛИЗ:/g, '<strong>🔍 СТРАТЕГИЧЕСКИЙ РАЗБОР:</strong>')
            .replace(/СТРАТЕГИЧЕСКИЙ РАЗБОР:/g, '<strong>🔍 СТРАТЕГИЧЕСКИЙ РАЗБОР:</strong>')
            .replace(/WARNING:/g, '<strong>⚠️ УГРОЗЫ:</strong>')
            .replace(/УГРОЗЫ:/g, '<strong>⚠️ УГРОЗЫ:</strong>')
            .replace(/УГРОЗЫ И ВОЗМОЖНОСТИ:/g, '<strong>⚠️ УГРОЗЫ И ВОЗМОЖНОСТИ:</strong>')
            .replace(/АНАЛИЗ ХОДОВ:/g, '<strong>📝 АНАЛИЗ ХОДОВ:</strong>')
            .replace(/АЛЬТЕРНАТИВЫ:/g, '<strong>📋 АЛЬТЕРНАТИВЫ:</strong>');
    }

    escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Handle voice input from user - send to AI with context
     */
    async handleVoiceInput(userText, lastAnalysisContext) {
        if (!userText || !this.gameId) return;
        
        console.log('[Voice] User asked:', userText);
        
        // Show user's question in analysis panel
        const userQuestionHtml = `
            <div class="voice-question">
                <span class="voice-icon">🎤</span>
                <span class="voice-text">"${this.escapeHtml(userText)}"</span>
            </div>
            <div class="analysis-result">
                <div class="analysis-preloader">
                    ${this.getMesmerizingLoaderHtml('AI обдумывает вопрос...')}
                </div>
            </div>
        `;
        this.elements.analysisContent.innerHTML = userQuestionHtml;
        
        // Send voice query via WebSocket with context
        if (api.ws && api.ws.readyState === WebSocket.OPEN) {
            api.ws.send(JSON.stringify({
                type: 'voice_query',
                query: userText,
                context: lastAnalysisContext || '',
                game_id: this.gameId
            }));
        }
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
     * Show mesmerizing loader over the board.
     */
    showBoardLoader(duration = 4000) {
        const boardWrapper = document.querySelector('.board-wrapper.classic-active');
        
        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:100',message:'showBoardLoader entry',data:{duration,found:!!boardWrapper},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
        // #endregion

        if (!boardWrapper) return null;

        // Wishes pool
        const wishes = [
            "Удачи в великой игре!",
            "Пусть каждый ход будет мудрым!",
            "Тоғыз құмалақ — игра смелых и терпеливых!",
            "Найди путь к победе!",
            "Сокруши ИИ своим интеллектом!",
            "Каждая лунка хранит секрет победы!",
            "Время показать мастерство!",
            "Пусть удача сопутствует тебе!",
            "Стратегия — твой главный союзник!",
            "Будь внимателен к каждому кумалаку!",
            "Твоя интуиция тебя не подведет!",
            "Разгадай замыслы противника!",
            "Стань легендой Тоғыз құмалақ!",
            "Победа куется в каждом ходе!",
            "Играй сердцем, побеждай разумом!",
            "Пусть твой казан всегда будет полон!",
            "Тактика — это искусство побеждать!",
            "Твой интеллект сильнее любого алгоритма!",
            "Шаг за шагом к триумфу!",
            "Наслаждайся красотой древней игры!"
        ];
        
        const randomWish = wishes[Math.floor(Math.random() * wishes.length)];

        const loader = document.createElement('div');
        loader.className = 'mesmerizing-loader';
        loader.innerHTML = this.getMesmerizingLoaderHtml(randomWish);
        boardWrapper.appendChild(loader);

        setTimeout(() => {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:140',message:'hiding board loader',data:{},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion
            loader.style.opacity = '0';
            setTimeout(() => {
                if (loader.parentNode) loader.remove();
            }, 500);
        }, duration);

        return loader; // Return the element
    }

    /**
     * Get HTML for mesmerizing geometric loader.
     */
    getMesmerizingLoaderHtml(text) {
        return `
            <div class="loader-geometry">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <div class="loader-text">${text}</div>
        `;
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
    
    // Initialize theme switcher
    initThemeSwitcher();
});

/**
 * Theme Switcher functionality
 */
function initThemeSwitcher() {
    const themeSwitcher = document.getElementById('themeSwitcher');
    const themeLabel = document.getElementById('themeLabel');
    
    const themes = [
        { id: 'default', name: 'Киберпанк' },
        { id: 'purple', name: 'Пурпур' },
        { id: 'coffee', name: 'Кофе' },
        { id: 'green', name: 'Лесная' }
    ];
    
    // Load saved theme
    let currentThemeIndex = 0;
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        const idx = themes.findIndex(t => t.id === savedTheme);
        if (idx !== -1) {
            currentThemeIndex = idx;
            applyTheme(themes[currentThemeIndex]);
        }
    }
    updateLabel();
    
    function applyTheme(theme) {
        if (theme.id === 'default') {
            document.documentElement.removeAttribute('data-theme');
        } else {
            document.documentElement.setAttribute('data-theme', theme.id);
        }
        localStorage.setItem('theme', theme.id);
        
        // Update board colors if initialized
        if (window.app && window.app.classicBoard) {
            window.app.classicBoard.updateColors();
        }
    }
    
    function updateLabel() {
        if (themeLabel) {
            themeLabel.textContent = themes[currentThemeIndex].name;
        }
    }
    
    if (themeSwitcher) {
        themeSwitcher.addEventListener('click', () => {
            currentThemeIndex = (currentThemeIndex + 1) % themes.length;
            applyTheme(themes[currentThemeIndex]);
            updateLabel();
        });
    }
}

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
    // Exclude game-panel and setup-panel from having expand button
    const panels = document.querySelectorAll('.panel:not(.game-panel):not(.setup-panel)');
    
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
