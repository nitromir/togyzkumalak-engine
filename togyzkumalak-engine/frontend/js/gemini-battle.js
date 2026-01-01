/**
 * VS Arena Module
 * 
 * Handles the UI for Model vs Gemini AND Model vs Model battles.
 * Features:
 * - Start/stop battle sessions
 * - Real-time progress tracking
 * - ELO chart visualization
 * - Game summaries display
 * - Model vs Local Model mode
 */

(function() {
    'use strict';
    
    // =========================================================================
    // State
    // =========================================================================
    
    let currentSessionId = null;
    let pollInterval = null;
    let eloChart = null;
    let arenaMode = 'vs-gemini';  // 'vs-gemini' or 'vs-local'
    
    // =========================================================================
    // DOM Elements
    // =========================================================================
    
    const elements = {
        // Config inputs
        numGames: document.getElementById('battleNumGames'),
        modelLevel: document.getElementById('battleModelLevel'),
        timeout: document.getElementById('battleTimeout'),
        saveReplays: document.getElementById('battleSaveReplays'),
        generateSummaries: document.getElementById('battleGenerateSummaries'),
        
        // Arena mode
        arenaModeButtons: document.querySelectorAll('.arena-mode-btn'),
        model2Section: document.getElementById('model2Section'),
        model2Select: document.getElementById('battleModel2Select'),
        vsGeminiOptions: document.querySelectorAll('.vs-gemini-option'),
        vsLocalOptions: document.querySelectorAll('.vs-local-option'),
        
        // Buttons
        btnStart: document.getElementById('btnStartBattle'),
        btnStop: document.getElementById('btnStopBattle'),
        btnExport: document.getElementById('btnExportGeminiData'),
        
        // Progress
        status: document.getElementById('battleStatus'),
        progressContainer: document.getElementById('battleProgress'),
        progressBar: document.getElementById('battleProgressBar'),
        progressText: document.getElementById('battleProgressText'),
        
        // Stats
        statsContainer: document.getElementById('battleStats'),
        modelElo: document.getElementById('modelElo'),
        modelCategory: document.getElementById('modelCategory'),
        modelWins: document.getElementById('battleModelWins'),
        geminiWins: document.getElementById('battleGeminiWins'),
        draws: document.getElementById('battleDraws'),
        
        // Chart
        chartCanvas: document.getElementById('eloChart'),
        
        // Summary
        lastSummary: document.getElementById('lastGameSummary'),
        
        // Active Model Selection
        modelSelect: document.getElementById('battleModelSelect'),
        
        // Sessions list
        sessionsList: document.getElementById('battleSessionsList')
    };

    // =========================================================================
    // API Functions
    // =========================================================================
    
    async function loadAvailableModels() {
        try {
            // Load AlphaZero checkpoints
            const resAz = await fetch('/api/training/alphazero/checkpoints');
            const dataAz = await resAz.json();
            
            // Load PROBS checkpoints
            const resProbs = await fetch('/api/training/probs/checkpoints');
            const dataProbs = await resProbs.json();
            
            const populateGroups = (selectEl) => {
                if (!selectEl) return;
                
                const currentVal = selectEl.value;
                
                const probsList = selectEl.querySelector('#arenaProbsList1, #arenaProbsList2');
                const azList = selectEl.querySelector('#arenaAzList1, #arenaAzList2');
                
                // We need to find them by id inside the specific select
                const myProbsGroup = selectEl.querySelector('optgroup[label*="PROBS"]');
                const myAzGroup = selectEl.querySelector('optgroup[label*="AlphaZero"]');

                if (myProbsGroup) {
                    myProbsGroup.innerHTML = (dataProbs.checkpoints || []).map(cp => 
                        `<option value="probs:${cp.filename}">${cp.is_best ? '⭐ ' : ''}${cp.filename}</option>`
                    ).join('');
                }
                
                if (myAzGroup) {
                    myAzGroup.innerHTML = (dataAz.checkpoints || []).map(cp => 
                        `<option value="az:${cp.name}">${cp.name}</option>`
                    ).join('');
                }
                
                if (currentVal) selectEl.value = currentVal;
            };
            
            populateGroups(elements.modelSelect);
            populateGroups(elements.model2Select);
            
        } catch (e) {
            console.error('Error loading models for VS Arena:', e);
        }
    }

    async function handleModelChange() {
        // We don't necessarily want to load the model into AIEngine immediately 
        // when changing in Arena setup, but we could.
        // For now, let's just log it.
        console.log(`[Arena] Model selection changed: ${elements.modelSelect.value} vs ${elements.model2Select.value}`);
    }
    
    async function startBattle(config) {
        const response = await fetch('/api/gemini-battle/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        return response.json();
    }
    
    async function stopBattle(sessionId) {
        const response = await fetch(`/api/gemini-battle/sessions/${sessionId}/stop`, {
            method: 'POST'
        });
        return response.json();
    }
    
    async function getSessionProgress(sessionId) {
        const response = await fetch(`/api/gemini-battle/sessions/${sessionId}`);
        return response.json();
    }
    
    async function getEloChartData(sessionId) {
        const response = await fetch(`/api/gemini-battle/sessions/${sessionId}/elo-chart`);
        return response.json();
    }
    
    async function listSessions() {
        const response = await fetch('/api/gemini-battle/sessions');
        return response.json();
    }
    
    async function updateActiveModelInfo() {
        try {
            const response = await fetch('/api/training/models/active');
            if (response.ok) {
                const data = await response.json();
                if (elements.modelSelect && data.model) {
                    // Check if the model exists in options, if not, reload list
                    const exists = Array.from(elements.modelSelect.options).some(o => o.value === data.model);
                    if (!exists && data.model !== 'default' && data.model !== 'policy_level_5') {
                        await loadAvailableModels();
                    }
                    
                    if (data.model === 'policy_level_5' || data.model === 'default') {
                        elements.modelSelect.value = 'default';
                    } else {
                        elements.modelSelect.value = data.model;
                    }
                }
            }
        } catch (e) {
            console.error('Error fetching active model:', e);
        }
    }
    
    // Expose for external use
    window.updateGeminiActiveModel = () => {
        loadAvailableModels().then(() => updateActiveModelInfo());
    };
    
    // =========================================================================
    // ELO Chart Drawing
    // =========================================================================
    
    function drawEloChart(data) {
        const canvas = elements.chartCanvas;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = { top: 20, right: 20, bottom: 40, left: 60 };
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Background
        ctx.fillStyle = 'rgba(20, 25, 35, 0.95)';
        ctx.fillRect(0, 0, width, height);
        
        if (!data || !data.values || data.values.length < 2) {
            ctx.fillStyle = '#6b7280';
            ctx.font = '14px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Данные графика появятся после первой игры...', width / 2, height / 2);
            return;
        }
        
        const { labels, values, thresholds, min_elo, max_elo } = data;
        
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        // Scale functions
        const xScale = (i) => padding.left + (i / Math.max(1, values.length - 1)) * chartWidth;
        const yScale = (v) => {
            const range = max_elo - min_elo;
            if (range === 0) return padding.top + chartHeight / 2;
            return padding.top + chartHeight - ((v - min_elo) / range) * chartHeight;
        };
        
        // Draw threshold lines
        if (thresholds && Array.isArray(thresholds)) {
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 1;
            thresholds.forEach(t => {
                if (t.value >= min_elo && t.value <= max_elo) {
                    const y = yScale(t.value);
                    
                    // Line color based on level
                    if (t.value >= 2400) ctx.strokeStyle = '#fbbf24'; // GM
                    else if (t.value >= 2200) ctx.strokeStyle = '#f59e0b'; // IM
                    else if (t.value >= 2000) ctx.strokeStyle = '#f97316'; // Master
                    else if (t.value >= 1800) ctx.strokeStyle = '#ef4444'; // KMS
                    else if (t.value >= 1400) ctx.strokeStyle = '#10b981'; // Club
                    else ctx.strokeStyle = '#6b7280';
                    
                    ctx.beginPath();
                    ctx.moveTo(padding.left, y);
                    ctx.lineTo(width - padding.right, y);
                    ctx.stroke();
                    
                    // Label
                    ctx.fillStyle = ctx.strokeStyle;
                    ctx.font = '10px "JetBrains Mono", monospace';
                    ctx.textAlign = 'right';
                    ctx.fillText(t.label, width - padding.right - 5, y - 3);
                }
            });
        }
        
        ctx.setLineDash([]);
        
        // Draw axes
        ctx.strokeStyle = '#4b5563';
        ctx.lineWidth = 1;
        
        // Y axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();
        
        // X axis
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();
        
        // Y axis labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        
        const ySteps = 5;
        for (let i = 0; i <= ySteps; i++) {
            const val = min_elo + (max_elo - min_elo) * (i / ySteps);
            const y = yScale(val);
            ctx.fillText(Math.round(val).toString(), padding.left - 8, y + 4);
        }
        
        // X axis labels
        ctx.textAlign = 'center';
        const xLabelStep = Math.max(1, Math.floor(labels.length / 10));
        for (let i = 0; i < labels.length; i += xLabelStep) {
            ctx.fillText(labels[i], xScale(i), height - padding.bottom + 20);
        }
        
        // Draw ELO line
        ctx.beginPath();
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 2;
        
        values.forEach((v, i) => {
            const x = xScale(i);
            const y = yScale(v);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        // Draw points
        values.forEach((v, i) => {
            const x = xScale(i);
            const y = yScale(v);
            
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#06b6d4';
            ctx.fill();
            ctx.strokeStyle = '#0e7490';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
        
        // Draw current ELO value
        if (values.length > 0) {
            const lastElo = values[values.length - 1];
            ctx.fillStyle = '#06b6d4';
            ctx.font = 'bold 16px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`ELO: ${lastElo}`, padding.left + 10, padding.top + 20);
        }
    }
    
    // =========================================================================
    // UI Update Functions
    // =========================================================================
    
    function updateProgress(data) {
        // Update status text
        let statusText = '';
        let statusClass = '';
        
        switch (data.status) {
            case 'pending':
                statusText = 'Подготовка...';
                statusClass = 'pending';
                break;
            case 'running':
                statusText = `Игра ${data.games_played + 1} из ${data.total_games}`;
                statusClass = 'running';
                break;
            case 'completed':
                statusText = 'Батл завершён!';
                statusClass = 'completed';
                break;
            case 'stopped':
                statusText = 'Батл остановлен';
                statusClass = 'stopped';
                break;
            case 'error':
                statusText = `Ошибка: ${data.error_message || 'Неизвестная ошибка'}`;
                statusClass = 'error';
                break;
        }
        
        // Update matchup display
        const p1Name = (data.player1_type || 'Player 1').replace('az:', 'AZ_').replace('probs:', 'PROBS_');
        const p2Name = (data.player2_type || 'Player 2').replace('az:', 'AZ_').replace('probs:', 'PROBS_');
        const matchupEl = document.getElementById('arenaMatchupDisplay');
        if (matchupEl) matchupEl.textContent = `${p1Name} vs ${p2Name}`;

        const sidesEl = document.getElementById('currentGameSides');
        if (sidesEl && data.status === 'running') {
            const p1Color = data.player1_color || 'white';
            const p2Color = p1Color === 'white' ? 'black' : 'white';
            sidesEl.innerHTML = `<span style="color:white">⚪ ${p1Color === 'white' ? p1Name : p2Name}</span> vs <span style="color:#aaa">⚫ ${p1Color === 'black' ? p1Name : p2Name}</span>`;
        }

        elements.status.innerHTML = `<p class="status-text ${statusClass}">${statusText}</p>`;
        
        // Update progress bar
        elements.progressContainer.classList.remove('hidden');
        elements.progressBar.style.width = `${data.progress_percent}%`;
        elements.progressText.textContent = `${data.games_played} / ${data.total_games} игр`;
        
        // Update stats
        elements.statsContainer.classList.remove('hidden');
        
        // Update labels based on player types
        const l1 = document.getElementById('labelP1Wins');
        const l2 = document.getElementById('labelP2Wins');
        if (l1) l1.textContent = `Побед ${p1Name.substring(0,12)}:`;
        if (l2) l2.textContent = `Побед ${p2Name.substring(0,12)}:`;

        elements.modelElo.textContent = data.model_elo;
        elements.modelCategory.textContent = `(${data.model_category})`;
        elements.modelWins.textContent = data.model_wins;
        elements.geminiWins.textContent = data.gemini_wins;
        elements.draws.textContent = data.draws;
        
        // Update ELO color based on change
        if (data.elo_history && data.elo_history.length > 1) {
            const lastChange = data.elo_history[data.elo_history.length - 1].change;
            if (lastChange > 0) {
                elements.modelElo.classList.add('elo-up');
                elements.modelElo.classList.remove('elo-down');
            } else if (lastChange < 0) {
                elements.modelElo.classList.add('elo-down');
                elements.modelElo.classList.remove('elo-up');
            }
        }
        
        // Update summary
        if (data.last_game_summary) {
            elements.lastSummary.innerHTML = `<p class="summary-text">${data.last_game_summary.replace(/\n/g, '<br>')}</p>`;
        }
        
        // Show/hide stop button
        if (data.status === 'running') {
            elements.btnStop.classList.remove('hidden');
            elements.btnStart.disabled = true;
        } else {
            elements.btnStop.classList.add('hidden');
            elements.btnStart.disabled = false;
        }
    }
    
    async function updateChart(sessionId) {
        try {
            const chartData = await getEloChartData(sessionId);
            drawEloChart(chartData);
        } catch (e) {
            console.error('Failed to update chart:', e);
        }
    }
    
    async function loadSessionsList() {
        try {
            const data = await listSessions();
            const sessions = data.sessions || [];
            
            if (sessions.length === 0) {
                elements.sessionsList.innerHTML = '<p class="empty-text">Нет завершённых батлов</p>';
                return;
            }
            
            let html = '';
            sessions.forEach(session => {
                const statusIcon = {
                    'completed': '✅',
                    'running': '🔄',
                    'stopped': '⏹️',
                    'error': '❌',
                    'pending': '⏳'
                }[session.status] || '❓';
                
                const winRate = session.games_played > 0 
                    ? ((session.model_wins / session.games_played) * 100).toFixed(1)
                    : 0;
                
                html += `
                    <div class="session-item" data-session="${session.session_id}">
                        <div class="session-header">
                            <span class="session-status">${statusIcon}</span>
                            <span class="session-id">#${session.session_id}</span>
                            <span class="session-date">${formatDate(session.started_at)}</span>
                        </div>
                        <div class="session-stats">
                            <span>Игр: ${session.games_played}/${session.total_games}</span>
                            <span>М:${session.model_wins} G:${session.gemini_wins} Н:${session.draws}</span>
                            <span>ELO: ${session.final_elo}</span>
                            <span>WR: ${winRate}%</span>
                        </div>
                    </div>
                `;
            });
            
            elements.sessionsList.innerHTML = html;
            
            // Add click handlers
            elements.sessionsList.querySelectorAll('.session-item').forEach(item => {
                item.addEventListener('click', () => {
                    const sessionId = item.dataset.session;
                    loadSession(sessionId);
                });
            });
            
        } catch (e) {
            console.error('Failed to load sessions:', e);
            elements.sessionsList.innerHTML = '<p class="empty-text error">Ошибка загрузки</p>';
        }
    }
    
    async function loadSession(sessionId) {
        try {
            currentSessionId = sessionId;
            const data = await getSessionProgress(sessionId);
            updateProgress(data);
            await updateChart(sessionId);
            
            // If session is running, start polling
            if (data.status === 'running') {
                startPolling(sessionId);
            }
        } catch (e) {
            console.error('Failed to load session:', e);
        }
    }
    
    // =========================================================================
    // Polling
    // =========================================================================
    
    function startPolling(sessionId) {
        stopPolling();
        
        pollInterval = setInterval(async () => {
            try {
                const data = await getSessionProgress(sessionId);
                updateProgress(data);
                await updateChart(sessionId);
                
                // Stop polling if session ended
                if (data.status !== 'running' && data.status !== 'pending') {
                    stopPolling();
                    loadSessionsList();
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
        }, 2000); // Poll every 2 seconds
    }
    
    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }
    
    // =========================================================================
    // Event Handlers
    // =========================================================================
    
    async function handleStartBattle() {
        const p1 = elements.modelSelect.value;
        const p2 = elements.model2Select.value;
        const numGames = parseInt(elements.numGames.value) || 10;
        
        const config = {
            player1: p1,
            player2: p2,
            num_games: numGames,
            gemini_timeout: parseInt(elements.timeout.value) || 30,
            save_replays: elements.saveReplays.checked
        };
        
        try {
            elements.btnStart.disabled = true;
            elements.btnStart.textContent = '⏳ Запуск...';
            
            const result = await startBattle(config);
            
            if (result.session_id) {
                currentSessionId = result.session_id;
                
                // Reset UI
                elements.lastSummary.innerHTML = '<p class="placeholder">Резюме появится после завершения игры...</p>';
                
                // Start polling
                startPolling(result.session_id);
                
                // Show initial state
                updateProgress({
                    status: 'pending',
                    games_played: 0,
                    total_games: config.num_games,
                    progress_percent: 0,
                    model_elo: 1500,
                    model_category: 'Клубный игрок',
                    model_wins: 0,
                    gemini_wins: 0,
                    draws: 0
                });
                
                // Clear chart
                drawEloChart({ labels: ['0'], values: [1500], thresholds: [], min_elo: 1400, max_elo: 1600 });
            }
        } catch (e) {
            console.error('Failed to start battle:', e);
            alert('Ошибка запуска батла: ' + e.message);
        } finally {
            elements.btnStart.disabled = false;
            elements.btnStart.textContent = '⚔️ Начать Батл';
        }
    }
    
    async function handleStopBattle() {
        if (!currentSessionId) return;
        
        try {
            await stopBattle(currentSessionId);
            elements.btnStop.classList.add('hidden');
            elements.btnStart.disabled = false;
        } catch (e) {
            console.error('Failed to stop battle:', e);
        }
    }
    
    async function handleExportData() {
        const btn = elements.btnExport;
        if (!btn) return;
        
        btn.disabled = true;
        btn.textContent = '⏳ Экспорт...';
        
        try {
            const response = await fetch('/api/gemini-battle/export-training-data', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Export failed');
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                alert(`✅ Экспорт завершён!\n\nИгр обработано: ${result.games_exported}\nПереходов создано: ${result.transitions_created}\n\nДанные сохранены в: gemini_transitions.jsonl\n\nТеперь можете использовать их для дообучения!`);
            } else if (result.status === 'no_data') {
                alert('⚠️ Нет данных для экспорта.\n\nСначала сыграйте несколько игр против Gemini!');
            }
            
        } catch (e) {
            console.error('Export error:', e);
            alert('❌ Ошибка экспорта: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = '📤 Экспорт для обучения';
        }
    }
    
    // =========================================================================
    // Utility Functions
    // =========================================================================
    
    function formatDate(isoString) {
        if (!isoString) return '-';
        try {
            const date = new Date(isoString);
            return date.toLocaleDateString('ru-RU', {
                day: '2-digit',
                month: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return isoString;
        }
    }
    
    // =========================================================================
    // Initialization
    // =========================================================================
    
    /**
     * Switch arena mode between vs-gemini and vs-local
     */
    function switchArenaMode(mode) {
        arenaMode = mode;
        
        // Update buttons
        elements.arenaModeButtons.forEach(btn => {
            if (btn.dataset.mode === mode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        // Show/hide relevant options
        if (mode === 'vs-gemini') {
            elements.vsGeminiOptions.forEach(el => el.classList.remove('hidden'));
            elements.vsLocalOptions.forEach(el => el.classList.add('hidden'));
            if (elements.model2Section) elements.model2Section.classList.add('hidden');
        } else {
            elements.vsGeminiOptions.forEach(el => el.classList.add('hidden'));
            elements.vsLocalOptions.forEach(el => el.classList.remove('hidden'));
            if (elements.model2Section) elements.model2Section.classList.remove('hidden');
        }
        
        console.log(`[VS Arena] Switched to mode: ${mode}`);
    }
    
    /**
     * Populate model 2 dropdown
     */
    async function loadModel2Options() {
        if (!elements.model2Select) return;
        
        try {
            const response = await fetch('/api/training/models');
            if (response.ok) {
                const data = await response.json();
                elements.model2Select.innerHTML = '<option value="default">🌐 Стандартная (Level 5)</option>';
                
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    const typeIcon = model.type === 'alphazero' ? '🦾' : '🧠';
                    option.textContent = `${typeIcon} ${model.name}`;
                    elements.model2Select.appendChild(option);
                });
            }
        } catch (e) {
            console.error('Error loading model 2 options:', e);
        }
    }

    function init() {
        // Check if elements exist
        if (!elements.btnStart) {
            console.warn('VS Arena elements not found');
            return;
        }
        
        // Event listeners
        elements.btnStart.addEventListener('click', handleStartBattle);
        elements.btnStop.addEventListener('click', handleStopBattle);
        elements.btnExport?.addEventListener('click', handleExportData);
        elements.modelSelect?.addEventListener('change', handleModelChange);
        
        // Arena mode buttons
        elements.arenaModeButtons.forEach(btn => {
            btn.addEventListener('click', () => switchArenaMode(btn.dataset.mode));
        });
        
        // Load existing sessions when tab is shown
        const geminiBattleTab = document.querySelector('[data-mode="gemini-battle"]');
        if (geminiBattleTab) {
            geminiBattleTab.addEventListener('click', () => {
                loadSessionsList();
                loadAvailableModels().then(() => updateActiveModelInfo());
                
                // Draw empty chart initially
                drawEloChart(null);
            });
        }
        
        // Initial chart draw
        setTimeout(() => {
            drawEloChart(null);
            loadAvailableModels().then(() => updateActiveModelInfo());
        }, 500);
        
        console.log('[OK] Gemini Battle module initialized');
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
})();

// =========================================================================
// GAME VISUALIZER MODULE
// =========================================================================

(function() {
    'use strict';
    
    let vizState = {
        moves: [],
        currentStep: 0,
        isPlaying: false,
        playInterval: null,
        // Board state: 18 pits (0-8 white, 9-17 black) + 2 kazans (18 white, 19 black)
        board: null,
        currentPlayer: 0 // 0 = white, 1 = black
    };
    
    const INITIAL_KUMALAKS = 9;
    const TOTAL_KUMALAKS = 162;
    
    // Initialize board state
    function initBoard() {
        vizState.board = new Array(18).fill(INITIAL_KUMALAKS).concat([0, 0]);
        vizState.currentPlayer = 0;
    }
    
    // Parse moves from various formats
    function parseMoves(input) {
        if (!input || !input.trim()) return [];
        
        const cleaned = input.trim();
        let moves = [];
        
        // Try comma-separated: 1,5,3,7...
        if (cleaned.includes(',')) {
            moves = cleaned.split(',').map(m => parseInt(m.trim())).filter(m => m >= 1 && m <= 9);
        }
        // Try space-separated: 1 5 3 7...
        else if (/^\d+(\s+\d+)*$/.test(cleaned)) {
            moves = cleaned.split(/\s+/).map(m => parseInt(m)).filter(m => m >= 1 && m <= 9);
        }
        // Try notation: 1.e4 e5 2.Nf3... (extract numbers)
        else {
            const matches = cleaned.match(/\d+/g);
            if (matches) {
                moves = matches.map(m => parseInt(m)).filter(m => m >= 1 && m <= 9);
            }
        }
        
        return moves;
    }
    
    // Execute a move on the board
    function executeMove(pitIndex) {
        // pitIndex is 1-9, convert to array index
        const isWhite = vizState.currentPlayer === 0;
        const baseIndex = isWhite ? 0 : 9;
        const pit = baseIndex + (pitIndex - 1);
        
        if (vizState.board[pit] === 0) return false; // Invalid move
        
        let kumalaks = vizState.board[pit];
        vizState.board[pit] = 0;
        
        let currentPit = pit;
        
        // Distribute kumalaks
        while (kumalaks > 0) {
            currentPit = (currentPit + 1) % 18;
            
            // Skip tuzdyk (not implemented here for simplicity)
            vizState.board[currentPit]++;
            kumalaks--;
        }
        
        // Check for capture (last pit has even number and in opponent's territory)
        const opponentBase = isWhite ? 9 : 0;
        const opponentEnd = isWhite ? 18 : 9;
        
        if (currentPit >= opponentBase && currentPit < opponentEnd) {
            const finalCount = vizState.board[currentPit];
            if (finalCount % 2 === 0) {
                const kazanIndex = isWhite ? 18 : 19;
                vizState.board[kazanIndex] += finalCount;
                vizState.board[currentPit] = 0;
            }
        }
        
        // Switch player
        vizState.currentPlayer = 1 - vizState.currentPlayer;
        return true;
    }
    
    // Draw board on canvas
    function drawBoard() {
        const canvas = document.getElementById('visualizerBoard');
        if (!canvas || !vizState.board) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear
        ctx.fillStyle = '#8B5A2B';
        ctx.fillRect(0, 0, width, height);
        
        // Draw pits
        const pitWidth = width / 11;
        const pitHeight = height / 3;
        const pitRadius = Math.min(pitWidth, pitHeight) * 0.35;
        
        // Draw black side (top row, pits 9-17, displayed right to left)
        for (let i = 0; i < 9; i++) {
            const x = (9 - i) * pitWidth + pitWidth / 2;
            const y = pitHeight / 2 + 20;
            const pitIndex = 9 + i;
            
            drawPit(ctx, x, y, pitRadius, vizState.board[pitIndex], 'black', i + 1);
        }
        
        // Draw white side (bottom row, pits 0-8, displayed left to right)
        for (let i = 0; i < 9; i++) {
            const x = (i + 1) * pitWidth + pitWidth / 2;
            const y = height - pitHeight / 2 - 20;
            const pitIndex = i;
            
            drawPit(ctx, x, y, pitRadius, vizState.board[pitIndex], 'white', i + 1);
        }
        
        // Draw kazans
        const kazanRadius = pitRadius * 0.8;
        
        // Black kazan (top left)
        ctx.fillStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.arc(pitWidth / 2, height / 2 - 40, kazanRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(vizState.board[19], pitWidth / 2, height / 2 - 35);
        
        // White kazan (bottom right)
        ctx.fillStyle = '#f0f0f0';
        ctx.beginPath();
        ctx.arc(width - pitWidth / 2, height / 2 + 40, kazanRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#000';
        ctx.fillText(vizState.board[18], width - pitWidth / 2, height / 2 + 45);
    }
    
    function drawPit(ctx, x, y, radius, count, side, pitNum) {
        // Pit background
        ctx.fillStyle = side === 'white' ? '#D2B48C' : '#654321';
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Border
        ctx.strokeStyle = side === 'white' ? '#A0522D' : '#3d2817';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Kumalak count
        ctx.fillStyle = side === 'white' ? '#000' : '#fff';
        ctx.font = 'bold 18px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(count, x, y);
        
        // Pit number (small)
        ctx.font = '10px Arial';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillText(pitNum, x, y + radius + 12);
    }
    
    // Update UI
    function updateUI() {
        const moveCounter = document.getElementById('vizMoveCounter');
        const currentPlayer = document.getElementById('vizCurrentPlayer');
        const blackScore = document.getElementById('vizBlackScore');
        const whiteScore = document.getElementById('vizWhiteScore');
        const slider = document.getElementById('vizSlider');
        
        if (moveCounter) {
            moveCounter.textContent = `Ход: ${vizState.currentStep} / ${vizState.moves.length}`;
        }
        if (currentPlayer) {
            currentPlayer.textContent = vizState.currentPlayer === 0 ? 'Ход белых' : 'Ход чёрных';
        }
        if (blackScore && vizState.board) {
            blackScore.textContent = vizState.board[19];
        }
        if (whiteScore && vizState.board) {
            whiteScore.textContent = vizState.board[18];
        }
        if (slider) {
            slider.max = vizState.moves.length;
            slider.value = vizState.currentStep;
        }
        
        // Update moves list
        renderMovesList();
        
        drawBoard();
    }
    
    function renderMovesList() {
        const container = document.getElementById('vizMovesList');
        if (!container || vizState.moves.length === 0) return;
        
        container.innerHTML = vizState.moves.map((move, i) => {
            const isWhite = i % 2 === 0;
            const moveNum = Math.floor(i / 2) + 1;
            const prefix = isWhite ? `${moveNum}.` : '';
            return `<span class="move-item ${i < vizState.currentStep ? 'played' : ''} ${i === vizState.currentStep - 1 ? 'current' : ''}" 
                          data-step="${i + 1}">${prefix}${move}</span>`;
        }).join(' ');
        
        // Add click handlers
        container.querySelectorAll('.move-item').forEach(el => {
            el.addEventListener('click', () => {
                goToStep(parseInt(el.dataset.step));
            });
        });
    }
    
    // Navigation functions
    function goToStep(step) {
        step = Math.max(0, Math.min(step, vizState.moves.length));
        
        // Reset and replay to step
        initBoard();
        vizState.currentStep = 0;
        
        for (let i = 0; i < step; i++) {
            executeMove(vizState.moves[i]);
            vizState.currentStep++;
        }
        
        updateUI();
    }
    
    function nextStep() {
        if (vizState.currentStep < vizState.moves.length) {
            executeMove(vizState.moves[vizState.currentStep]);
            vizState.currentStep++;
            updateUI();
        }
    }
    
    function prevStep() {
        if (vizState.currentStep > 0) {
            goToStep(vizState.currentStep - 1);
        }
    }
    
    function firstStep() {
        goToStep(0);
    }
    
    function lastStep() {
        goToStep(vizState.moves.length);
    }
    
    function togglePlay() {
        if (vizState.isPlaying) {
            clearInterval(vizState.playInterval);
            vizState.isPlaying = false;
            document.getElementById('btnVizPlay').textContent = '▶️';
        } else {
            vizState.isPlaying = true;
            document.getElementById('btnVizPlay').textContent = '⏸';
            vizState.playInterval = setInterval(() => {
                if (vizState.currentStep >= vizState.moves.length) {
                    togglePlay();
                    return;
                }
                nextStep();
            }, 800);
        }
    }
    
    // Load moves from input
    function loadCustomMoves() {
        const input = document.getElementById('visualizerMovesInput');
        if (!input) return;
        
        const moves = parseMoves(input.value);
        if (moves.length === 0) {
            alert('Не удалось распознать ходы. Используйте формат: 1,5,3,7... или 1 5 3 7...');
            return;
        }
        
        vizState.moves = moves;
        initBoard();
        vizState.currentStep = 0;
        updateUI();
        
        document.getElementById('vizMovesList').innerHTML = '';
        renderMovesList();
    }
    
    // Load game from battle history
    async function loadGameFromHistory(gameId) {
        try {
            const response = await fetch(`/api/gemini-battle/game/${gameId}`);
            if (!response.ok) return;
            
            const data = await response.json();
            if (data.moves && data.moves.length > 0) {
                vizState.moves = data.moves;
                initBoard();
                vizState.currentStep = 0;
                updateUI();
            }
        } catch (e) {
            console.error('Error loading game:', e);
        }
    }
    
    // Populate game selector from sessions
    async function populateGameSelector() {
        const select = document.getElementById('visualizerGameSelect');
        if (!select) return;
        
        try {
            const response = await fetch('/api/gemini-battle/sessions');
            if (!response.ok) return;
            
            const data = await response.json();
            const sessions = data.sessions || [];
            
            select.innerHTML = '<option value="">-- Выберите игру --</option>';
            
            sessions.forEach(session => {
                if (session.games && session.games.length > 0) {
                    session.games.forEach((game, i) => {
                        const option = document.createElement('option');
                        option.value = game.id || `${session.session_id}_${i}`;
                        const result = game.winner === 'model' ? '✅' : game.winner === 'gemini' ? '❌' : '🤝';
                        option.textContent = `${result} Игра ${i + 1} (${session.session_id.slice(0, 8)})`;
                        select.appendChild(option);
                    });
                }
            });
        } catch (e) {
            console.error('Error loading sessions for visualizer:', e);
        }
    }
    
    function initVisualizer() {
        // Buttons
        document.getElementById('btnVizFirst')?.addEventListener('click', firstStep);
        document.getElementById('btnVizPrev')?.addEventListener('click', prevStep);
        document.getElementById('btnVizPlay')?.addEventListener('click', togglePlay);
        document.getElementById('btnVizNext')?.addEventListener('click', nextStep);
        document.getElementById('btnVizLast')?.addEventListener('click', lastStep);
        document.getElementById('btnLoadCustomMoves')?.addEventListener('click', loadCustomMoves);
        
        // Slider
        document.getElementById('vizSlider')?.addEventListener('input', (e) => {
            goToStep(parseInt(e.target.value));
        });
        
        // Game selector
        document.getElementById('visualizerGameSelect')?.addEventListener('change', (e) => {
            if (e.target.value) {
                loadGameFromHistory(e.target.value);
            }
        });
        
        // Initialize board and draw
        initBoard();
        drawBoard();
        
        // Populate selector when tab is shown
        const geminiBattleTab = document.querySelector('[data-mode="gemini-battle"]');
        if (geminiBattleTab) {
            geminiBattleTab.addEventListener('click', populateGameSelector);
        }
        
        console.log('[OK] Game Visualizer initialized');
    }
    
    // Initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initVisualizer);
    } else {
        initVisualizer();
    }
    
})();

