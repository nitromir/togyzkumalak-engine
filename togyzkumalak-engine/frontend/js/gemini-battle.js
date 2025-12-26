/**
 * Gemini Battle Module
 * 
 * Handles the UI for Model vs Gemini battles.
 * Features:
 * - Start/stop battle sessions
 * - Real-time progress tracking
 * - ELO chart visualization
 * - Game summaries display
 */

(function() {
    'use strict';
    
    // =========================================================================
    // State
    // =========================================================================
    
    let currentSessionId = null;
    let pollInterval = null;
    let eloChart = null;
    
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
        
        // Buttons
        btnStart: document.getElementById('btnStartBattle'),
        btnStop: document.getElementById('btnStopBattle'),
        
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
        
        // Sessions list
        sessionsList: document.getElementById('battleSessionsList')
    };
    
    // =========================================================================
    // API Functions
    // =========================================================================
    
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
        const xScale = (i) => padding.left + (i / (values.length - 1)) * chartWidth;
        const yScale = (v) => padding.top + chartHeight - ((v - min_elo) / (max_elo - min_elo)) * chartHeight;
        
        // Draw threshold lines
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 1;
        thresholds.forEach(t => {
            if (t.value >= min_elo && t.value <= max_elo) {
                const y = yScale(t.value);
                
                // Line color based on level
                if (t.value >= 2200) ctx.strokeStyle = '#fbbf24';
                else if (t.value >= 2000) ctx.strokeStyle = '#f97316';
                else if (t.value >= 1800) ctx.strokeStyle = '#ef4444';
                else if (t.value >= 1400) ctx.strokeStyle = '#10b981';
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
        
        elements.status.innerHTML = `<p class="status-text ${statusClass}">${statusText}</p>`;
        
        // Update progress bar
        elements.progressContainer.classList.remove('hidden');
        elements.progressBar.style.width = `${data.progress_percent}%`;
        elements.progressText.textContent = `${data.games_played} / ${data.total_games} игр`;
        
        // Update stats
        elements.statsContainer.classList.remove('hidden');
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
        const config = {
            num_games: parseInt(elements.numGames.value) || 10,
            model_level: parseInt(elements.modelLevel.value) || 5,
            gemini_timeout: parseInt(elements.timeout.value) || 30,
            save_replays: elements.saveReplays.checked,
            generate_summaries: elements.generateSummaries.checked
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
    
    function init() {
        // Check if elements exist
        if (!elements.btnStart) {
            console.warn('Gemini Battle elements not found');
            return;
        }
        
        // Event listeners
        elements.btnStart.addEventListener('click', handleStartBattle);
        elements.btnStop.addEventListener('click', handleStopBattle);
        
        // Load existing sessions when tab is shown
        const geminiBattleTab = document.querySelector('[data-mode="gemini-battle"]');
        if (geminiBattleTab) {
            geminiBattleTab.addEventListener('click', () => {
                loadSessionsList();
                
                // Draw empty chart initially
                drawEloChart(null);
            });
        }
        
        // Initial chart draw
        setTimeout(() => {
            drawEloChart(null);
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

