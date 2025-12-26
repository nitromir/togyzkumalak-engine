/**
 * Analytics Controller - REAL DATA ONLY
 * 
 * All data comes from actual game files via API.
 * NO MOCKS, NO FAKE DATA.
 */

class AnalyticsController {
    constructor() {
        this.metricsData = null;
        this.eloChart = null;
        this.lossChart = null;
        this.datasetChart = null;
        this.refreshInterval = null;
        this.isAutoRefresh = false;
    }

    async init() {
        console.log('[Analytics] Initializing with REAL data...');
        
        // Bind controls
        this.bindControls();
        
        // Initial load
        await this.loadAllMetrics();
        
        console.log('[Analytics] Initialized');
    }

    bindControls() {
        // Refresh button
        const refreshBtn = document.getElementById('btnRefreshMetrics');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadAllMetrics());
        }

        // Auto-refresh toggle
        const autoRefreshBtn = document.getElementById('btnAutoRefresh');
        if (autoRefreshBtn) {
            autoRefreshBtn.addEventListener('click', () => this.toggleAutoRefresh());
        }

        // Sync to W&B button
        const syncBtn = document.getElementById('btnSyncWandB');
        if (syncBtn) {
            syncBtn.addEventListener('click', () => this.syncToWandB());
        }
    }

    async loadAllMetrics() {
        try {
            console.log('[Analytics] Loading REAL metrics from files...');
            
            const response = await fetch('/api/metrics/all');
            if (!response.ok) throw new Error('Failed to fetch metrics');
            
            this.metricsData = await response.json();
            console.log('[Analytics] Loaded metrics:', this.metricsData);
            
            // Update all displays
            this.updateGeminiBattleStats();
            this.updateEloDisplay();
            this.updateDatasetDisplay();
            this.updateConvergenceDisplay();
            this.updateEloChart();
            
            // Update timestamp
            const timestampEl = document.getElementById('metricsTimestamp');
            if (timestampEl) {
                timestampEl.textContent = `Обновлено: ${new Date().toLocaleTimeString()}`;
            }
        } catch (error) {
            console.error('[Analytics] Error loading metrics:', error);
        }
    }

    updateGeminiBattleStats() {
        const data = this.metricsData?.gemini_battles || {};
        
        // Update stat values
        this.setElementText('statTotalGames', data.total_games || 0);
        this.setElementText('statModelWins', data.model_wins || 0);
        this.setElementText('statGeminiWins', data.gemini_wins || 0);
        this.setElementText('statDraws', data.draws || 0);
        this.setElementText('statWinrate', `${(data.winrate || 0).toFixed(1)}%`);
        this.setElementText('statAvgGameLength', (data.avg_game_length || 0).toFixed(1));
        this.setElementText('statTotalMoves', data.total_moves || 0);
        
        // Update recent games list
        this.updateRecentGames(data.recent_games || []);
    }

    updateRecentGames(games) {
        const container = document.getElementById('recentGamesList');
        if (!container) return;

        if (games.length === 0) {
            container.innerHTML = '<p class="empty-text">Нет сыгранных игр</p>';
            return;
        }

        container.innerHTML = games.map(game => {
            const resultClass = game.winner === game.model_color ? 'win' : 
                               (game.winner === 'draw' ? 'draw' : 'loss');
            const resultText = game.winner === game.model_color ? 'Победа' :
                              (game.winner === 'draw' ? 'Ничья' : 'Поражение');
            
            return `
                <div class="recent-game-item ${resultClass}">
                    <span class="game-result">${resultText}</span>
                    <span class="game-moves">${game.total_moves} ходов</span>
                    <span class="game-session">#${game.session_id}</span>
                </div>
            `;
        }).join('');
    }

    updateEloDisplay() {
        const data = this.metricsData?.elo || {};
        
        this.setElementText('currentElo', data.current_elo || 1500);
        this.setElementText('peakElo', data.peak_elo || 1500);
        this.setElementText('eloCategory', data.category || 'Начинающий');
        this.setElementText('eloTotalGames', data.total_games || 0);

        // Update ELO change indicator
        const history = data.history || [];
        if (history.length >= 2) {
            const lastChange = history[history.length - 1].change || 0;
            const changeEl = document.getElementById('lastEloChange');
            if (changeEl) {
                const sign = lastChange >= 0 ? '+' : '';
                changeEl.textContent = `${sign}${lastChange}`;
                changeEl.className = `elo-change ${lastChange >= 0 ? 'positive' : 'negative'}`;
            }
        }
    }

    updateDatasetDisplay() {
        const data = this.metricsData?.dataset || {};
        
        this.setElementText('datasetGeminiGames', data.gemini_games || 0);
        this.setElementText('datasetSelfPlayGames', data.self_play_games || 0);
        this.setElementText('datasetTotalTransitions', data.total_transitions || 0);
        this.setElementText('datasetDiskSize', `${(data.total_disk_size_mb || 0).toFixed(2)} MB`);

        // Update composition bar
        this.updateDatasetBar(data);
    }

    updateDatasetBar(data) {
        const geminiBar = document.getElementById('barGemini');
        const selfPlayBar = document.getElementById('barSelfPlay');
        const humanBar = document.getElementById('barHuman');

        if (geminiBar) geminiBar.style.width = `${data.gemini_pct || 0}%`;
        if (selfPlayBar) selfPlayBar.style.width = `${data.self_play_pct || 0}%`;
        if (humanBar) humanBar.style.width = `${data.human_pct || 0}%`;
    }

    updateConvergenceDisplay() {
        const data = this.metricsData?.convergence || {};
        
        const statusEl = document.getElementById('convergenceStatus');
        if (statusEl) {
            const statusMap = {
                'improving': { text: '📈 Улучшается', class: 'improving' },
                'converged': { text: '⚠️ Сходимость', class: 'converged' },
                'degrading': { text: '📉 Ухудшается', class: 'degrading' },
                'stable': { text: '➡️ Стабильно', class: 'stable' },
                'insufficient_data': { text: '❓ Мало данных', class: 'unknown' }
            };
            
            const status = statusMap[data.status] || statusMap['insufficient_data'];
            statusEl.textContent = status.text;
            statusEl.className = `convergence-status ${status.class}`;
        }

        this.setElementText('convergenceVariance', data.variance?.toFixed(2) || 'N/A');
        this.setElementText('convergenceTrend', data.trend?.toFixed(2) || 'N/A');
        this.setElementText('convergenceRecommendation', data.recommendation || '');
    }

    updateEloChart() {
        const data = this.metricsData?.elo || {};
        const history = data.history || [];

        if (history.length === 0) return;

        const canvas = document.getElementById('eloProgressChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Destroy existing chart
        if (this.eloChart) {
            this.eloChart.destroy();
        }

        // Prepare data
        const labels = history.map((h, i) => i + 1);
        const values = history.map(h => h.elo);

        // Create new chart
        this.eloChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'ELO Rating',
                    data: values,
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0, 242, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                    pointBackgroundColor: '#00f2ff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#00f2ff',
                        bodyColor: '#f1f5f9',
                        borderColor: '#00f2ff',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Игра',
                            color: '#94a3b8'
                        },
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(148, 163, 184, 0.1)' }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'ELO',
                            color: '#94a3b8'
                        },
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(148, 163, 184, 0.1)' },
                        min: Math.min(...values) - 50,
                        max: Math.max(...values) + 50
                    }
                }
            }
        });
    }

    toggleAutoRefresh() {
        this.isAutoRefresh = !this.isAutoRefresh;
        
        const btn = document.getElementById('btnAutoRefresh');
        if (btn) {
            btn.textContent = this.isAutoRefresh ? '⏹️ Стоп' : '▶️ Авто';
            btn.classList.toggle('active', this.isAutoRefresh);
        }

        if (this.isAutoRefresh) {
            this.refreshInterval = setInterval(() => this.loadAllMetrics(), 5000);
        } else {
            if (this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        }
    }

    async syncToWandB() {
        try {
            const response = await fetch('/api/wandb/sync', { method: 'POST' });
            if (response.ok) {
                this.showNotification('Метрики синхронизированы с W&B', 'success');
            } else {
                this.showNotification('Ошибка синхронизации', 'error');
            }
        } catch (error) {
            console.error('[Analytics] W&B sync error:', error);
            this.showNotification('Ошибка синхронизации', 'error');
        }
    }

    setElementText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    showNotification(message, type = 'info') {
        // Simple notification
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}

// A/B Test Controller - REAL EXPERIMENTS
class ABTestController {
    constructor() {
        this.experiments = [];
    }

    async init() {
        console.log('[A/B Test] Initializing...');
        this.bindControls();
        await this.loadExperiments();
    }

    bindControls() {
        const createBtn = document.getElementById('btnCreateExperiment');
        if (createBtn) {
            createBtn.addEventListener('click', () => this.createExperiment());
        }
    }

    async loadExperiments() {
        try {
            const response = await fetch('/api/ab-test/experiments');
            if (!response.ok) throw new Error('Failed to load experiments');
            
            const data = await response.json();
            this.experiments = data.experiments || [];
            this.renderExperiments();
        } catch (error) {
            console.error('[A/B Test] Error loading experiments:', error);
        }
    }

    async createExperiment() {
        const name = prompt('Название эксперимента:');
        if (!name) return;

        try {
            const response = await fetch('/api/ab-test/experiments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    variants: ['structured', 'tactical', 'beginner'],
                    description: 'A/B test for commentary schemas'
                })
            });

            if (response.ok) {
                console.log('[A/B Test] Experiment created');
                await this.loadExperiments();
            }
        } catch (error) {
            console.error('[A/B Test] Error creating experiment:', error);
        }
    }

    async loadExperimentStats(experimentId) {
        try {
            const response = await fetch(`/api/ab-test/experiments/${experimentId}/stats`);
            if (!response.ok) throw new Error('Failed to load stats');
            
            return await response.json();
        } catch (error) {
            console.error('[A/B Test] Error loading stats:', error);
            return null;
        }
    }

    async submitFeedback(experimentId, variant, gameId, moveNumber, rating, helpful, accurate) {
        try {
            const response = await fetch('/api/ab-test/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    experiment_id: experimentId,
                    variant: variant,
                    game_id: gameId,
                    move_number: moveNumber,
                    user_rating: rating,
                    was_helpful: helpful,
                    was_accurate: accurate
                })
            });

            return response.ok;
        } catch (error) {
            console.error('[A/B Test] Error submitting feedback:', error);
            return false;
        }
    }

    renderExperiments() {
        const container = document.getElementById('experimentsList');
        if (!container) return;

        if (this.experiments.length === 0) {
            container.innerHTML = '<p class="empty-text">Нет активных экспериментов</p>';
            return;
        }

        container.innerHTML = this.experiments.map(exp => `
            <div class="experiment-card" data-id="${exp.experiment_id}">
                <div class="experiment-header">
                    <span class="experiment-name">${exp.name}</span>
                    <span class="experiment-status ${exp.is_active ? 'active' : 'stopped'}">
                        ${exp.is_active ? '🟢 Активен' : '🔴 Остановлен'}
                    </span>
                </div>
                <div class="experiment-variants">
                    ${exp.variants.map(v => `<span class="variant-tag">${v}</span>`).join('')}
                </div>
                <div class="experiment-actions">
                    <button class="btn btn-small" onclick="abTestController.showStats('${exp.experiment_id}')">
                        📊 Статистика
                    </button>
                    ${exp.is_active ? `
                        <button class="btn btn-small btn-danger" onclick="abTestController.stopExperiment('${exp.experiment_id}')">
                            ⏹️ Стоп
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }

    async showStats(experimentId) {
        const stats = await this.loadExperimentStats(experimentId);
        if (!stats) return;

        // Show stats in modal or panel
        const statsHtml = `
            <div class="stats-popup">
                <h3>Статистика эксперимента</h3>
                <p>Всего тестов: ${stats.total_samples}</p>
                <p>Победитель: ${stats.winner || 'Недостаточно данных'}</p>
                <div class="variants-stats">
                    ${Object.entries(stats.variants || {}).map(([variant, data]) => `
                        <div class="variant-stat ${variant === stats.winner ? 'winner' : ''}">
                            <h4>${variant}</h4>
                            <p>Тестов: ${data.sample_size}</p>
                            <p>Средний рейтинг: ${data.avg_rating?.toFixed(1) || 'N/A'}</p>
                            <p>Полезность: ${data.helpful_rate ? (data.helpful_rate * 100).toFixed(0) + '%' : 'N/A'}</p>
                            <p>Время ответа: ${data.avg_response_time_ms?.toFixed(0) || 'N/A'}ms</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        console.log('[A/B Test] Stats:', stats);
        alert(`Статистика:\n\nВсего тестов: ${stats.total_samples}\nПобедитель: ${stats.winner || 'Недостаточно данных'}`);
    }

    async stopExperiment(experimentId) {
        try {
            const response = await fetch(`/api/ab-test/experiments/${experimentId}/stop`, {
                method: 'POST'
            });

            if (response.ok) {
                console.log('[A/B Test] Experiment stopped');
                await this.loadExperiments();
            }
        } catch (error) {
            console.error('[A/B Test] Error stopping experiment:', error);
        }
    }
}

// Global instances
let analyticsController;
let abTestController;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    analyticsController = new AnalyticsController();
    abTestController = new ABTestController();
    
    // Export to window for use from other scripts
    window.analyticsController = analyticsController;
    window.abTestController = abTestController;
});

