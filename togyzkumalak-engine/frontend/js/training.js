/**
 * Training Mode Controller
 * Manages gym training sessions, progress monitoring, and model management
 */

class TrainingController {
    constructor() {
        this.currentSessionId = null;
        this.humanTrainingSessionId = null;
        this.pollInterval = null;
        this.humanPollInterval = null;
        this.trainingChart = null;
        this.chartData = { epochs: [], loss: [], accuracy: [] };
        this.init();
    }

    init() {
        // Configuration controls
        document.getElementById('btnStartTraining').addEventListener('click', () => this.startTraining());
        
        // Data management buttons
        document.getElementById('btnParseData')?.addEventListener('click', () => this.parseData());
        document.getElementById('btnTrainOnHuman')?.addEventListener('click', () => this.trainOnHumanData());
        
        // FAQ modal
        document.getElementById('btnOpenFaq')?.addEventListener('click', () => this.openFaqModal());
        document.getElementById('closeFaqModal')?.addEventListener('click', () => this.closeFaqModal());
        document.getElementById('faqModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'faqModal') this.closeFaqModal();
        });
        
        // Load initial data
        this.loadModels();
        this.loadSessions();
        this.loadDataStats();
        this.loadTrainingFiles();
        
        // AlphaZero Initialization
        this.initAlphaZero();
        
        // Set up periodic refresh when training mode is active
        setInterval(() => {
            if (!document.getElementById('trainingMode').classList.contains('hidden')) {
                this.loadModels();
                this.loadSessions();
                this.loadDataStats();
            }
        }, 5000);
    }

    /**
     * Load training data statistics
     */
    async loadDataStats() {
        try {
            const response = await fetch('/api/data/stats');
            if (!response.ok) return;
            
            const data = await response.json();
            
            if (data.parsed && data.stats) {
                const stats = data.stats;
                document.getElementById('statOpeningBook').textContent = stats.opening_book || 0;
                document.getElementById('statTournament').textContent = stats.human_tournament || 0;
                document.getElementById('statPlayOK').textContent = stats.playok || 0;
                document.getElementById('statTotalGamesData').textContent = stats.total_games || 0;
                document.getElementById('statTransitions').textContent = this.formatNumber(stats.total_transitions || 0);
            }
        } catch (error) {
            console.error('Error loading data stats:', error);
        }
    }

    /**
     * Load list of training files
     */
    async loadTrainingFiles() {
        try {
            const response = await fetch('/api/data/training-files');
            if (!response.ok) return;
            
            const data = await response.json();
            const filesList = document.getElementById('dataFilesList');
            
            if (!data.files || data.files.length === 0) {
                filesList.innerHTML = '<p class="empty-text">Нет файлов данных. Нажмите "Обновить парсинг"</p>';
                return;
            }
            
            filesList.innerHTML = data.files.map(file => `
                <div class="data-file-item">
                    <span class="data-file-name">📄 ${file.name}</span>
                    <div class="data-file-info">
                        <span class="data-file-size">${file.size_mb} MB</span>
                        ${file.lines > 0 ? `<span class="data-file-lines">${this.formatNumber(file.lines)} записей</span>` : ''}
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Error loading training files:', error);
        }
    }

    /**
     * Parse all training data from sources
     */
    async parseData() {
        const btn = document.getElementById('btnParseData');
        btn.disabled = true;
        btn.textContent = '⏳ Парсинг...';
        
        try {
            const response = await fetch('/api/data/parse-auto', { method: 'POST' });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Parsing failed');
            }
            
            const data = await response.json();
            
            alert(`✅ Парсинг завершён!\n\nИгр: ${data.stats.total_games}\nПереходов: ${data.stats.total_transitions}`);
            
            // Reload stats and files
            this.loadDataStats();
            this.loadTrainingFiles();
            
        } catch (error) {
            console.error('Error parsing data:', error);
            alert('❌ Ошибка парсинга: ' + error.message);
        } finally {
            btn.disabled = false;
            btn.textContent = '🔄 Обновить парсинг';
        }
    }

    /**
     * Start training on human data
     */
    async trainOnHumanData() {
        const btn = document.getElementById('btnTrainOnHuman');
        btn.disabled = true;
        btn.textContent = '⏳ Запуск...';
        
        // Get config from inputs
        const epochs = parseInt(document.getElementById('humanEpochs')?.value) || 50;
        const batchSize = parseInt(document.getElementById('humanBatchSize')?.value) || 128;
        const learningRate = parseFloat(document.getElementById('humanLearningRate')?.value) || 0.001;
        const baseModel = document.getElementById('humanBaseModel')?.value || '';
        
        // Reset chart data
        this.chartData = { epochs: [], loss: [], accuracy: [] };
        this.initTrainingChart();
        
        try {
            const response = await fetch('/api/training/human-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    batch_size: batchSize,
                    epochs: epochs,
                    learning_rate: learningRate,
                    model_name: 'policy_net_human',
                    use_compact: true,
                    base_model: baseModel  // Use existing model as starting point
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Training start failed');
            }
            
            const data = await response.json();
            this.humanTrainingSessionId = data.session_id;
            
            // Show progress section
            const progressSection = document.getElementById('humanTrainingProgress');
            progressSection.classList.remove('hidden');
            
            btn.textContent = '🔄 Обучение...';
            
            // Start polling for progress
            this.startHumanTrainingPolling();
            
        } catch (error) {
            console.error('Error starting human training:', error);
            alert('❌ Ошибка запуска: ' + error.message);
            btn.disabled = false;
            btn.textContent = '🎓 Дообучить на человеческих данных';
        }
    }

    /**
     * Start polling for human training progress
     */
    startHumanTrainingPolling() {
        if (this.humanPollInterval) {
            clearInterval(this.humanPollInterval);
        }
        
        this.humanPollInterval = setInterval(() => this.updateHumanTrainingProgress(), 1000);
    }

    /**
     * Update human training progress display
     */
    async updateHumanTrainingProgress() {
        if (!this.humanTrainingSessionId) return;
        
        try {
            const response = await fetch(`/api/training/human-data/${this.humanTrainingSessionId}`);
            if (!response.ok) return;
            
            const progress = await response.json();
            
            const progressBar = document.getElementById('humanTrainingBar');
            const progressText = document.getElementById('humanTrainingText');
            
            progressBar.style.width = `${progress.progress || 0}%`;
            
            if (progress.status === 'loading') {
                progressText.textContent = '📂 Загрузка данных...';
            } else if (progress.status === 'training') {
                const epoch = progress.epoch || 0;
                const loss = progress.loss || 0;
                const accuracy = progress.accuracy || 0;
                const modelInfo = progress.model_info ? `<div class="training-info-badge ${progress.model_info.includes('новая') ? 'new' : ''}">${progress.model_info}</div>` : '';
                
                progressText.innerHTML = `
                    ${modelInfo}
                    <div>Эпоха ${epoch}/${progress.total_epochs} | 
                    Loss: ${loss.toFixed(4)} | 
                    Accuracy: ${accuracy.toFixed(1)}%</div>
                    <div class="training-metrics-grid">
                        <div class="training-metric">
                            <div class="value">${this.formatNumber(progress.samples_trained || 0)}</div>
                            <div class="label">Примеров</div>
                        </div>
                        <div class="training-metric">
                            <div class="value">${epoch}</div>
                            <div class="label">Эпоха</div>
                        </div>
                        <div class="training-metric">
                            <div class="value">${loss.toFixed(4)}</div>
                            <div class="label">Loss</div>
                        </div>
                        <div class="training-metric">
                            <div class="value">${accuracy.toFixed(1)}%</div>
                            <div class="label">Accuracy</div>
                        </div>
                    </div>
                `;
                
                // Update chart
                this.updateTrainingChart(epoch, loss, accuracy);
            } else if (progress.status === 'completed') {
                clearInterval(this.humanPollInterval);
                this.humanPollInterval = null;
                
                const modelName = progress.model_name || 'policy_net_human';
                const version = progress.version || '?';
                
                progressText.innerHTML = `
                    ✅ Обучение завершено!<br>
                    <b>Accuracy: ${(progress.accuracy || 0).toFixed(1)}%</b> | 
                    Final Loss: ${(progress.loss || 0).toFixed(4)}<br>
                    <span style="color: var(--accent-gold);">📦 Модель: ${modelName}.pt</span><br><br>
                    <b>Что дальше:</b><br>
                    • 🎯 Играть → уровень 5 (Эксперт)<br>
                    • 🔄 Self-Play → ещё улучшить<br>
                    • 🤖 Gemini Battle → проверить ELO
                `;
                
                const btn = document.getElementById('btnTrainOnHuman');
                btn.disabled = false;
                btn.textContent = '🎓 Дообучить на человеческих данных';
                
                this.loadModels();
                
            } else if (progress.status === 'error') {
                clearInterval(this.humanPollInterval);
                this.humanPollInterval = null;
                
                progressText.textContent = `❌ Ошибка: ${progress.error || 'Unknown error'}`;
                
                const btn = document.getElementById('btnTrainOnHuman');
                btn.disabled = false;
                btn.textContent = '🎓 Дообучить на человеческих данных';
            }
            
        } catch (error) {
            console.error('Error updating human training progress:', error);
        }
    }

    /**
     * Open FAQ modal
     */
    openFaqModal() {
        document.getElementById('faqModal')?.classList.remove('hidden');
    }

    /**
     * Close FAQ modal
     */
    closeFaqModal() {
        document.getElementById('faqModal')?.classList.add('hidden');
    }

    /**
     * Initialize training chart
     */
    initTrainingChart() {
        const ctx = document.getElementById('trainingChart')?.getContext('2d');
        if (!ctx) return;
        
        if (this.trainingChart) {
            this.trainingChart.destroy();
        }
        
        this.trainingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Loss',
                        data: [],
                        borderColor: '#ff0055',
                        backgroundColor: 'rgba(255, 0, 85, 0.1)',
                        yAxisID: 'y',
                        tension: 0.3
                    },
                    {
                        label: 'Accuracy %',
                        data: [],
                        borderColor: '#00f2ff',
                        backgroundColor: 'rgba(0, 242, 255, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { 
                        labels: { color: '#94a3b8', font: { size: 11 } }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Эпоха', color: '#64748b' },
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(100, 116, 139, 0.2)' }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: { display: true, text: 'Loss', color: '#ff0055' },
                        ticks: { color: '#ff0055' },
                        grid: { color: 'rgba(100, 116, 139, 0.2)' }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: 'Accuracy %', color: '#00f2ff' },
                        ticks: { color: '#00f2ff' },
                        grid: { drawOnChartArea: false },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    /**
     * Update training chart with new data
     */
    updateTrainingChart(epoch, loss, accuracy) {
        if (!this.trainingChart) return;
        
        // Avoid duplicate epochs
        if (!this.chartData.epochs.includes(epoch)) {
            this.chartData.epochs.push(epoch);
            this.chartData.loss.push(loss);
            this.chartData.accuracy.push(accuracy);
            
            this.trainingChart.data.labels = [...this.chartData.epochs];
            this.trainingChart.data.datasets[0].data = [...this.chartData.loss];
            this.trainingChart.data.datasets[1].data = [...this.chartData.accuracy];
            this.trainingChart.update('none');
        }
    }

    /**
     * AlphaZero Methods
     */
    initAlphaZero() {
        this.azTaskId = null;
        this.azPollInterval = null;
        this.azLossChart = null;
        this.azEloChart = null;
        
        const btnStart = document.getElementById('btnStartAlphaZero');
        const btnStop = document.getElementById('btnStopAlphaZero');
        
        btnStart?.addEventListener('click', () => this.startAlphaZero());
        btnStop?.addEventListener('click', () => this.stopAlphaZero());
        
        this.initAlphaZeroCharts();
        this.loadAlphaZeroMetrics();  // Load last training metrics
    }
    
    /**
     * Load and display AlphaZero training metrics from last training
     */
    async loadAlphaZeroMetrics() {
        try {
            const response = await fetch('/api/training/alphazero/metrics');
            if (!response.ok) return;
            
            const data = await response.json();
            const summary = data.summary || {};
            const checkpoints = data.checkpoints || [];
            const metrics = data.metrics || [];
            
            // Update metrics display
            const policyLoss = summary.latest_policy_loss || 0;
            const valueLoss = summary.latest_value_loss || 0;
            const winRate = (summary.latest_win_rate || 0) * 100;
            const totalExamples = summary.total_examples || 0;
            
            const policyEl = document.getElementById('lastPolicyLoss');
            const valueEl = document.getElementById('lastValueLoss');
            const winEl = document.getElementById('lastWinRate');
            const examplesEl = document.getElementById('totalExamples');
            
            if (policyEl) {
                policyEl.textContent = policyLoss.toFixed(3);
                policyEl.className = 'metric-value ' + (policyLoss < 1.0 ? 'good' : policyLoss < 1.5 ? 'warning' : 'bad');
            }
            if (valueEl) {
                valueEl.textContent = valueLoss.toFixed(3);
                valueEl.className = 'metric-value ' + (valueLoss < 0.1 ? 'good' : valueLoss < 0.2 ? 'warning' : 'bad');
            }
            if (winEl) {
                winEl.textContent = winRate.toFixed(0) + '%';
                winEl.className = 'metric-value ' + (winRate > 55 ? 'good' : winRate > 50 ? 'warning' : 'bad');
            }
            if (examplesEl) {
                examplesEl.textContent = this.formatNumber(totalExamples);
            }
            
            // Show best checkpoint info
            if (summary.best_checkpoint) {
                const best = summary.best_checkpoint;
                const bestInfoEl = document.getElementById('bestCheckpointInfo');
                if (bestInfoEl) {
                    bestInfoEl.innerHTML = `
                        <strong>🏆 Лучший:</strong> iter ${best.iteration} 
                        (loss: ${best.policy_loss.toFixed(3)})
                        <button class="btn btn-small btn-secondary" 
                                onclick="trainingController.loadAlphaZeroCheckpoint('${best.filename}')">
                            Загрузить
                        </button>
                    `;
                }
            }
            
            // Show checkpoints list
            this.renderCheckpointsList(checkpoints);
            
            // Update charts with full metrics
            if (metrics.length > 0 && this.azLossChart) {
                this.updateAlphaZeroChartsWithMetrics(metrics);
            }
            
        } catch (e) {
            console.error('Error loading AlphaZero metrics:', e);
        }
    }
    
    /**
     * Render list of best checkpoints
     */
    renderCheckpointsList(checkpoints) {
        const container = document.getElementById('azCheckpointsList');
        if (!container || !checkpoints.length) return;
        
        container.innerHTML = checkpoints.slice(0, 5).map((cp, i) => `
            <div class="checkpoint-item ${i === 0 ? 'best' : ''}">
                <span class="cp-rank">#${i + 1}</span>
                <span class="cp-iter">iter ${cp.iteration}</span>
                <span class="cp-loss">${cp.policy_loss.toFixed(3)}</span>
                <button class="btn btn-tiny" onclick="trainingController.loadAlphaZeroCheckpoint('${cp.filename}')">
                    ⬇️
                </button>
            </div>
        `).join('');
    }
    
    /**
     * Update charts with full training metrics
     */
    updateAlphaZeroChartsWithMetrics(metrics) {
        if (!this.azLossChart || !this.azEloChart) return;
        
        // Loss chart
        this.azLossChart.data.labels = metrics.map(m => m.iteration);
        this.azLossChart.data.datasets[0].data = metrics.map(m => m.policy_loss);
        if (this.azLossChart.data.datasets[1]) {
            this.azLossChart.data.datasets[1].data = metrics.map(m => m.value_loss);
        }
        this.azLossChart.update('none');
        
        // Win rate chart (as proxy for ELO improvement)
        this.azEloChart.data.labels = metrics.map(m => m.iteration);
        this.azEloChart.data.datasets[0].data = metrics.map(m => (m.win_rate || 0) * 100);
        this.azEloChart.update('none');
    }
    
    /**
     * Load specific AlphaZero checkpoint
     */
    async loadAlphaZeroCheckpoint(filename) {
        try {
            const modelName = filename.replace('.pth.tar', '');
            const response = await fetch(`/api/training/models/alphazero/${modelName}/load`, {
                method: 'POST'
            });
            
            if (response.ok) {
                alert(`✅ Загружен чекпоинт: ${filename}`);
                this.loadModels();
            } else {
                throw new Error('Failed to load checkpoint');
            }
        } catch (e) {
            console.error('Error loading checkpoint:', e);
            alert('Ошибка загрузки чекпоинта: ' + e.message);
        }
    }

    async startAlphaZero() {
        const useBootstrap = document.getElementById('azUseBootstrap')?.checked ?? true;
        
        const config = {
            numIters: parseInt(document.getElementById('azIters').value),
            numEps: parseInt(document.getElementById('azEps').value),
            numMCTSSims: parseInt(document.getElementById('azSims').value),
            cpuct: parseFloat(document.getElementById('azCpuct').value),
            useBootstrap: useBootstrap
        };

        try {
            document.getElementById('btnStartAlphaZero').disabled = true;
            
            const response = await fetch('/api/training/alphazero/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (!response.ok) throw new Error('Failed to start AlphaZero');

            const data = await response.json();
            this.azTaskId = data.task_id;

            document.getElementById('azProgressSection').classList.remove('hidden');
            document.getElementById('btnStopAlphaZero').classList.remove('hidden');
            
            this.startAlphaZeroPolling();
        } catch (error) {
            console.error('Error starting AlphaZero:', error);
            alert('Ошибка запуска AlphaZero: ' + error.message);
            document.getElementById('btnStartAlphaZero').disabled = false;
        }
    }

    startAlphaZeroPolling() {
        if (this.azPollInterval) clearInterval(this.azPollInterval);
        this.azPollInterval = setInterval(() => this.updateAlphaZeroProgress(), 2000);
    }

    async updateAlphaZeroProgress() {
        if (!this.azTaskId) return;

        try {
            const response = await fetch(`/api/training/alphazero/sessions/${this.azTaskId}`);
            if (!response.ok) return;

            const task = await response.json();
            
            document.getElementById('azProgressBar').style.width = `${task.progress}%`;
            document.getElementById('azCurrentIter').textContent = `${task.current_iteration} / ${task.total_iterations}`;
            document.getElementById('azStatusText').textContent = task.status;

            // Update charts
            if (task.metrics.loss.length > 0) {
                this.azLossChart.data.labels = task.metrics.loss.map(m => m.iter);
                this.azLossChart.data.datasets[0].data = task.metrics.loss.map(m => m.value);
                this.azLossChart.update('none');
            }

            if (task.metrics.accuracy && task.metrics.accuracy.length > 0) {
                this.azEloChart.data.labels = task.metrics.accuracy.map(m => m.iter);
                this.azEloChart.data.datasets[0].data = task.metrics.accuracy.map(m => m.value);
                this.azEloChart.update('none');
            }

            if (task.status === 'completed' || task.status === 'error' || task.status === 'stopped') {
                clearInterval(this.azPollInterval);
                document.getElementById('btnStartAlphaZero').disabled = false;
                document.getElementById('btnStopAlphaZero').classList.add('hidden');
                
                if (task.status === 'completed') {
                    this.showNotification('🦾 AlphaZero: Обучение завершено! Новая модель готова.');
                }
            }
        } catch (error) {
            console.error('Error polling AlphaZero:', error);
        }
    }

    async stopAlphaZero() {
        if (!this.azTaskId) return;
        await fetch(`/api/training/alphazero/sessions/${this.azTaskId}/stop`, { method: 'POST' });
    }

    initAlphaZeroCharts() {
        const ctxLoss = document.getElementById('azLossChart')?.getContext('2d');
        const ctxElo = document.getElementById('azEloChart')?.getContext('2d');
        
        if (!ctxLoss || !ctxElo) return;

        this.azLossChart = new Chart(ctxLoss, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#ff0055', tension: 0.3 }] },
            options: { responsive: true, maintainAspectRatio: false }
        });

        this.azEloChart = new Chart(ctxElo, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Accuracy %', data: [], borderColor: '#00f2ff', tension: 0.3 }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    showNotification(message) {
        // Simple alert for now, can be upgraded to Toast
        alert(message);
    }

    /**
     * Format large numbers
     */
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    async startTraining() {
        const config = {
            num_games: parseInt(document.getElementById('numGames').value),
            epsilon: parseFloat(document.getElementById('epsilon').value),
            hidden_size: parseInt(document.getElementById('hiddenSize').value),
            learning_rate: 0.001,
            save_replays: document.getElementById('saveReplays').checked,
            model_name: document.getElementById('modelName').value
        };

        // Validate
        if (config.num_games < 1 || config.num_games > 1000) {
            alert('Количество игр должно быть от 1 до 1000');
            return;
        }

        if (config.epsilon < 0 || config.epsilon > 1) {
            alert('Epsilon должен быть от 0 до 1');
            return;
        }

        try {
            document.getElementById('btnStartTraining').disabled = true;
            
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                throw new Error('Failed to start training');
            }

            const data = await response.json();
            this.currentSessionId = data.session_id;

            // Update UI
            document.getElementById('trainingStatus').innerHTML = `
                <p class="status-text status-running">🔄 Тренировка запущена</p>
                <p class="session-id">Session ID: ${this.currentSessionId}</p>
            `;

            document.getElementById('trainingProgress').classList.remove('hidden');
            document.getElementById('trainingStats').classList.remove('hidden');

            // Start polling for progress
            this.startProgressPolling();

        } catch (error) {
            console.error('Error starting training:', error);
            alert('Ошибка запуска тренировки: ' + error.message);
            document.getElementById('btnStartTraining').disabled = false;
        }
    }

    startProgressPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(() => {
            this.updateProgress();
        }, 1000);
    }

    stopProgressPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async updateProgress() {
        if (!this.currentSessionId) return;

        try {
            const response = await fetch(`/api/training/sessions/${this.currentSessionId}`);
            if (!response.ok) return;

            const progress = await response.json();

            // Update progress bar
            const percentage = (progress.games_completed / progress.total_games) * 100;
            const modelInfo = progress.model_info ? `<div class="training-info-badge ${progress.model_info.includes('новая') ? 'new' : ''}" style="margin-top: 8px;">${progress.model_info}</div>` : '';
            
            document.getElementById('trainingProgressBar').style.width = `${percentage}%`;
            document.getElementById('trainingProgressText').innerHTML = `
                ${progress.games_completed} / ${progress.total_games} игр
                ${modelInfo}
            `;

            // Update stats
            document.getElementById('whiteWins').textContent = progress.white_wins;
            document.getElementById('blackWins').textContent = progress.black_wins;
            document.getElementById('draws').textContent = progress.draws;
            document.getElementById('avgSteps').textContent = progress.avg_steps.toFixed(1);

            // Check if completed
            if (progress.status === 'completed') {
                this.stopProgressPolling();
                const modelInfo = progress.model_info ? `<div class="training-info-badge ${progress.model_info.includes('новая') ? 'new' : ''}">${progress.model_info}</div>` : '';
                document.getElementById('trainingStatus').innerHTML = `
                    <p class="status-text status-completed">✅ Тренировка завершена</p>
                    ${modelInfo}
                    <p class="session-id">Session ID: ${this.currentSessionId}</p>
                `;
                document.getElementById('btnStartTraining').disabled = false;
                this.currentSessionId = null;

                // Refresh lists
                this.loadModels();
                this.loadSessions();

                // Show success message
                alert(`Тренировка завершена!\n\nБелые: ${progress.white_wins}\nЧёрные: ${progress.black_wins}\nНичьи: ${progress.draws}`);
            } else if (progress.status === 'error') {
                this.stopProgressPolling();
                document.getElementById('trainingStatus').innerHTML = `
                    <p class="status-text status-error">❌ Ошибка тренировки</p>
                    <p class="error-message">${progress.error_message || 'Unknown error'}</p>
                `;
                document.getElementById('btnStartTraining').disabled = false;
                this.currentSessionId = null;
            }

        } catch (error) {
            console.error('Error updating progress:', error);
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/api/training/models');
            if (!response.ok) return;

            const data = await response.json();
            const modelsList = document.getElementById('modelsList');
            const baseModelSelect = document.getElementById('humanBaseModel');

            if (data.models.length === 0) {
                modelsList.innerHTML = '<p class="empty-text">Нет сохранённых моделей</p>';
                // Reset base model selector
                if (baseModelSelect) {
                    baseModelSelect.innerHTML = '<option value="">🆕 Новая модель (с нуля)</option>';
                }
                return;
            }

            modelsList.innerHTML = data.models.map(model => `
                <div class="model-card ${model.type === 'alphazero' ? 'alphazero-model' : ''}">
                    <div class="model-header">
                        <span class="model-name">📦 ${model.name}</span>
                        <span class="model-badge ${model.type}">${model.type === 'alphazero' ? 'AlphaZero' : 'Gym'}</span>
                        <span class="model-size">${model.size_mb} MB</span>
                    </div>
                    <div class="model-info">
                        <span class="model-arch">${model.architecture || ''}</span>
                        <span class="model-date">${new Date(model.created).toLocaleString('ru-RU')}</span>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-secondary btn-small" onclick="trainingController.loadModel('${model.name}')">
                            Загрузить
                        </button>
                        <button class="btn btn-danger btn-small" onclick="trainingController.deleteModel('${model.name}')">
                            Удалить
                        </button>
                    </div>
                </div>
            `).join('');

            // Also populate base model selector for training (only Gym models)
            if (baseModelSelect) {
                const currentValue = baseModelSelect.value;
                const gymModels = data.models.filter(m => m.type !== 'alphazero');
                baseModelSelect.innerHTML = '<option value="">🆕 Новая модель (с нуля)</option>' +
                    gymModels.map(model => `<option value="${model.name}">🧠 ${model.name}</option>`).join('');
                // Restore previous selection if it still exists
                if (currentValue && gymModels.find(m => m.name === currentValue)) {
                    baseModelSelect.value = currentValue;
                }
            }
            
            // Update active model info for Self-Play
            this.updateSelfPlayModelInfo();

        } catch (error) {
            console.error('Error loading models:', error);
        }
    }
    
    /**
     * Update the Self-Play section to show which model is active
     */
    async updateSelfPlayModelInfo() {
        try {
            const response = await fetch('/api/ai/model-info?level=5');
            if (!response.ok) return;
            
            const modelInfo = await response.json();
            const activeModelEl = document.getElementById('selfPlayActiveModel');
            const warningEl = document.getElementById('selfPlayArchWarning');
            
            if (activeModelEl) {
                const typeIcon = modelInfo.type === 'alphazero' ? '🦾' : '🧠';
                activeModelEl.textContent = `${typeIcon} ${modelInfo.name || 'default'}`;
            }
            
            // Show warning if AlphaZero model is loaded
            if (warningEl) {
                if (modelInfo.type === 'alphazero') {
                    warningEl.style.display = 'block';
                } else {
                    warningEl.style.display = 'none';
                }
            }
        } catch (e) {
            console.error('Error updating Self-Play model info:', e);
        }
    }

    async loadModel(modelName) {
        try {
            const useMcts = document.getElementById('azUseMcts')?.checked || false;
            const response = await fetch(`/api/training/models/${modelName}/load?use_mcts=${useMcts}`, {
                method: 'POST'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to load model');
            }

            const result = await response.json();
            
            // Update Self-Play model info
            this.updateSelfPlayModelInfo();
            
            // Update hidden size input if returned from server
            if (result.hidden_size) {
                const hiddenSizeInput = document.getElementById('hiddenSize');
                if (hiddenSizeInput) {
                    hiddenSizeInput.value = result.hidden_size;
                    console.log(`[Training] Updated hiddenSize input to ${result.hidden_size}`);
                }
            }

            let message = `Модель "${modelName}" успешно загружена!\n\nТип: ${result.type === 'alphazero' ? 'AlphaZero' : 'Gym'}`;
            if (result.type === 'alphazero') {
                message += `\nMCTS: ${result.use_mcts ? 'ВКЛЮЧЕН' : 'ВЫКЛЮЧЕН'}`;
            }
            message += `\n\nТеперь она будет использоваться для игры на уровнях 4-5.`;
            
            alert(message);
            
            // Update active model info in Gemini Battle if available
            if (window.updateGeminiActiveModel) {
                window.updateGeminiActiveModel();
            }

        } catch (error) {
            console.error('Error loading model:', error);
            alert('Ошибка загрузки модели: ' + error.message);
        }
    }

    async deleteModel(modelName) {
        if (!confirm(`Вы уверены, что хотите безвозвратно удалить модель "${modelName}"?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/training/models/${modelName}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to delete model');
            }

            alert(`Модель "${modelName}" успешно удалена.`);
            this.loadModels();

        } catch (error) {
            console.error('Error deleting model:', error);
            alert('Ошибка удаления модели: ' + error.message);
        }
    }

    async loadSessions() {
        try {
            const response = await fetch('/api/training/sessions');
            if (!response.ok) return;

            const data = await response.json();
            const sessionsList = document.getElementById('sessionsList');

            if (data.sessions.length === 0) {
                sessionsList.innerHTML = '<p class="empty-text">Нет завершённых сессий</p>';
                return;
            }

            // Sort by start time, newest first
            const sessions = data.sessions.sort((a, b) => 
                new Date(b.start_time) - new Date(a.start_time)
            );

            sessionsList.innerHTML = sessions.map(session => {
                const statusIcon = session.status === 'completed' ? '✅' : 
                                   session.status === 'running' ? '🔄' : 
                                   session.status === 'error' ? '❌' : '⏸️';
                
                const winRate = session.total_games > 0 
                    ? ((session.white_wins / session.total_games) * 100).toFixed(1)
                    : 0;

                return `
                    <div class="session-card">
                        <div class="session-header">
                            <span class="session-status">${statusIcon} ${session.status}</span>
                            <span class="session-id">${session.session_id}</span>
                        </div>
                        <div class="session-stats">
                            <span>Игр: ${session.games_completed}/${session.total_games}</span>
                            <span>W: ${session.white_wins} | B: ${session.black_wins} | D: ${session.draws}</span>
                            <span>Win Rate (W): ${winRate}%</span>
                            <span>Avg Steps: ${session.avg_steps.toFixed(1)}</span>
                        </div>
                        <div class="session-time">
                            ${new Date(session.start_time).toLocaleString('ru-RU')}
                        </div>
                    </div>
                `;
            }).join('');

        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    }
}

// Initialize training controller
const trainingController = new TrainingController();

