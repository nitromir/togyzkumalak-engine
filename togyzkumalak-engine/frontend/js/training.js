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
        // ... (existing buttons)
        document.getElementById('btnStartTraining').addEventListener('click', () => this.startTraining());
        
        // SYNC buttons
        document.getElementById('btnSaveSyncConfig')?.addEventListener('click', () => this.saveSyncConfig());
        document.getElementById('btnRestartSync')?.addEventListener('click', () => this.restartSync());
        
        // ... (rest of init)
        this.initPROBS();
        this.initAlphaZero();
        this.initSync();
        this.initLogModal(); // Добавляем инициализацию модалки логов
        
        // Set up periodic refresh when training mode is active
        setInterval(() => {
            if (!document.getElementById('trainingMode').classList.contains('hidden')) {
                this.loadModels();
                this.loadSessions();
                this.loadDataStats();
                this.updateSyncStatus();
            }
        }, 5000);
    }

    /**
     * SYNC Methods
     */
    initSync() {
        this.updateSyncStatus();
    }

    async updateSyncStatus() {
        try {
            const response = await fetch('/api/sync/status');
            if (!response.ok) return;
            const data = await response.json();
            
            const indicator = document.getElementById('syncIndicator');
            const lastTime = document.getElementById('syncLastTime');
            const urlInput = document.getElementById('syncRemoteUrl');
            const portsInput = document.getElementById('syncPorts');
            
            if (indicator) {
                if (data.is_running && data.status.status === 'active') {
                    indicator.className = 'sync-indicator active';
                    indicator.querySelector('.text').textContent = `Синхронизация: АКТИВНА (${data.status.server})`;
                } else {
                    indicator.className = 'sync-indicator disconnected';
                    indicator.querySelector('.text').textContent = data.is_running 
                        ? 'Синхронизация: ОЖИДАНИЕ СЕРВЕРА' 
                        : 'Синхронизация: ВЫКЛЮЧЕНА';
                }
            }
            
            if (lastTime && data.status.last_sync) {
                const date = new Date(data.status.last_sync);
                lastTime.textContent = `Последняя проверка: ${date.toLocaleTimeString()}`;
            }
            
            // Fill inputs if they are empty
            if (urlInput && !urlInput.value && data.config.remote_url) {
                urlInput.value = data.config.remote_url;
            }
            if (portsInput && !portsInput.value && data.config.ports) {
                portsInput.value = data.config.ports.join(', ');
            }
        } catch (e) {
            console.error('Error updating sync status:', e);
        }
    }

    async saveSyncConfig() {
        const url = document.getElementById('syncRemoteUrl').value;
        const portsStr = document.getElementById('syncPorts').value;
        const ports = portsStr.split(',').map(p => parseInt(p.trim())).filter(p => !isNaN(p));
        
        try {
            const response = await fetch('/api/sync/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    remote_url: url,
                    ports: ports,
                    interval: 30,
                    enabled: true
                })
            });
            
            if (response.ok) {
                alert('✅ Настройки синхронизации сохранены и процесс запущен!');
                this.updateSyncStatus();
            }
        } catch (e) {
            alert('❌ Ошибка сохранения конфига: ' + e.message);
        }
    }

    async restartSync() {
        try {
            const response = await fetch('/api/sync/restart', { method: 'POST' });
            if (response.ok) {
                alert('🔄 Процесс синхронизации перезапущен!');
                this.updateSyncStatus();
            }
        } catch (e) {
            alert('❌ Ошибка перезапуска: ' + e.message);
        }
    }

    /**
     * Handle file upload for training data
     */
    async handleFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        
        const statusEl = document.getElementById('uploadStatus');
        statusEl.textContent = '⏳ Загрузка...';
        statusEl.className = 'upload-status uploading';
        
        let successCount = 0;
        let errorCount = 0;
        
        for (const file of files) {
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/api/data/upload-training-file', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('Uploaded:', result);
                    successCount++;
                } else {
                    const error = await response.json();
                    console.error('Upload error:', error);
                    errorCount++;
                }
            } catch (error) {
                console.error('Upload failed:', error);
                errorCount++;
            }
        }
        
        // Update status
        if (errorCount === 0) {
            statusEl.textContent = `✅ Загружено ${successCount} файл(ов)`;
            statusEl.className = 'upload-status success';
        } else {
            statusEl.textContent = `⚠️ Загружено: ${successCount}, ошибок: ${errorCount}`;
            statusEl.className = 'upload-status error';
        }
        
        // Refresh file list
        this.loadTrainingFiles();
        this.loadTrainingFilesStatus();
        
        // Clear file input
        event.target.value = '';
        
        // Hide status after 5 seconds
        setTimeout(() => {
            statusEl.textContent = '';
            statusEl.className = 'upload-status';
        }, 5000);
    }
    
    /**
     * Load detailed training files status
     */
    async loadTrainingFilesStatus() {
        try {
            const response = await fetch('/api/data/training-files-status');
            if (!response.ok) return;
            
            const data = await response.json();
            console.log('Training files status:', data);
            
            // Update bootstrap ready indicator
            if (data.bootstrap_ready) {
                const checkbox = document.getElementById('azUseBootstrap');
                if (checkbox && !checkbox.checked) {
                    console.log('Bootstrap data available, checkbox can be enabled');
                }
            }
            
            return data;
        } catch (error) {
            console.error('Error loading training files status:', error);
        }
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

    initLogModal() {
        const btnCloseLogs = document.getElementById('btnCloseTrainingLogs');
        const btnRefreshLogs = document.getElementById('btnRefreshLogs');
        const btnCopyLogs = document.getElementById('btnCopyLogs');
        const logsModal = document.getElementById('trainingLogsModal');
        
        btnCloseLogs?.addEventListener('click', () => {
            if (logsModal) logsModal.classList.add('hidden');
        });
        
        btnRefreshLogs?.addEventListener('click', () => {
            const titleEl = document.getElementById('trainingLogsTitle');
            if (titleEl && titleEl.textContent.includes('PROBS')) {
                this.loadPROBSLogs();
            } else {
                this.loadTrainingLogs();
            }
        });
        
        btnCopyLogs?.addEventListener('click', () => {
            const content = document.getElementById('trainingLogsContent');
            if (content) {
                navigator.clipboard.writeText(content.textContent).then(() => {
                    this.showNotification('📋 Логи скопированы в буфер обмена!');
                }).catch(() => {
                    alert('Не удалось скопировать логи');
                });
            }
        });
        
        logsModal?.addEventListener('click', (e) => {
            if (e.target === logsModal) {
                logsModal.classList.add('hidden');
            }
        });
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
        const btnOptimal = document.getElementById('btnLoadOptimalConfig');
        
        btnStart?.addEventListener('click', () => this.startAlphaZero());
        btnStop?.addEventListener('click', () => this.stopAlphaZero());
        btnOptimal?.addEventListener('click', () => this.loadOptimalConfig());
        
        // Training logs button (Always visible one)
        const btnShowLogsAlways = document.getElementById('btnShowTrainingLogsAlways');
        btnShowLogsAlways?.addEventListener('click', () => this.showTrainingLogs());
        
        // Training logs button (Progress section one)
        const btnShowLogs = document.getElementById('btnShowTrainingLogs');
        btnShowLogs?.addEventListener('click', () => this.showTrainingLogs());
        
        this.initAlphaZeroCharts();
        
        // Optimal config modal handlers
        const btnApplyOptimal = document.getElementById('btnApplyOptimalConfig');
        const btnCancelOptimal = document.getElementById('btnCancelOptimalConfig');
        const optimalModal = document.getElementById('optimalConfigModal');
        
        btnApplyOptimal?.addEventListener('click', () => {
            const hours = parseFloat(document.getElementById('hoursInput')?.value) || 1;
            this._applyOptimalConfig(hours);
        });
        
        btnCancelOptimal?.addEventListener('click', () => {
            if (optimalModal) optimalModal.classList.add('hidden');
        });
        
        // Close modal on outside click
        optimalModal?.addEventListener('click', (e) => {
            if (e.target === optimalModal) {
                optimalModal.classList.add('hidden');
            }
        });
        
        // Close on Enter key
        document.getElementById('hoursInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const hours = parseFloat(e.target.value) || 1;
                this._applyOptimalConfig(hours);
            }
        });
        
        this.initAlphaZeroCharts();
        this.loadAlphaZeroMetrics();  // Load last training metrics
        this.loadGpuInfo();  // Load GPU information
        this.checkActiveAlphaZeroTasks(); // Check if training is already running
        
        // System update button
        const btnUpdate = document.getElementById('btnUpdateAndRestart');
        btnUpdate?.addEventListener('click', () => this.updateAndRestart());
        this.loadGitStatus();  // Load git status on init
    }
    
    /**
     * Load GPU information from server
     */
    async loadGpuInfo() {
        try {
            const response = await fetch('/api/training/alphazero/gpu-info');
            if (!response.ok) return;
            
            const data = await response.json();
            const gpuInfoEl = document.getElementById('gpuInfo');
            
            if (gpuInfoEl) {
                if (data.cuda_available && data.gpu_count > 0) {
                    const gpuNames = data.gpus.map(g => `${g.name} (${g.memory_total_gb}GB)`).join(', ');
                    gpuInfoEl.innerHTML = `
                        <span class="gpu-status good">🎮 ${data.gpu_count}x GPU: ${gpuNames}</span>
                    `;
                } else {
                    gpuInfoEl.innerHTML = `<span class="gpu-status warning">⚠️ CPU режим (CUDA недоступен)</span>`;
                }
            }
            
            // Update optimal config button
            if (data.gpu_count > 0) {
                const btnOptimal = document.getElementById('btnLoadOptimalConfig');
                if (btnOptimal) {
                    btnOptimal.textContent = `⚡ Авто-конфиг для ${data.gpu_count} GPU`;
                }
            }
            
        } catch (e) {
            console.error('Error loading GPU info:', e);
        }
    }

    /**
     * Check if there are any active AlphaZero training tasks running
     */
    async checkActiveAlphaZeroTasks() {
        try {
            const response = await fetch('/api/training/alphazero/sessions');
            if (!response.ok) return;
            
            const tasks = await response.json();
            
            // Find first running task
            const activeTaskId = Object.keys(tasks).find(id => tasks[id].status === 'running');
            
            if (activeTaskId) {
                console.log('Detected active AlphaZero task:', activeTaskId);
                this.azTaskId = activeTaskId;
                
                // Show progress section
                const progressSection = document.getElementById('azProgressSection');
                if (progressSection) progressSection.classList.remove('hidden');
                
                const btnStop = document.getElementById('btnStopAlphaZero');
                if (btnStop) btnStop.classList.remove('hidden');
                
                const btnStart = document.getElementById('btnStartAlphaZero');
                if (btnStart) btnStart.disabled = true;
                
                this.startAlphaZeroPolling();
            }
        } catch (error) {
            console.error('Error checking active AlphaZero tasks:', error);
        }
    }
    
    /**
     * Load optimal training configuration based on available GPUs
     */
    async loadOptimalConfig() {
        // Show modal for hours input
        const modal = document.getElementById('optimalConfigModal');
        if (modal) {
            modal.classList.remove('hidden');
            
            // Focus on input
            setTimeout(() => {
                const input = document.getElementById('hoursInput');
                if (input) input.focus();
            }, 100);
        } else {
            // Fallback to prompt if modal not found
            this._applyOptimalConfig(1);
        }
    }
    
    /**
     * Apply optimal config with given hours
     */
    async _applyOptimalConfig(hours) {
        try {
            const response = await fetch(`/api/training/alphazero/optimal-config?hours=${hours}&gpus=16`);
            if (!response.ok) throw new Error('Failed to load optimal config');
            
            const data = await response.json();
            const config = data.recommended_config || data;
            
            // Apply config to form
            const itersEl = document.getElementById('azIters');
            const epsEl = document.getElementById('azEps');
            const simsEl = document.getElementById('azSims');
            const cpuctEl = document.getElementById('azCpuct');
            const batchEl = document.getElementById('azBatchSize');
            const hiddenEl = document.getElementById('azHiddenSize');
            const bootstrapEl = document.getElementById('azUseBootstrap');
            
            if (itersEl) itersEl.value = config.numIters || config.numIters || 250;
            if (epsEl) epsEl.value = config.numEps || config.numEps || 200;
            if (simsEl) simsEl.value = config.numMCTSSims || config.numMCTSSims || 200;
            if (cpuctEl) cpuctEl.value = config.cpuct || 1.0;
            if (batchEl) batchEl.value = config.batch_size || 4096;
            if (hiddenEl) hiddenEl.value = config.hidden_size || 512;
            if (bootstrapEl) bootstrapEl.checked = config.use_bootstrap !== false;
            
            // Close modal
            const modal = document.getElementById('optimalConfigModal');
            if (modal) modal.classList.add('hidden');
            
            // Show success message
            const tips = data.tips || [];
            const timeEst = data.estimated_time_minutes || config.estimated_time_min || 60;
            const message = `🚀 Оптимальная конфигурация установлена!\n\n` +
                          `Итерации: ${config.numIters || 250}\n` +
                          `Игр/итер: ${config.numEps || 200}\n` +
                          `MCTS: ${config.numMCTSSims || 200}\n` +
                          `Batch: ${config.batch_size || 4096}\n` +
                          `Ожидаемое время: ~${Math.round(timeEst)} мин\n\n` +
                          (tips.length > 0 ? tips.join('\n') : '');
            
            alert(message);
            
        } catch (e) {
            console.error('Error loading optimal config:', e);
            alert('Ошибка загрузки оптимальной конфигурации: ' + e.message);
            
            // Close modal on error
            const modal = document.getElementById('optimalConfigModal');
            if (modal) modal.classList.add('hidden');
        }
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
     * Render list of best checkpoints with download buttons
     */
    renderCheckpointsList(checkpoints) {
        const container = document.getElementById('azCheckpointsList');
        if (!container || !checkpoints.length) return;
        
        container.innerHTML = checkpoints.slice(0, 10).map((cp, i) => `
            <div class="checkpoint-item ${i === 0 ? 'best' : ''} ${cp.accepted ? 'accepted' : 'rejected'}">
                <div class="cp-main-info">
                    <span class="cp-rank">#${i + 1}</span>
                    <span class="cp-iter">iter ${cp.iteration}</span>
                    <span class="cp-time">${cp.timestamp ? new Date(cp.timestamp).toLocaleString('ru-RU', {hour:'2-digit', minute:'2-digit', day:'2-digit', month:'2-digit'}) : ''}</span>
                </div>
                <div class="cp-metrics">
                    <span class="cp-loss" title="Policy Loss">📉 ${cp.policy_loss?.toFixed(3) || '?'}</span>
                    <span class="cp-value" title="Value Loss">📊 ${cp.value_loss?.toFixed(3) || '?'}</span>
                    <span class="cp-winrate" title="Win Rate">${cp.win_rate ? (cp.win_rate * 100).toFixed(0) + '%' : '-'}</span>
                </div>
                <div class="cp-actions">
                    <button class="btn btn-tiny btn-primary" onclick="trainingController.loadAlphaZeroCheckpoint('${cp.filename}')" title="Загрузить для игры">
                        📦
                    </button>
                    <button class="btn btn-tiny btn-secondary" onclick="trainingController.downloadCheckpoint('${cp.filename}')" title="Скачать файл">
                        💾
                    </button>
                </div>
            </div>
        `).join('');
    }
    
    /**
     * Download a checkpoint file to local machine
     */
    async downloadCheckpoint(filename) {
        try {
            const modelName = filename.replace('.pth.tar', '');
            const url = `/api/training/alphazero/checkpoints/${modelName}/download`;
            
            // Create temporary link and click it
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showNotification(`⬇️ Скачивание: ${filename}`);
        } catch (e) {
            console.error('Error downloading checkpoint:', e);
            alert('Ошибка скачивания: ' + e.message);
        }
    }
    
    /**
     * Load all checkpoints with detailed metrics for visualization
     */
    async loadAllCheckpointsMetrics() {
        try {
            const response = await fetch('/api/training/alphazero/checkpoints');
            if (!response.ok) return [];
            
            const data = await response.json();
            return data.checkpoints || [];
        } catch (e) {
            console.error('Error loading checkpoints:', e);
            return [];
        }
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
        
        // Получаем значения отдельно
        const numIters = parseInt(document.getElementById('azIters').value) || 100;
        const numEps = parseInt(document.getElementById('azEps').value) || 100;
        const numMCTSSims = parseInt(document.getElementById('azSims').value) || 100;
        const cpuct = parseFloat(document.getElementById('azCpuct').value) || 1.0;
        const batch_size = parseInt(document.getElementById('azBatchSize')?.value) || 256;
        const hidden_size = parseInt(document.getElementById('azHiddenSize')?.value) || 256;
        const epochs = parseInt(document.getElementById('azEpochs')?.value) || 10;
        const use_multiprocessing = document.getElementById('azParallel')?.checked ?? true;
        
        const config = {
            numIters: numIters,
            numEps: numEps,
            numMCTSSims: numMCTSSims,
            cpuct: cpuct,
            batch_size: batch_size,
            hidden_size: hidden_size,
            epochs: epochs,
            use_bootstrap: useBootstrap,
            use_multiprocessing: use_multiprocessing,
            save_every_n_iters: Math.max(1, Math.floor(numIters / 20) || 5)
        };

        try {
            const btnStart = document.getElementById('btnStartAlphaZero');
            if (!btnStart) {
                throw new Error('Кнопка запуска не найдена');
            }
            
            btnStart.disabled = true;
            btnStart.textContent = '⏳ Запуск...';
            
            console.log('Отправляю конфигурацию:', config);
            
            const response = await fetch('/api/training/alphazero/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Ошибка ответа сервера:', response.status, errorText);
                throw new Error(`Сервер вернул ошибку ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            console.log('Ответ сервера:', data);
            
            if (!data.task_id) {
                throw new Error('Сервер не вернул task_id');
            }
            
            this.azTaskId = data.task_id;

            const progressSection = document.getElementById('azProgressSection');
            if (progressSection) progressSection.classList.remove('hidden');
            
            const btnStop = document.getElementById('btnStopAlphaZero');
            if (btnStop) btnStop.classList.remove('hidden');
            
            // Show live checkpoints section
            const liveCheckpointsEl = document.getElementById('azLiveCheckpoints');
            if (liveCheckpointsEl) liveCheckpointsEl.classList.remove('hidden');
            
            this.startAlphaZeroPolling();
            
            // Показываем уведомление об успешном запуске
            alert('✅ Обучение AlphaZero запущено! Task ID: ' + data.task_id);
            
        } catch (error) {
            console.error('Error starting AlphaZero:', error);
            alert('Ошибка запуска AlphaZero: ' + error.message + '\n\nПроверь консоль браузера (F12) для деталей.');
            const btnStart = document.getElementById('btnStartAlphaZero');
            if (btnStart) {
                btnStart.disabled = false;
                btnStart.textContent = '🚀 Запустить AlphaZero';
            }
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
            
            // Update progress bar
            document.getElementById('azProgressBar').style.width = `${task.progress}%`;
            document.getElementById('azCurrentIter').textContent = `${task.current_iteration} / ${task.total_iterations}`;
            document.getElementById('azStatusText').textContent = task.status;
            
            // Update elapsed time
            const elapsedEl = document.getElementById('azElapsedTime');
            if (elapsedEl && task.elapsed_time) {
                const mins = Math.floor(task.elapsed_time / 60);
                const secs = Math.floor(task.elapsed_time % 60);
                elapsedEl.textContent = `${mins}м ${secs}с`;
            }
            
            // Update GPU info
            const gpuEl = document.getElementById('azGpuUsed');
            if (gpuEl && task.gpus !== undefined) {
                gpuEl.textContent = task.gpus > 0 ? `${task.gpus}x GPU` : 'CPU';
            }

            // Update charts with detailed metrics
            if (task.metrics && task.metrics.length > 0) {
                // Loss chart
                this.azLossChart.data.labels = task.metrics.map(m => m.iteration);
                this.azLossChart.data.datasets[0].data = task.metrics.map(m => m.policy_loss);
                if (this.azLossChart.data.datasets[1]) {
                    this.azLossChart.data.datasets[1].data = task.metrics.map(m => m.value_loss);
                }
                this.azLossChart.update('none');
                
                // Win rate chart
                this.azEloChart.data.labels = task.metrics.map(m => m.iteration);
                this.azEloChart.data.datasets[0].data = task.metrics.map(m => (m.win_rate || 0) * 100);
                this.azEloChart.update('none');
                
                // Update live metrics display
                const lastMetrics = task.metrics[task.metrics.length - 1];
                if (lastMetrics) {
                    const policyEl = document.getElementById('lastPolicyLoss');
                    const valueEl = document.getElementById('lastValueLoss');
                    const winEl = document.getElementById('lastWinRate');
                    
                    if (policyEl) policyEl.textContent = lastMetrics.policy_loss?.toFixed(3) || '-';
                    if (valueEl) valueEl.textContent = lastMetrics.value_loss?.toFixed(3) || '-';
                    if (winEl) winEl.textContent = ((lastMetrics.win_rate || 0) * 100).toFixed(0) + '%';
                }
                
                // Refresh checkpoints list to show new ones
                this.loadAlphaZeroMetrics();
            }

            if (task.status === 'completed' || task.status === 'error' || task.status === 'stopped') {
                clearInterval(this.azPollInterval);
                document.getElementById('btnStartAlphaZero').disabled = false;
                document.getElementById('btnStopAlphaZero').classList.add('hidden');
                
                // Final refresh of checkpoints
                this.loadAlphaZeroMetrics();
                
                if (task.status === 'completed') {
                    this.showNotification('🦾 AlphaZero: Обучение завершено! Чекпоинты готовы к скачиванию.');
                } else if (task.status === 'error') {
                    this.showNotification('❌ AlphaZero: Ошибка обучения. Проверьте логи.');
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
        
        if (!ctxLoss || !ctxElo) {
            console.warn('[Training] AlphaZero chart canvases not found');
            return;
        }

        // Destroy existing charts if any
        if (this.azLossChart) this.azLossChart.destroy();
        if (this.azEloChart) this.azEloChart.destroy();

        this.azLossChart = new Chart(ctxLoss, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'Policy Loss', data: [], borderColor: '#ff0055', tension: 0.3, borderWidth: 2 },
                    { label: 'Value Loss', data: [], borderColor: '#f59e0b', tension: 0.3, borderWidth: 2 }
                ]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(100,116,139,0.2)' } },
                    x: { grid: { color: 'rgba(100,116,139,0.2)' } }
                },
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
                }
            }
        });

        this.azEloChart = new Chart(ctxElo, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [{ label: 'Win Rate %', data: [], borderColor: '#22c55e', tension: 0.3, borderWidth: 2, fill: true, backgroundColor: 'rgba(34,197,94,0.1)' }]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 100, grid: { color: 'rgba(100,116,139,0.2)' } },
                    x: { grid: { color: 'rgba(100,116,139,0.2)' } }
                },
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
                }
            }
        });
        
        console.log('[Training] AlphaZero charts initialized');
    }

    /**
     * PROBS Training Methods
     */
    initPROBS() {
        this.probsTaskId = null;
        this.probsPollInterval = null;
        
        const btnStart = document.getElementById('btnStartPROBS');
        const btnStop = document.getElementById('btnStopPROBS');
        const btnShowLogs = document.getElementById('btnShowPROBSLogs');
        const btnMonster = document.getElementById('btnMonsterConfig');
        
        btnStart?.addEventListener('click', () => this.startPROBS());
        btnStop?.addEventListener('click', () => this.stopPROBS());
        btnShowLogs?.addEventListener('click', () => this.showPROBSLogs());
        btnMonster?.addEventListener('click', () => this.applyMonsterConfig());
        
        this.loadPROBSCheckpoints();
        this.checkActivePROBSTasks();
    }
    
    showPROBSLogs() {
        const modal = document.getElementById('trainingLogsModal'); // Reuse AlphaZero modal for now but change title
        const titleEl = document.getElementById('trainingLogsTitle');
        if (modal) {
            if (titleEl) titleEl.textContent = '📋 Логи Обучения PROBS';
            modal.classList.remove('hidden');
            this.loadPROBSLogs();
        }
    }

    applyMonsterConfig() {
        if (!confirm('🔥 Применить настройки для МОНСТР-СЕРВЕРА (128 ядер, 4x GPU)?\n\nЭто радикально увеличит объемы данных и параллелизм.')) {
            return;
        }

        const settings = {
            'probsIters': 300,           // 300 итераций
            'probsVEpisodes': 8000,      // 8K партий для Self-play (GPU inference)
            'probsQEpisodes': 4000,      // 4K эпизодов Q-train (CPU + GPU)
            'probsMemEpisodes': 80000,   // Буфер памяти
            'probsBatchSize': 2048,      // Большой батч для GPU (было 1024)
            'probsQCalls': 50,           // Глубина поиска Q
            'probsMaxDepth': 100,        // Макс глубина дерева
            'probsThreads': 16,          // 16 потоков Self-play (было 8)
            'probsEvalGames': 50,        // 50 игр для оценки
            'probsProcesses': 64,        // 64 воркера Q-train (было 32)
            'probsDevice': 'cuda'
        };

        for (const [id, value] of Object.entries(settings)) {
            const el = document.getElementById(id);
            if (el) el.value = value;
        }

        // Включаем Boosting
        const boost = document.getElementById('probsUseBoost');
        if (boost) boost.checked = true;

        this.showNotification('🔥 Настройки МОНСТРА применены! Не забудьте нажать "Запустить PROBS".');
    }

    async loadPROBSLogs() {
        const contentEl = document.getElementById('trainingLogsContent');
        if (!contentEl) return;
        
        // Show current task status at top of logs if active
        let statusPrefix = "";
        if (this.probsTaskId) {
            try {
                const statusRes = await fetch(`/api/training/probs/sessions/${this.probsTaskId}`);
                if (statusRes.ok) {
                    const task = await statusRes.json();
                    statusPrefix = `СТАТУС ЗАДАЧИ: ${task.status.toUpperCase()}\n` +
                                   `Прогресс: ${task.progress.toFixed(1)}%\n` +
                                   `Итерация: ${task.current_iteration} / ${task.total_iterations}\n` +
                                   `----------------------------------------------------------------\n\n`;
                }
            } catch (e) {}
        }

        try {
            const response = await fetch('/api/training/probs/logs?lines=300');
            const data = await response.json();
            
            if (data.status === 'ok') {
                contentEl.textContent = statusPrefix + data.output.join('\n');
                // Scroll to bottom
                contentEl.scrollTop = contentEl.scrollHeight;
            } else if (data.status === 'no_log') {
                contentEl.textContent = statusPrefix + 'Логов пока нет. Запустите обучение.';
            } else {
                contentEl.textContent = statusPrefix + 'Ошибка загрузки логов: ' + (data.error || 'неизвестная ошибка');
            }
        } catch (error) {
            contentEl.textContent = statusPrefix + 'Ошибка сети при загрузке логов.';
        }
    }
    
    async checkActivePROBSTasks() {
        try {
            const response = await fetch('/api/training/probs/sessions');
            if (!response.ok) return;
            
            const data = await response.json();
            const sessions = data.sessions || {};
            
            // Find first running task
            const activeTaskId = Object.keys(sessions).find(id => sessions[id].status === 'running');
            
            if (activeTaskId) {
                console.log('Detected active PROBS task:', activeTaskId);
                this.probsTaskId = activeTaskId;
                
                const progressSection = document.getElementById('probsProgressSection');
                if (progressSection) progressSection.classList.remove('hidden');
                
                const btnStop = document.getElementById('btnStopPROBS');
                if (btnStop) btnStop.classList.remove('hidden');
                
                const btnStart = document.getElementById('btnStartPROBS');
                if (btnStart) btnStart.disabled = true;
                
                this.startPROBSPolling();
            }
        } catch (error) {
            console.error('Error checking active PROBS tasks:', error);
        }
    }
    
    async startPROBS() {
        const config = {
            n_high_level_iterations: parseInt(document.getElementById('probsIters')?.value) || 100,
            v_train_episodes: parseInt(document.getElementById('probsVEpisodes')?.value) || 500,
            q_train_episodes: parseInt(document.getElementById('probsQEpisodes')?.value) || 250,
            mem_max_episodes: parseInt(document.getElementById('probsMemEpisodes')?.value) || 10000,
            train_batch_size: parseInt(document.getElementById('probsBatchSize')?.value) || 64,
            num_q_s_a_calls: parseInt(document.getElementById('probsQCalls')?.value) || 30,
            max_depth: parseInt(document.getElementById('probsMaxDepth')?.value) || 50,
            self_play_threads: parseInt(document.getElementById('probsThreads')?.value) || 4,
            sub_processes_cnt: parseInt(document.getElementById('probsProcesses')?.value) || 4,
            evaluate_n_games: parseInt(document.getElementById('probsEvalGames')?.value) || 20,
            device: document.getElementById('probsDevice')?.value || 'cpu',
            use_boost: document.getElementById('probsUseBoost')?.checked || false,
            initial_checkpoint: document.getElementById('probsInitialCheckpoint')?.value || null
        };
        
        try {
            const btnStart = document.getElementById('btnStartPROBS');
            if (btnStart) {
                btnStart.disabled = true;
                btnStart.textContent = '⏳ Запуск...';
            }
            
            const response = await fetch('/api/training/probs/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Сервер вернул ошибку ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            this.probsTaskId = data.task_id;
            
            const progressSection = document.getElementById('probsProgressSection');
            if (progressSection) progressSection.classList.remove('hidden');
            
            const btnStop = document.getElementById('btnStopPROBS');
            if (btnStop) btnStop.classList.remove('hidden');
            
            this.startPROBSPolling();
            
            alert('✅ Обучение PROBS запущено! Task ID: ' + data.task_id);
            
        } catch (error) {
            console.error('Error starting PROBS:', error);
            alert('Ошибка запуска PROBS: ' + error.message);
            const btnStart = document.getElementById('btnStartPROBS');
            if (btnStart) {
                btnStart.disabled = false;
                btnStart.textContent = '🚀 Запустить PROBS';
            }
        }
    }
    
    startPROBSPolling() {
        if (this.probsPollInterval) clearInterval(this.probsPollInterval);
        this.probsPollInterval = setInterval(() => this.updatePROBSProgress(), 2000);
    }
    
    async updatePROBSProgress() {
        if (!this.probsTaskId) return;
        
        try {
            const response = await fetch(`/api/training/probs/sessions/${this.probsTaskId}`);
            if (!response.ok) return;
            
            const task = await response.json();
            
            // Update progress bar
            const progressBar = document.getElementById('probsProgressBar');
            const currentIter = document.getElementById('probsCurrentIter');
            const statusText = document.getElementById('probsStatusText');
            const elapsedTime = document.getElementById('probsElapsedTime');
            
            if (progressBar) progressBar.style.width = `${task.progress}%`;
            if (currentIter) currentIter.textContent = `${task.current_iteration} / ${task.total_iterations}`;
            if (statusText) statusText.textContent = task.status;
            
            // Auto-refresh logs if modal is open
            const modal = document.getElementById('trainingLogsModal');
            const titleEl = document.getElementById('trainingLogsTitle');
            if (modal && !modal.classList.contains('hidden') && titleEl && titleEl.textContent.includes('PROBS')) {
                this.loadPROBSLogs();
            }

            if (elapsedTime && task.elapsed_time) {
                const mins = Math.floor(task.elapsed_time / 60);
                const secs = Math.floor(task.elapsed_time % 60);
                elapsedTime.textContent = `${mins}м ${secs}с`;
            }
            
            if (task.status === 'completed' || task.status === 'error' || task.status === 'stopped') {
                clearInterval(this.probsPollInterval);
                document.getElementById('btnStartPROBS').disabled = false;
                document.getElementById('btnStartPROBS').textContent = '🚀 Запустить PROBS';
                document.getElementById('btnStopPROBS')?.classList.add('hidden');
                
                this.loadPROBSCheckpoints();
                
                if (task.status === 'completed') {
                    this.showNotification('🎯 PROBS: Обучение завершено! Чекпоинты готовы.');
                } else if (task.status === 'error') {
                    this.showNotification('❌ PROBS: Ошибка обучения. ' + (task.error || ''));
                }
            }
        } catch (error) {
            console.error('Error polling PROBS:', error);
        }
    }
    
    async stopPROBS() {
        if (!this.probsTaskId) return;
        try {
            const response = await fetch(`/api/training/probs/sessions/${this.probsTaskId}/stop`, { method: 'POST' });
            if (response.ok) {
                const statusText = document.getElementById('probsStatusText');
                if (statusText) statusText.textContent = 'stopping';
                this.showNotification('🛑 Запрос на остановку отправлен. Процесс завершится по окончании текущего шага.');
            }
        } catch (error) {
            console.error('Error stopping PROBS:', error);
            alert('Ошибка при остановке: ' + error.message);
        }
    }
    
    async loadPROBSCheckpoints() {
        try {
            const response = await fetch('/api/training/probs/checkpoints');
            if (!response.ok) return;
            
            const data = await response.json();
            const container = document.getElementById('probsCheckpointsList');
            
            if (!container) return;
            
            if (!data.checkpoints || data.checkpoints.length === 0) {
                container.innerHTML = '<p class="empty-text">Нет сохранённых чекпоинтов PROBS</p>';
                return;
            }
            
            container.innerHTML = data.checkpoints.slice(0, 10).map((cp, i) => `
                <div class="checkpoint-item ${cp.is_best ? 'best' : ''}">
                    <div class="cp-main-info">
                        <span class="cp-rank">#${i + 1}</span>
                        <span class="cp-name" style="${cp.is_best ? 'color: #ffcc00; font-weight: bold;' : ''}">${cp.filename} ${cp.is_best ? '👑 BEST' : ''}</span>
                        <span class="cp-time">${new Date(cp.timestamp).toLocaleString('ru-RU', {hour:'2-digit', minute:'2-digit', day:'2-digit', month:'2-digit'})}</span>
                    </div>
                    <div class="cp-metrics">
                        <span class="cp-size">${cp.size_mb} MB</span>
                    </div>
                    <div class="cp-actions">
                        <button class="btn btn-tiny btn-primary" onclick="trainingController.loadPROBSCheckpoint('${cp.filename}')" title="Загрузить для игры">
                            📦
                        </button>
                        <button class="btn btn-tiny btn-secondary" onclick="trainingController.downloadPROBSCheckpoint('${cp.filename}')" title="Скачать файл">
                            💾
                        </button>
                    </div>
                </div>
            `).join('');

            // Populate checkpoint selector for continuation
            const select = document.getElementById('probsInitialCheckpoint');
            if (select) {
                const currentValue = select.value;
                select.innerHTML = '<option value="">🆕 Новая модель (с нуля)</option>' + 
                    data.checkpoints.map(cp => `<option value="${cp.filename}">${cp.is_best ? '⭐ ' : ''}${cp.filename}</option>`).join('');
                select.value = currentValue;
            }
            
        } catch (error) {
            console.error('Error loading PROBS checkpoints:', error);
        }
    }
    
    async loadPROBSCheckpoint(filename) {
        try {
            const checkpointName = filename.replace('.ckpt', '');
            const response = await fetch(`/api/training/probs/checkpoints/${checkpointName}/load`, {
                method: 'POST'
            });
            
            if (response.ok) {
                alert(`✅ Загружен PROBS чекпоинт: ${filename}\n\nТеперь можно играть против PROBS AI (уровень 7)!`);
                this.loadPROBSCheckpoints();
            } else {
                throw new Error('Failed to load checkpoint');
            }
        } catch (e) {
            console.error('Error loading PROBS checkpoint:', e);
            alert('Ошибка загрузки чекпоинта: ' + e.message);
        }
    }
    
    async downloadPROBSCheckpoint(filename) {
        try {
            const checkpointName = filename.replace('.ckpt', '');
            const url = `/api/training/probs/checkpoints/${checkpointName}/download`;
            
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showNotification(`⬇️ Скачивание: ${filename}`);
        } catch (e) {
            console.error('Error downloading PROBS checkpoint:', e);
            alert('Ошибка скачивания: ' + e.message);
        }
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
    
    /**
     * Load git status and display update info
     */
    async loadGitStatus() {
        try {
            const response = await fetch('/api/system/git-status');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            
            const statusInfo = document.getElementById('gitStatusInfo');
            if (!statusInfo) return;
            
            // Check if data exists and has is_git_repo property
            if (!data || !data.is_git_repo) {
                statusInfo.innerHTML = `<span style="color: var(--text-secondary);">Не git репозиторий</span>`;
                return;
            }
            
            const lastCommit = data.last_commit;
            const commitsBehind = data.commits_behind || 0;
            
            let statusHtml = '';
            if (lastCommit) {
                statusHtml += `<div style="margin-bottom: 5px;">`;
                statusHtml += `<strong>Последний коммит:</strong> ${lastCommit.hash} - ${lastCommit.message}<br>`;
                statusHtml += `<small>${new Date(lastCommit.date).toLocaleString('ru-RU')}</small>`;
                statusHtml += `</div>`;
            }
            
            if (commitsBehind > 0) {
                statusHtml += `<div style="color: #ffa500; font-weight: bold;">⚠️ Доступно ${commitsBehind} обновлений на GitHub</div>`;
            } else {
                statusHtml += `<div style="color: #4caf50;">✅ Код актуален</div>`;
            }
            
            statusInfo.innerHTML = statusHtml;
            
        } catch (error) {
            console.error('Error loading git status:', error);
            const statusInfo = document.getElementById('gitStatusInfo');
            if (statusInfo) {
                statusInfo.innerHTML = `<span style="color: var(--text-secondary);">Статус недоступен</span>`;
            }
        }
    }
    
    /**
     * Update code from GitHub and restart server
     */
    async updateAndRestart() {
        const btn = document.getElementById('btnUpdateAndRestart');
        const statusInfo = document.getElementById('gitStatusInfo');
        
        if (!confirm('Обновить код с GitHub и перезапустить сервер?\n\nСервер будет недоступен на несколько секунд.')) {
            return;
        }
        
        try {
            if (btn) {
                btn.disabled = true;
                btn.textContent = '⏳ Обновление...';
            }
            
            if (statusInfo) {
                statusInfo.innerHTML = '<span style="color: #ffa500;">⏳ Выполняется git pull...</span>';
            }
            
            const response = await fetch('/api/system/update-and-restart', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                if (statusInfo) {
                    statusInfo.innerHTML = `<span style="color: #4caf50;">✅ ${data.message}</span>`;
                }
                
                if (data.restarting) {
                    // Show countdown and reload page
                    let countdown = 5;
                    const countdownInterval = setInterval(() => {
                        if (statusInfo) {
                            statusInfo.innerHTML = `<span style="color: #ffa500;">🔄 Перезапуск через ${countdown} сек...</span>`;
                        }
                        countdown--;
                        if (countdown < 0) {
                            clearInterval(countdownInterval);
                            // Reload page after restart
                            setTimeout(() => {
                                window.location.reload();
                            }, 2000);
                        }
                    }, 1000);
                } else {
                    // No restart needed, just reload status
                    setTimeout(() => {
                        this.loadGitStatus();
                        if (btn) {
                            btn.disabled = false;
                            btn.textContent = '📥 Обновить с GitHub и перезапустить';
                        }
                    }, 2000);
                }
            } else {
                if (statusInfo) {
                    statusInfo.innerHTML = `<span style="color: #f44336;">❌ Ошибка: ${data.error}</span>`;
                }
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '📥 Обновить с GitHub и перезапустить';
                }
                alert('Ошибка обновления: ' + data.error);
            }
            
        } catch (error) {
            console.error('Error updating:', error);
            if (statusInfo) {
                statusInfo.innerHTML = `<span style="color: #f44336;">❌ Ошибка: ${error.message}</span>`;
            }
            if (btn) {
                btn.disabled = false;
                btn.textContent = '📥 Обновить с GitHub и перезапустить';
            }
            alert('Ошибка обновления: ' + error.message);
        }
    }
    
    /**
     * Show training logs modal
     */
    showTrainingLogs() {
        const modal = document.getElementById('trainingLogsModal');
        const titleEl = document.getElementById('trainingLogsTitle');
        if (modal) {
            if (titleEl) titleEl.textContent = '📋 Логи Обучения AlphaZero';
            modal.classList.remove('hidden');
            this.loadTrainingLogs();
        }
    }
    
    /**
     * Load training logs from server
     */
    async loadTrainingLogs() {
        const contentEl = document.getElementById('trainingLogsContent');
        if (!contentEl) return;
        
        contentEl.textContent = 'Загрузка логов...';
        
        try {
            // Get current task ID
            const taskId = this.azTaskId || 'current';
            
            const response = await fetch(`/api/training/alphazero/logs?task_id=${taskId}&lines=300`);
            const data = await response.json();
            
            let logsText = '='.repeat(80) + '\n';
            logsText += '  ЛОГИ ОБУЧЕНИЯ ALPHAZERO\n';
            logsText += '='.repeat(80) + '\n\n';
            
            // Task status
            if (data.task_status) {
                logsText += 'СТАТУС ЗАДАЧИ:\n';
                logsText += `  Task ID: ${taskId}\n`;
                logsText += `  Статус: ${data.task_status.status || 'unknown'}\n`;
                logsText += `  Итерация: ${data.task_status.current_iteration || 0} / ${data.task_status.total_iterations || 0}\n`;
                logsText += `  Игр: ${data.task_status.games_completed || 0} / ${data.task_status.total_games || 0}\n`;
                logsText += `  Примеров: ${data.task_status.examples_collected || 0}\n`;
                logsText += '\n';
            }
            
            // Errors
            if (data.errors && data.errors.length > 0) {
                logsText += 'ОШИБКИ (server_error.log):\n';
                logsText += '-'.repeat(80) + '\n';
                for (const line of data.errors) {
                    logsText += line + '\n';
                }
                logsText += '\n';
            }
            
            // Output
            if (data.output && data.output.length > 0) {
                logsText += 'ВЫВОД (server.log):\n';
                logsText += '-'.repeat(80) + '\n';
                for (const line of data.output) {
                    logsText += line + '\n';
                }
                logsText += '\n';
            }

            // Training Log (New)
            if (data.training && data.training.length > 0) {
                logsText += 'ПРОЦЕСС ОБУЧЕНИЯ (alphazero_training.log):\n';
                logsText += '-'.repeat(80) + '\n';
                for (const line of data.training) {
                    logsText += line + '\n';
                }
                logsText += '\n';
            }
            
            if (data.errors.length === 0 && data.output.length === 0 && (!data.training || data.training.length === 0)) {
                logsText += 'Логи пока пусты. Обучение может еще не начаться или логи не записываются.\n';
            }
            
            logsText += '='.repeat(80) + '\n';
            logsText += `Обновлено: ${new Date().toLocaleString('ru-RU')}\n`;
            
            contentEl.textContent = logsText;
            
        } catch (error) {
            contentEl.textContent = `Ошибка загрузки логов: ${error.message}\n\nПроверь консоль браузера (F12) для деталей.`;
            console.error('Error loading logs:', error);
        }
    }
}

// Initialize training controller
const trainingController = new TrainingController();

