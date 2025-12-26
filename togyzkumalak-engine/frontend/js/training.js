/**
 * Training Mode Controller
 * Manages gym training sessions, progress monitoring, and model management
 */

class TrainingController {
    constructor() {
        this.currentSessionId = null;
        this.pollInterval = null;
        this.init();
    }

    init() {
        // Configuration controls
        document.getElementById('btnStartTraining').addEventListener('click', () => this.startTraining());
        
        // Load initial data
        this.loadModels();
        this.loadSessions();
        
        // Set up periodic refresh when training mode is active
        setInterval(() => {
            if (!document.getElementById('trainingMode').classList.contains('hidden')) {
                this.loadModels();
                this.loadSessions();
            }
        }, 5000);
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
            document.getElementById('trainingProgressBar').style.width = `${percentage}%`;
            document.getElementById('trainingProgressText').textContent = 
                `${progress.games_completed} / ${progress.total_games} игр`;

            // Update stats
            document.getElementById('whiteWins').textContent = progress.white_wins;
            document.getElementById('blackWins').textContent = progress.black_wins;
            document.getElementById('draws').textContent = progress.draws;
            document.getElementById('avgSteps').textContent = progress.avg_steps.toFixed(1);

            // Check if completed
            if (progress.status === 'completed') {
                this.stopProgressPolling();
                document.getElementById('trainingStatus').innerHTML = `
                    <p class="status-text status-completed">✅ Тренировка завершена</p>
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

            if (data.models.length === 0) {
                modelsList.innerHTML = '<p class="empty-text">Нет сохранённых моделей</p>';
                return;
            }

            modelsList.innerHTML = data.models.map(model => `
                <div class="model-card">
                    <div class="model-header">
                        <span class="model-name">📦 ${model.name}</span>
                        <span class="model-size">${model.size_mb} MB</span>
                    </div>
                    <div class="model-info">
                        <span class="model-date">${new Date(model.created).toLocaleString('ru-RU')}</span>
                    </div>
                    <button class="btn btn-secondary btn-small" onclick="trainingController.loadModel('${model.name}')">
                        Загрузить
                    </button>
                </div>
            `).join('');

        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    async loadModel(modelName) {
        try {
            const response = await fetch(`/api/training/models/${modelName}/load`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to load model');
            }

            alert(`Модель "${modelName}" успешно загружена!\n\nТеперь она будет использоваться для игры на уровнях 4-5.`);

        } catch (error) {
            console.error('Error loading model:', error);
            alert('Ошибка загрузки модели: ' + error.message);
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

