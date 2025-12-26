/**
 * Togyzkumalak Engine - Replay Viewer
 * Handles loading and playing back recorded game simulations.
 */

class ReplayViewer {
    constructor() {
        this.replays = [];
        this.currentReplay = null;
        this.currentStep = 0;
        this.isPlaying = false;
        this.playInterval = null;
        this.playSpeed = 500;
        
        // Canvas for replay board
        this.canvas = document.getElementById('replayBoard');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        
        // Board drawing config (same as ClassicBoard)
        this.padding = 40;
        this.pitWidth = 70;
        this.pitHeight = 120;
        this.gap = 15;
        
        this.colors = {
            board: '#0f172a',
            pit: '#1e293b',
            pitBorder: '#00f2ff',
            tuzduk: 'rgba(255, 204, 0, 0.4)',
            kumalak: '#e2e8f0',
            kumalakHighlight: '#ffffff',
            text: '#00f2ff',
            textLight: '#94a3b8',
            lastMove: '#ff8800',
            redMarker: '#ff0055'
        };
        
        this.init();
    }

    init() {
        this.bindModeSwitch();
        this.bindControls();
        this.loadReplayList();
    }

    bindModeSwitch() {
        const modeTabs = document.querySelectorAll('.mode-tab');
        modeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                modeTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                const mode = tab.dataset.mode;
                document.getElementById('playMode').classList.toggle('hidden', mode !== 'play');
                document.getElementById('replaysMode').classList.toggle('hidden', mode !== 'replays');
                document.getElementById('trainingMode').classList.toggle('hidden', mode !== 'training');
                
                if (mode === 'replays') {
                    this.loadReplayList();
                } else if (mode === 'training') {
                    // Refresh training data when switching to training mode
                    if (window.trainingController) {
                        trainingController.loadModels();
                        trainingController.loadSessions();
                    }
                }
            });
        });
    }

    bindControls() {
        const btnFirst = document.getElementById('btnReplayFirst');
        const btnPrev = document.getElementById('btnReplayPrev');
        const btnPlay = document.getElementById('btnReplayPlay');
        const btnNext = document.getElementById('btnReplayNext');
        const btnLast = document.getElementById('btnReplayLast');
        const speedSlider = document.getElementById('replaySpeed');
        const progressBar = document.getElementById('replayProgressBar');

        if (btnFirst) btnFirst.onclick = () => this.goToStep(0);
        if (btnPrev) btnPrev.onclick = () => this.goToStep(this.currentStep - 1);
        if (btnPlay) btnPlay.onclick = () => this.togglePlay();
        if (btnNext) btnNext.onclick = () => this.goToStep(this.currentStep + 1);
        if (btnLast) btnLast.onclick = () => this.goToStep(this.currentReplay ? this.currentReplay.total_steps : 0);
        
        if (speedSlider) {
            speedSlider.oninput = (e) => {
                this.playSpeed = parseInt(e.target.value);
                document.getElementById('replaySpeedValue').textContent = this.playSpeed + 'ms';
                if (this.isPlaying) {
                    this.restartPlayInterval();
                }
            };
        }

        if (progressBar) {
            progressBar.onclick = (e) => {
                if (!this.currentReplay) return;
                const rect = e.target.getBoundingClientRect();
                const percent = (e.clientX - rect.left) / rect.width;
                this.goToStep(Math.round(percent * this.currentReplay.total_steps));
            };
        }

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (document.getElementById('replaysMode').classList.contains('hidden')) return;
            
            switch (e.key) {
                case 'ArrowLeft':
                    this.goToStep(this.currentStep - 1);
                    break;
                case 'ArrowRight':
                    this.goToStep(this.currentStep + 1);
                    break;
                case ' ':
                    e.preventDefault();
                    this.togglePlay();
                    break;
            }
        });
    }

    async loadReplayList() {
        const listContainer = document.getElementById('replayList');
        if (!listContainer) return;

        listContainer.innerHTML = '<p class="placeholder">Загрузка...</p>';

        try {
            const response = await fetch('/api/replays');
            const data = await response.json();

            if (data.error) {
                listContainer.innerHTML = `<p class="error">Ошибка: ${data.error}</p>`;
                return;
            }

            if (data.replays.length === 0) {
                listContainer.innerHTML = '<p class="placeholder">Нет записанных партий. Запустите python examples/record_game.py для записи.</p>';
                return;
            }

            this.replays = data.replays;
            listContainer.innerHTML = '';

            data.replays.forEach(replay => {
                const btn = document.createElement('button');
                btn.className = 'replay-btn';
                btn.innerHTML = `
                    <span class="replay-id">Партия #${replay.game_id}</span>
                    <span class="replay-winner ${replay.winner.toLowerCase()}">${replay.winner}</span>
                    <span class="replay-score">${replay.final_score.white}:${replay.final_score.black}</span>
                    <span class="replay-steps">${replay.total_steps} ходов</span>
                `;
                btn.onclick = () => this.loadReplay(replay.game_id);
                listContainer.appendChild(btn);
            });

        } catch (error) {
            console.error('Failed to load replays:', error);
            listContainer.innerHTML = '<p class="error">Не удалось загрузить список партий</p>';
        }
    }

    async loadReplay(gameId) {
        try {
            const response = await fetch(`/api/replays/${gameId}`);
            if (!response.ok) throw new Error('Failed to load replay');
            
            this.currentReplay = await response.json();
            this.currentStep = 0;

            // Show viewer
            document.getElementById('replayViewer').classList.remove('hidden');

            // Update info panel
            const infoPanel = document.getElementById('replayInfo');
            if (infoPanel) {
                infoPanel.innerHTML = `
                    <div class="info-row"><strong>Партия:</strong> #${this.currentReplay.game_id}</div>
                    <div class="info-row"><strong>Дата:</strong> ${this.currentReplay.timestamp}</div>
                    <div class="info-row"><strong>Победитель:</strong> <span class="${this.currentReplay.winner.toLowerCase()}">${this.currentReplay.winner}</span></div>
                    <div class="info-row"><strong>Финальный счёт:</strong> ${this.currentReplay.final_score.white}:${this.currentReplay.final_score.black}</div>
                    <div class="info-row"><strong>Всего ходов:</strong> ${this.currentReplay.total_steps}</div>
                `;
            }

            // Highlight selected replay
            document.querySelectorAll('.replay-btn').forEach((btn, i) => {
                btn.classList.toggle('active', this.replays[i]?.game_id === gameId);
            });

            this.renderState();

        } catch (error) {
            console.error('Failed to load replay:', error);
            alert('Не удалось загрузить партию');
        }
    }

    goToStep(step) {
        if (!this.currentReplay) return;
        this.currentStep = Math.max(0, Math.min(step, this.currentReplay.total_steps));
        this.renderState();
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('btnReplayPlay');
        
        if (this.isPlaying) {
            btn.textContent = '⏸ Пауза';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-danger');
            this.restartPlayInterval();
        } else {
            btn.textContent = '▶️ Воспроизвести';
            btn.classList.remove('btn-danger');
            btn.classList.add('btn-primary');
            clearInterval(this.playInterval);
        }
    }

    restartPlayInterval() {
        clearInterval(this.playInterval);
        this.playInterval = setInterval(() => {
            if (this.currentStep < this.currentReplay.total_steps) {
                this.currentStep++;
                this.renderState();
            } else {
                this.togglePlay(); // Stop at end
            }
        }, this.playSpeed);
    }

    renderState() {
        if (!this.currentReplay || !this.ctx) return;

        const state = this.currentReplay.states[this.currentStep];
        if (!state) return;

        // Clear canvas
        this.ctx.fillStyle = this.colors.board;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw rows
        this.drawRow(0, state.black.holes, 'black', state);
        this.drawRow(1, state.white.holes, 'white', state);

        // Update UI
        document.getElementById('replayStepCounter').textContent = `Шаг: ${this.currentStep} / ${this.currentReplay.total_steps}`;
        document.getElementById('replayWhiteScore').textContent = `Казан: ${state.white.kazan}`;
        document.getElementById('replayBlackScore').textContent = `Казан: ${state.black.kazan}`;
        
        const playerIndicator = document.getElementById('replayCurrentPlayer');
        playerIndicator.textContent = state.current_player.toUpperCase();
        playerIndicator.className = 'turn-indicator ' + state.current_player;

        // Progress bar
        const progress = (this.currentStep / this.currentReplay.total_steps) * 100;
        document.getElementById('replayProgressFill').style.width = progress + '%';

        // Last action
        const lastAction = document.getElementById('replayLastAction');
        if (state.action !== null) {
            lastAction.textContent = `${state.player.toUpperCase()} → Лунка ${state.action + 1}`;
        } else {
            lastAction.textContent = 'Начало партии';
        }

        // Winner banner
        const banner = document.getElementById('replayWinnerBanner');
        if (this.currentStep === this.currentReplay.total_steps) {
            banner.textContent = `🏆 Победитель: ${this.currentReplay.winner} (${this.currentReplay.final_score.white}:${this.currentReplay.final_score.black}) 🏆`;
            banner.classList.remove('hidden');
        } else {
            banner.classList.add('hidden');
        }
    }

    drawRow(rowIndex, pits, player, state) {
        const isBottom = rowIndex === 1;
        const y = isBottom ? 280 : 50;
        
        for (let i = 0; i < 9; i++) {
            const pitIndex = player === 'black' ? 8 - i : i;
            const x = this.padding + i * (this.pitWidth + this.gap);
            const value = pits[pitIndex];
            const isTuzduk = value === -1;

            // Draw pit
            this.ctx.fillStyle = isTuzduk ? this.colors.tuzduk : this.colors.pit;
            this.ctx.strokeStyle = this.colors.pitBorder;
            this.ctx.lineWidth = 2;
            this.roundRect(this.ctx, x, y, this.pitWidth, this.pitHeight, 8, true, true);

            // Pit number
            this.ctx.fillStyle = this.colors.textLight;
            this.ctx.font = '11px Segoe UI';
            this.ctx.textAlign = 'center';
            const labelY = isBottom ? y + this.pitHeight + 18 : y - 8;
            this.ctx.fillText(pitIndex + 1, x + this.pitWidth / 2, labelY);

            // Value
            this.ctx.fillStyle = this.colors.text;
            this.ctx.font = 'bold 18px Segoe UI';
            const countY = isBottom ? y - 12 : y + this.pitHeight + 22;
            this.ctx.fillText(isTuzduk ? 'T' : value.toString(), x + this.pitWidth / 2, countY);

            // Draw kumalaks
            if (!isTuzduk && value > 0) {
                this.drawStackedKumalaks(x, y, value);
            }

            // Highlight last action
            if (state.action === pitIndex && state.player === player) {
                this.ctx.strokeStyle = this.colors.lastMove;
                this.ctx.lineWidth = 4;
                this.ctx.strokeRect(x - 4, y - 4, this.pitWidth + 8, this.pitHeight + 8);
            }
        }
    }

    drawStackedKumalaks(pitX, pitY, count) {
        const kumalakRadius = 9;
        const pitCenterX = pitX + this.pitWidth / 2;
        const pitCenterY = pitY + this.pitHeight / 2;
        const maxVisual = 15;
        const displayCount = Math.min(count, maxVisual);

        const seed = pitX * 17 + pitY * 31;
        const pseudoRandom = (i) => {
            const x = Math.sin(seed + i * 9999) * 10000;
            return x - Math.floor(x);
        };

        const positions = [];
        
        if (displayCount === 1) {
            positions.push({ x: pitCenterX, y: pitCenterY, z: 0 });
        } else if (displayCount === 2) {
            positions.push({ x: pitCenterX - 10, y: pitCenterY - 6, z: 0 });
            positions.push({ x: pitCenterX + 10, y: pitCenterY + 6, z: 1 });
        } else {
            const rows = Math.ceil(displayCount / 2);
            let idx = 0;
            for (let row = 0; row < rows && idx < displayCount; row++) {
                const itemsInRow = Math.min(2, displayCount - idx);
                const rowY = pitY + 20 + row * 12;
                const startX = pitCenterX - ((itemsInRow - 1) * 22 / 2);
                for (let col = 0; col < itemsInRow && idx < displayCount; col++) {
                    positions.push({
                        x: startX + col * 22 + (pseudoRandom(idx) * 3 - 1.5),
                        y: rowY + (pseudoRandom(idx + 30) * 2),
                        z: idx
                    });
                    idx++;
                }
            }
        }

        positions.sort((a, b) => a.z - b.z);

        for (let i = 0; i < positions.length; i++) {
            const pos = positions[i];
            this.drawKumalak(pos.x, pos.y, kumalakRadius, i === positions.length - 1);
        }

        if (count > maxVisual) {
            this.ctx.fillStyle = 'rgba(255,255,255,0.9)';
            this.ctx.font = 'bold 12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`+${count - maxVisual}`, pitCenterX, pitY + this.pitHeight - 10);
        }
    }

    drawKumalak(x, y, radius, isTop) {
        // Shadow
        this.ctx.beginPath();
        this.ctx.arc(x + 2, y + 2, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        this.ctx.fill();

        // Body
        const gradient = this.ctx.createRadialGradient(x - 2, y - 2, 0, x, y, radius);
        gradient.addColorStop(0, this.colors.kumalakHighlight);
        gradient.addColorStop(0.7, this.colors.kumalak);
        gradient.addColorStop(1, '#1a1a1a');
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
        this.ctx.strokeStyle = '#1a1a1a';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();

        // Highlight
        this.ctx.beginPath();
        this.ctx.arc(x - 3, y - 3, 2, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        this.ctx.fill();

        // Red marker
        if (isTop) {
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, Math.PI * 2);
            this.ctx.fillStyle = this.colors.redMarker;
            this.ctx.fill();
        }
    }

    roundRect(ctx, x, y, width, height, radius, fill, stroke) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        if (fill) ctx.fill();
        if (stroke) ctx.stroke();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.replayViewer = new ReplayViewer();
});

