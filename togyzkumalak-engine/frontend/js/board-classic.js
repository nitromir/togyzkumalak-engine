/**
 * Togyzkumalak Engine - Classic Board Renderer (Canvas)
 * With proper kumalak stacking visualization matching reference.
 */

class ClassicBoard {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Layout: increased gap between pits for better readability of confidence indicator
        // Keep total width within 900px canvas by reducing padding.
        this.padding = 25;
        this.pitWidth = 70;
        this.pitHeight = 120;
        this.gap = 30;
        this.rowGap = 80; // Increased gap between rows for labels
        
        this.colors = {
            board: '#0f172a',           // Dark slate tech background
            pit: '#1e293b',             // Lighter slate for pits
            pitBorder: '#00f2ff',       // Cyan tech border
            tuzduk: 'rgba(255, 204, 0, 0.4)', // Golden tuzduk glow
            kumalak: '#e2e8f0',         // Bright kumalak (light slate/white)
            kumalakBorder: '#000000',   // Black border for kumalak
            kumalakHighlight: '#ffffff', // White highlight for 3D effect
            text: '#00f2ff',            // Cyan labels
            textLight: '#94a3b8',       // Slate secondary text
            lastMove: '#ff8800',        // Orange for last move
            playable: 'rgba(0, 242, 255, 0.4)',
            redMarker: '#ff0055'        // Neon pink/red center marker
        };

        this.gameState = null;
        this.humanColor = 'white';
        this.onMoveClick = null;
        this.lastMove = null;
        this.probabilities = null; // Map of pitIndex -> float (0-1)
        this.showProbabilities = true;
        
        this.init();
    }

    setShowProbabilities(enabled) {
        this.showProbabilities = !!enabled;
        if (this.gameState) {
            this.render(this.gameState);
        }
    }

    setProbabilities(probs) {
        this.probabilities = probs;
        if (this.gameState) {
            this.render(this.gameState);
        }
    }

    highlightLastMove(index, player) {
        this.lastMove = { index, player };
        if (this.gameState) {
            this.render(this.gameState);
        }
    }

    init() {
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    }

    setHumanColor(color) {
        this.humanColor = color;
    }

    setMoveCallback(callback) {
        this.onMoveClick = callback;
    }

    render(state) {
        if (!state) return;
        this.gameState = state;
        const { ctx, canvas } = this;

        // Clear canvas
        ctx.fillStyle = this.colors.board;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw rows
        this.drawRow(0, state.black_pits, 'black');
        this.drawRow(1, state.white_pits, 'white');
        
        // Draw score panel between rows
        this.drawScorePanel(state);
    }

    /**
     * Draw the central score panel showing kazans for both players.
     * Displays numbers and progress bars toward victory (82 kumalaks).
     */
    drawScorePanel(state) {
        const { ctx, canvas } = this;
        const whiteKazan = state.white_kazan || 0;
        const blackKazan = state.black_kazan || 0;
        const victoryTarget = 82;
        
        // Panel positioned between the two rows (Y: 170 to 280)
        const panelY = 185;
        const panelHeight = 80;
        const panelWidth = 700;
        const panelX = (canvas.width - panelWidth) / 2;
        
        // Draw panel background
        ctx.fillStyle = 'rgba(30, 41, 59, 0.8)';
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        this.roundRect(ctx, panelX, panelY, panelWidth, panelHeight, 12, true, true);
        
        // === WHITE KAZAN (Left Side) ===
        const whiteX = panelX + 60;
        const scoreY = panelY + 32;
        
        // White label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Segoe UI';
        ctx.textAlign = 'center';
        ctx.fillText('БЕЛЫЕ', whiteX, panelY + 18);
        
        // White score (large number)
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 36px Segoe UI';
        ctx.fillText(whiteKazan.toString(), whiteX, scoreY + 28);
        
        // White progress bar
        const barWidth = 100;
        const barHeight = 10;
        const whiteBarX = whiteX - barWidth / 2;
        const barY = panelY + panelHeight - 18;
        
        // Background bar
        ctx.fillStyle = '#1e293b';
        this.roundRect(ctx, whiteBarX, barY, barWidth, barHeight, 5, true, false);
        
        // Progress fill
        const whiteProgress = Math.min(whiteKazan / victoryTarget, 1);
        if (whiteProgress > 0) {
            const gradient = ctx.createLinearGradient(whiteBarX, barY, whiteBarX + barWidth * whiteProgress, barY);
            gradient.addColorStop(0, '#10b981');
            gradient.addColorStop(1, '#34d399');
            ctx.fillStyle = gradient;
            this.roundRect(ctx, whiteBarX, barY, barWidth * whiteProgress, barHeight, 5, true, false);
        }
        
        // === VS / DIVIDER (Center) ===
        const centerX = canvas.width / 2;
        
        // Decorative divider
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(centerX - 80, panelY + panelHeight / 2);
        ctx.lineTo(centerX - 30, panelY + panelHeight / 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(centerX + 30, panelY + panelHeight / 2);
        ctx.lineTo(centerX + 80, panelY + panelHeight / 2);
        ctx.stroke();
        
        // VS text
        ctx.fillStyle = '#00f2ff';
        ctx.font = 'bold 16px Segoe UI';
        ctx.textAlign = 'center';
        ctx.fillText('VS', centerX, panelY + panelHeight / 2 + 5);
        
        // Victory indicator (small text)
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Segoe UI';
        ctx.fillText(`Цель: ${victoryTarget}`, centerX, panelY + panelHeight - 10);
        
        // === BLACK KAZAN (Right Side) ===
        const blackX = panelX + panelWidth - 60;
        
        // Black label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Segoe UI';
        ctx.textAlign = 'center';
        ctx.fillText('ЧЁРНЫЕ', blackX, panelY + 18);
        
        // Black score (large number)
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 36px Segoe UI';
        ctx.fillText(blackKazan.toString(), blackX, scoreY + 28);
        
        // Black progress bar
        const blackBarX = blackX - barWidth / 2;
        
        // Background bar
        ctx.fillStyle = '#1e293b';
        this.roundRect(ctx, blackBarX, barY, barWidth, barHeight, 5, true, false);
        
        // Progress fill
        const blackProgress = Math.min(blackKazan / victoryTarget, 1);
        if (blackProgress > 0) {
            const gradient = ctx.createLinearGradient(blackBarX, barY, blackBarX + barWidth * blackProgress, barY);
            gradient.addColorStop(0, '#f59e0b');
            gradient.addColorStop(1, '#fbbf24');
            ctx.fillStyle = gradient;
            this.roundRect(ctx, blackBarX, barY, barWidth * blackProgress, barHeight, 5, true, false);
        }
        
        // === KUMALAK ICONS (Visual representation) ===
        // Draw small kumalak icons proportional to score
        this.drawKazanKumalaks(whiteX, scoreY - 15, whiteKazan, '#e2e8f0');
        this.drawKazanKumalaks(blackX, scoreY - 15, blackKazan, '#1e1e1e');
    }

    /**
     * Draw small kumalak indicators for kazan score.
     */
    drawKazanKumalaks(centerX, y, count, baseColor) {
        const { ctx } = this;
        if (count === 0) return;
        
        // Draw up to 5 small kumalaks as visual indicator
        const maxIcons = 5;
        const iconCount = Math.min(Math.ceil(count / 20), maxIcons); // 1 icon per ~20 kumalaks
        const iconRadius = 4;
        const spacing = 12;
        const startX = centerX - ((iconCount - 1) * spacing / 2);
        
        for (let i = 0; i < iconCount; i++) {
            const x = startX + i * spacing;
            
            // Shadow
            ctx.beginPath();
            ctx.arc(x + 1, y + 1, iconRadius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fill();
            
            // Main kumalak
            const gradient = ctx.createRadialGradient(x - 1, y - 1, 0, x, y, iconRadius);
            gradient.addColorStop(0, '#ffffff');
            gradient.addColorStop(0.6, baseColor);
            gradient.addColorStop(1, '#333333');
            
            ctx.beginPath();
            ctx.arc(x, y, iconRadius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
    }

    drawRow(rowIndex, pits, player) {
        const { ctx } = this;
        const isBottom = rowIndex === 1;
        // Black row at top, White row at bottom with more space between
        const y = isBottom ? 280 : 50;
        
        for (let i = 0; i < 9; i++) {
            // In classic view, black row is reversed (9-1)
            const pitIndex = player === 'black' ? 8 - i : i;
            const x = this.padding + i * (this.pitWidth + this.gap);
            const value = pits[pitIndex];
            const isTuzduk = (player === 'white' && this.gameState.black_tuzduk === pitIndex + 1) ||
                            (player === 'black' && this.gameState.white_tuzduk === pitIndex + 1);

            // Draw Pit Background
            ctx.fillStyle = isTuzduk ? this.colors.tuzduk : this.colors.pit;
            ctx.strokeStyle = this.colors.pitBorder;
            ctx.lineWidth = 2;
            this.roundRect(ctx, x, y, this.pitWidth, this.pitHeight, 8, true, true);

            // Draw Pit Number Label (small, at edge)
            ctx.fillStyle = this.colors.textLight;
            ctx.font = '11px Segoe UI';
            ctx.textAlign = 'center';
            const labelY = isBottom ? y + this.pitHeight + 18 : y - 8;
            ctx.fillText(pitIndex + 1, x + this.pitWidth / 2, labelY);

            // Draw Kumalak Count (larger number above/below)
            ctx.fillStyle = this.colors.text;
            ctx.font = 'bold 18px Segoe UI';
            const countY = isBottom ? y - 12 : y + this.pitHeight + 22;
            
            if (isTuzduk) {
                ctx.fillText('T', x + this.pitWidth / 2, countY);
            } else {
                ctx.fillText(value.toString(), x + this.pitWidth / 2, countY);
            }

            // Draw Kumalaks with STACKING if not tuzduk
            if (!isTuzduk && value > 0) {
                this.drawStackedKumalaks(x, y, value);
            }

            // Highlight if playable (white border)
            const isTurn = this.gameState.current_player === this.humanColor;
            const canPlay = isTurn && player === this.humanColor && 
                           this.gameState.legal_moves.includes(pitIndex) && value > 0;
            
            if (canPlay) {
                ctx.strokeStyle = this.colors.playable;
                ctx.lineWidth = 3;
                ctx.strokeRect(x - 2, y - 2, this.pitWidth + 4, this.pitHeight + 4);
            }

            // Highlight last move (orange border)
            if (this.lastMove && this.lastMove.index === pitIndex + 1 && this.lastMove.player === player) {
                ctx.strokeStyle = this.colors.lastMove;
                ctx.lineWidth = 4;
                ctx.strokeRect(x - 4, y - 4, this.pitWidth + 8, this.pitHeight + 8);
            }

            // Draw Confidence Bar (Probabilities)
            if (this.showProbabilities && this.probabilities && this.probabilities[pitIndex] !== undefined && player === this.humanColor) {
                const prob = this.probabilities[pitIndex];
                if (prob > 0.01) {
                    const barHeight = Math.max(2, this.pitHeight * prob);
                    const barY = y + this.pitHeight - barHeight;
                    
                    ctx.fillStyle = prob > 0.5 ? '#10b981' : (prob > 0.2 ? '#f59e0b' : '#6366f1');
                    ctx.globalAlpha = 0.6;
                    // Place bar into the gap area between pits (or inside the last pit to avoid clipping)
                    const isLastCol = i === 8;
                    const barX = isLastCol
                        ? (x + this.pitWidth - 8)
                        : (x + this.pitWidth + (this.gap / 2) - 3);
                    ctx.fillRect(barX, barY, 6, barHeight);
                    ctx.globalAlpha = 1.0;
                    
                    // Percentage text
                    ctx.fillStyle = this.colors.textLight;
                    ctx.font = '9px Segoe UI';
                    ctx.textAlign = 'left';
                    const textX = isLastCol ? (x + this.pitWidth - 32) : (barX + 10);
                    ctx.fillText(`${Math.round(prob * 100)}%`, textX, barY + barHeight/2 + 4);
                }
            }
        }
    }

    /**
     * Draw kumalaks with proper stacking/overlap like the reference.
     * Kumalaks are positioned in layers, overlapping each other.
     */
    drawStackedKumalaks(pitX, pitY, count) {
        const { ctx } = this;
        const kumalakRadius = 9;
        const pitCenterX = pitX + this.pitWidth / 2;
        const pitCenterY = pitY + this.pitHeight / 2;
        
        // Maximum kumalaks to render visually (more = just show number)
        const maxVisual = 15;
        const displayCount = Math.min(count, maxVisual);
        
        // Stacking configuration
        const baseY = pitY + 20;
        const rowHeight = 18;  // Vertical spacing between rows
        const colWidth = 22;   // Horizontal spacing
        const overlapY = 6;    // How much each layer overlaps
        const jitterX = 3;     // Random horizontal jitter
        const jitterY = 2;     // Random vertical jitter
        
        // Seed random for consistent rendering (based on pit position)
        const seed = pitX * 17 + pitY * 31;
        const pseudoRandom = (i) => {
            const x = Math.sin(seed + i * 9999) * 10000;
            return x - Math.floor(x);
        };
        
        // Calculate positions for stacking effect
        const positions = [];
        
        if (displayCount === 1) {
            // Just one center
            positions.push({
                x: pitCenterX + (pseudoRandom(0) * jitterX - jitterX/2),
                y: pitCenterY + (pseudoRandom(100) * jitterY - jitterY/2),
                z: 0
            });
        } else if (displayCount === 2) {
            // Two kumalaks - explicitly separated
            // Left-Top and Right-Bottom pattern for clear visibility
            positions.push({
                x: pitCenterX - 10 + (pseudoRandom(0) * 2),
                y: pitCenterY - 6 + (pseudoRandom(100) * 2),
                z: 0
            });
            positions.push({
                x: pitCenterX + 10 + (pseudoRandom(1) * 2),
                y: pitCenterY + 6 + (pseudoRandom(101) * 2),
                z: 1
            });
        } else if (displayCount <= 5) {
            // 3-5: Two rows with overlap
            const row1Count = Math.ceil(displayCount / 2);
            const row2Count = displayCount - row1Count;
            
            // Bottom row
            const startX1 = pitCenterX - ((row1Count - 1) * colWidth / 2);
            for (let i = 0; i < row1Count; i++) {
                positions.push({
                    x: startX1 + i * colWidth + (pseudoRandom(i) * jitterX - jitterX/2),
                    y: pitCenterY + 12 + (pseudoRandom(i + 50) * jitterY),
                    z: i
                });
            }
            
            // Top row (overlapping)
            const startX2 = pitCenterX - ((row2Count - 1) * colWidth / 2);
            for (let i = 0; i < row2Count; i++) {
                positions.push({
                    x: startX2 + i * colWidth + (pseudoRandom(i + 20) * jitterX - jitterX/2),
                    y: pitCenterY - 8 + (pseudoRandom(i + 70) * jitterY),
                    z: row1Count + i
                });
            }
        } else {
            // 6+: Multi-layer stacking with proper overlap
            const rows = Math.ceil(displayCount / 2);
            let idx = 0;
            
            for (let row = 0; row < rows && idx < displayCount; row++) {
                const itemsInRow = Math.min(2, displayCount - idx);
                const rowY = baseY + row * (rowHeight - overlapY);
                const startX = pitCenterX - ((itemsInRow - 1) * colWidth / 2);
                
                for (let col = 0; col < itemsInRow && idx < displayCount; col++) {
                    positions.push({
                        x: startX + col * colWidth + (pseudoRandom(idx) * jitterX - jitterX/2),
                        y: rowY + (pseudoRandom(idx + 30) * jitterY),
                        z: idx
                    });
                    idx++;
                }
            }
        }
        
        // Sort by Z to draw back-to-front (proper overlap)
        positions.sort((a, b) => a.z - b.z);
        
        // Draw each kumalak with 3D effect
        for (let i = 0; i < positions.length; i++) {
            const pos = positions[i];
            this.drawKumalak(pos.x, pos.y, kumalakRadius, i === positions.length - 1);
        }
        
        // If more than max visual, show "..." or total count
        if (count > maxVisual) {
            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`+${count - maxVisual}`, pitCenterX, pitY + this.pitHeight - 10);
        }
    }

    /**
     * Draw a single kumalak with 3D effect (shadow, gradient, highlight).
     */
    drawKumalak(x, y, radius, isTop) {
        const { ctx } = this;
        
        // Shadow
        ctx.beginPath();
        ctx.arc(x + 2, y + 2, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.fill();
        
        // Main body with gradient
        const gradient = ctx.createRadialGradient(x - 2, y - 2, 0, x, y, radius);
        gradient.addColorStop(0, this.colors.kumalakHighlight);
        gradient.addColorStop(0.7, this.colors.kumalak);
        gradient.addColorStop(1, '#1a1a1a');
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
        
        // Border for depth
        ctx.strokeStyle = this.colors.kumalakBorder;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Highlight reflection (small white dot top-left)
        ctx.beginPath();
        ctx.arc(x - 3, y - 3, 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.fill();
        
        // Red center marker on top kumalak (like reference)
        if (isTop) {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = this.colors.redMarker;
            ctx.fill();
        }
    }

    handleClick(e) {
        if (!this.onMoveClick || !this.gameState) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check only bottom pits (player's pits) - updated Y range
        if (y >= 280 && y <= 400) {
            for (let i = 0; i < 9; i++) {
                const pitX = this.padding + i * (this.pitWidth + this.gap);
                if (x >= pitX && x <= pitX + this.pitWidth) {
                    this.onMoveClick(i + 1);
                    break;
                }
            }
        }
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        let found = false;
        if (y >= 280 && y <= 400) {
            for (let i = 0; i < 9; i++) {
                const pitX = this.padding + i * (this.pitWidth + this.gap);
                if (x >= pitX && x <= pitX + this.pitWidth) {
                    this.canvas.style.cursor = 'pointer';
                    found = true;
                    break;
                }
            }
        }
        if (!found) this.canvas.style.cursor = 'default';
    }

    roundRect(ctx, x, y, width, height, radius, fill, stroke) {
        if (typeof radius === 'number') {
            radius = {tl: radius, tr: radius, bl: radius, br: radius};
        }
        ctx.beginPath();
        ctx.moveTo(x + radius.tl, y);
        ctx.lineTo(x + width - radius.tr, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
        ctx.lineTo(x + width, y + height - radius.br);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
        ctx.lineTo(x + radius.bl, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
        ctx.lineTo(x, y + radius.tl);
        ctx.quadraticCurveTo(x, y, x + radius.tl, y);
        ctx.closePath();
        if (fill) ctx.fill();
        if (stroke) ctx.stroke();
    }
}

window.ClassicBoard = ClassicBoard;
