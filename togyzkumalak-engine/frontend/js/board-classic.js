/**
 * Togyzkumalak Engine - Classic Board Renderer (Canvas)
 * With proper kumalak stacking visualization matching reference.
 */

class ClassicBoard {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        this.padding = 40;
        this.pitWidth = 70;
        this.pitHeight = 120;
        this.gap = 15;
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
        
        this.init();
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
