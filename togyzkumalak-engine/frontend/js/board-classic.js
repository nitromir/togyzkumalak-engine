/**
 * Togyzkumalak Engine - Classic Board Renderer (Canvas)
 * Mimics the traditional web-based interface.
 */

class ClassicBoard {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        this.padding = 40;
        this.pitWidth = 70;
        this.pitHeight = 130;
        this.gap = 15;
        
        this.colors = {
            board: '#e5b16d',      // Sandy board background
            pit: '#c88c4a',        // Pit background
            pitBorder: '#8d5a2d',  // Pit border
            tuzduk: '#f1c40f',     // Golden tuzduk
            kumalak: '#3d3429',    // Dark kumalak
            text: '#2c3e50',       // Labels
            highlight: 'rgba(255, 255, 255, 0.5)',
            lastMove: '#f39c12',
            redMarker: '#e74c3c'   // Red dots for last move
        };

        this.gameState = null;
        this.humanColor = 'white';
        this.onMoveClick = null;
        
        this.init();
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
        const y = isBottom ? 220 : 50;
        
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

            // Draw Pit Number Label
            ctx.fillStyle = this.colors.text;
            ctx.font = 'bold 14px Segoe UI';
            ctx.textAlign = 'center';
            const labelY = isBottom ? y + this.pitHeight + 20 : y - 10;
            ctx.fillText(pitIndex + 1, x + this.pitWidth / 2, labelY);

            // Draw Kumalak Count (The big number)
            ctx.fillStyle = this.colors.text;
            ctx.font = 'bold 16px Segoe UI';
            const countY = isBottom ? y - 10 : y + this.pitHeight + 20;
            ctx.fillText(value === -1 ? 'T' : value, x + this.pitWidth / 2, countY);

            // Draw Kumalaks visually if not too many
            if (value > 0) {
                this.drawKumalaksInPit(x, y, value);
            }

            // Highlight if playable
            const isTurn = this.gameState.current_player === this.humanColor;
            const canPlay = isTurn && player === this.humanColor && this.gameState.legal_moves.includes(pitIndex);
            
            if (canPlay) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = 3;
                ctx.strokeRect(x - 2, y - 2, this.pitWidth + 4, this.pitHeight + 4);
            }
        }
    }

    drawKumalaksInPit(x, y, count) {
        const { ctx } = this;
        const rows = 5;
        const cols = 2;
        const radius = 8;
        const spacingX = 25;
        const spacingY = 22;
        const startX = x + 22;
        const startY = y + 22;

        const displayCount = Math.min(count, 10);
        
        for (let i = 0; i < displayCount; i++) {
            const r = Math.floor(i / cols);
            const c = i % cols;
            const kX = startX + c * spacingX;
            const kY = startY + r * spacingY;

            // Kumalak shadow
            ctx.beginPath();
            ctx.arc(kX + 1, kY + 1, radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fill();

            // Kumalak body
            ctx.beginPath();
            ctx.arc(kX, kY, radius, 0, Math.PI * 2);
            ctx.fillStyle = this.colors.kumalak;
            ctx.fill();

            // Red dot if it's the "last" one (marker)
            if (i === displayCount - 1 && count > 0) {
                ctx.beginPath();
                ctx.rect(kX - 2, kY - 2, 4, 4);
                ctx.fillStyle = this.colors.redMarker;
                ctx.fill();
            }
        }
        
        // If more than 10, show a small "+" or just the number
        if (count > 10) {
            ctx.fillStyle = 'white';
            ctx.font = 'bold 10px Arial';
            ctx.fillText('...', x + this.pitWidth / 2, y + this.pitHeight - 15);
        }
    }

    handleClick(e) {
        if (!this.onMoveClick || !this.gameState) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check only bottom pits (player's pits)
        if (y >= 220 && y <= 350) {
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
        if (y >= 220 && y <= 350) {
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
