/**
 * Togyzkumalak Engine - API Client
 * Handles communication with the backend server.
 */

class TogyzkumalakAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
        this.ws = null;
        this.gameId = null;
        this.onMessage = null;
    }

    /**
     * Make HTTP request to the API.
     */
    async request(method, endpoint, data = null) {
        const url = `${this.baseUrl}/api${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            const json = await response.json();
            
            if (!response.ok) {
                throw new Error(json.detail || 'API Error');
            }
            
            return json;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    /**
     * Create a new game.
     */
    async createGame(playerColor = 'white', aiLevel = 3) {
        const data = await this.request('POST', '/games', {
            player_color: playerColor,
            ai_level: aiLevel,
            player_id: 'human'
        });
        
        this.gameId = data.game_id;
        return data;
    }

    /**
     * Get game state.
     */
    async getGame(gameId = null) {
        const id = gameId || this.gameId;
        return await this.request('GET', `/games/${id}`);
    }

    /**
     * Make a move.
     */
    async makeMove(move) {
        return await this.request('POST', `/games/${this.gameId}/move`, { move });
    }

    /**
     * Get move history.
     */
    async getMoveHistory() {
        return await this.request('GET', `/games/${this.gameId}/history`);
    }

    /**
     * Resign the game.
     */
    async resign() {
        return await this.request('POST', `/games/${this.gameId}/resign`);
    }

    /**
     * Get position analysis from Gemini.
     */
    async analyzePosition() {
        return await this.request('POST', `/games/${this.gameId}/analyze`, {
            include_history: true
        });
    }

    /**
     * Get move suggestion from Gemini.
     */
    async suggestMove() {
        return await this.request('POST', `/games/${this.gameId}/suggest`);
    }

    /**
     * Get ELO statistics.
     */
    async getEloStats() {
        return await this.request('GET', '/elo');
    }

    /**
     * Get AI levels info.
     */
    async getAILevels() {
        return await this.request('GET', '/ai/levels');
    }

    /**
     * Get move probabilities for visualization.
     */
    async getMoveProbabilities(level = 3) {
        return await this.request('GET', `/ai/probabilities/${this.gameId}?level=${level}`);
    }

    /**
     * Connect WebSocket for real-time gameplay.
     */
    connectWebSocket(gameId, onMessage) {
        this.gameId = gameId;
        this.onMessage = onMessage;

        const wsUrl = `${this.baseUrl.replace('http', 'ws')}/ws/${gameId}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (this.onMessage) {
                this.onMessage(data);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
        };
    }

    /**
     * Send WebSocket message.
     */
    sendWS(type, data = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, ...data }));
        }
    }

    /**
     * Make move via WebSocket.
     */
    makeMoveWS(move) {
        this.sendWS('make_move', { move });
    }

    /**
     * Request analysis via WebSocket.
     */
    requestAnalysisWS() {
        this.sendWS('request_analysis');
    }

    /**
     * Request move suggestion via WebSocket.
     */
    requestSuggestionWS() {
        this.sendWS('request_suggestion');
    }

    /**
     * Resign via WebSocket.
     */
    resignWS() {
        this.sendWS('resign');
    }

    /**
     * Disconnect WebSocket.
     */
    disconnectWebSocket() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Global API instance
const api = new TogyzkumalakAPI();

