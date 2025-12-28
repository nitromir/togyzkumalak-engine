#!/bin/bash
# ==============================================================================
# Simple Training Watch Script
# ==============================================================================
# Lightweight alternative to monitor.py - uses curl + jq
#
# Usage (on the server):
#   ./watch.sh                  # Watch localhost:8000
#   ./watch.sh 192.168.1.100    # Watch remote host
#   ./watch.sh localhost 8080   # Custom port
# ==============================================================================

HOST="${1:-localhost}"
PORT="${2:-8000}"
BASE_URL="http://$HOST:$PORT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    apt-get update -qq && apt-get install -y jq -qq 2>/dev/null || \
    yum install -y jq -q 2>/dev/null || \
    brew install jq 2>/dev/null || \
    echo "Please install jq manually"
fi

clear

while true; do
    clear
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  ${BOLD}🦾 AlphaZero Training Monitor${NC}${CYAN}                               ║${NC}"
    echo -e "${CYAN}║  $(date '+%Y-%m-%d %H:%M:%S')                                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Fetch metrics
    METRICS=$(curl -s "$BASE_URL/api/training/alphazero/metrics" 2>/dev/null)
    SESSIONS=$(curl -s "$BASE_URL/api/training/alphazero/sessions" 2>/dev/null)
    GPU_INFO=$(curl -s "$BASE_URL/api/training/alphazero/gpu-info" 2>/dev/null)
    
    if [ -z "$METRICS" ]; then
        echo -e "${RED}❌ Cannot connect to $BASE_URL${NC}"
        echo "Make sure the server is running."
        sleep 5
        continue
    fi
    
    # GPU Info
    echo -e "${BOLD}📊 System:${NC}"
    GPU_COUNT=$(echo "$GPU_INFO" | jq -r '.gpu_count // 0' 2>/dev/null)
    CUDA_VER=$(echo "$GPU_INFO" | jq -r '.cuda_version // "N/A"' 2>/dev/null)
    
    if [ "$GPU_COUNT" != "0" ] && [ "$GPU_COUNT" != "null" ]; then
        GPU_NAME=$(echo "$GPU_INFO" | jq -r '.gpus[0].name // "Unknown"' 2>/dev/null)
        echo -e "  ${GREEN}✓ CUDA $CUDA_VER - ${GPU_COUNT}x $GPU_NAME${NC}"
    else
        echo -e "  ${YELLOW}⚠ CPU Mode${NC}"
    fi
    echo ""
    
    # Training Status
    echo -e "${BOLD}🎯 Training Status:${NC}"
    
    # Check for active session
    ACTIVE_STATUS=$(echo "$SESSIONS" | jq -r '.sessions | to_entries | map(select(.value.status == "running")) | .[0].value.status // "none"' 2>/dev/null)
    
    if [ "$ACTIVE_STATUS" = "running" ]; then
        CURRENT=$(echo "$SESSIONS" | jq -r '.sessions | to_entries | map(select(.value.status == "running")) | .[0].value.current_iteration // 0' 2>/dev/null)
        TOTAL=$(echo "$SESSIONS" | jq -r '.sessions | to_entries | map(select(.value.status == "running")) | .[0].value.total_iterations // 100' 2>/dev/null)
        PROGRESS=$(echo "$SESSIONS" | jq -r '.sessions | to_entries | map(select(.value.status == "running")) | .[0].value.progress // 0' 2>/dev/null)
        
        # Progress bar
        FILLED=$(printf "%.0f" $(echo "$PROGRESS / 100 * 30" | bc -l 2>/dev/null || echo "0"))
        EMPTY=$((30 - FILLED))
        BAR=$(printf "%${FILLED}s" | tr ' ' '█')$(printf "%${EMPTY}s" | tr ' ' '░')
        
        echo -e "  ${GREEN}🔄 TRAINING IN PROGRESS${NC}"
        echo -e "  Iteration: $CURRENT / $TOTAL"
        echo -e "  [$BAR] ${PROGRESS}%"
    else
        SUMMARY_STATUS=$(echo "$METRICS" | jq -r '.summary.status // "none"' 2>/dev/null)
        if [ "$SUMMARY_STATUS" = "completed" ]; then
            TOTAL_ITERS=$(echo "$METRICS" | jq -r '.summary.total_iterations // 0' 2>/dev/null)
            echo -e "  ${GREEN}✅ Last training completed ($TOTAL_ITERS iterations)${NC}"
        else
            echo -e "  ${YELLOW}⏸ No active training${NC}"
        fi
    fi
    echo ""
    
    # Metrics
    echo -e "${BOLD}📉 Latest Metrics:${NC}"
    POLICY_LOSS=$(echo "$METRICS" | jq -r '.summary.latest_policy_loss // 0' 2>/dev/null)
    VALUE_LOSS=$(echo "$METRICS" | jq -r '.summary.latest_value_loss // 0' 2>/dev/null)
    WIN_RATE=$(echo "$METRICS" | jq -r '.summary.latest_win_rate // 0' 2>/dev/null)
    
    # Color coding
    PL_NUM=$(echo "$POLICY_LOSS" | bc -l 2>/dev/null || echo "999")
    if (( $(echo "$PL_NUM < 1.0" | bc -l 2>/dev/null || echo "0") )); then
        PL_COLOR=$GREEN
    elif (( $(echo "$PL_NUM < 1.5" | bc -l 2>/dev/null || echo "0") )); then
        PL_COLOR=$YELLOW
    else
        PL_COLOR=$RED
    fi
    
    echo -e "  Policy Loss: ${PL_COLOR}${POLICY_LOSS}${NC}"
    echo -e "  Value Loss:  ${VALUE_LOSS}"
    WIN_PERCENT=$(echo "$WIN_RATE * 100" | bc -l 2>/dev/null | xargs printf "%.1f" 2>/dev/null || echo "0")
    echo -e "  Win Rate:    ${WIN_PERCENT}%"
    echo ""
    
    # Checkpoints
    echo -e "${BOLD}💾 Best Checkpoints:${NC}"
    BEST=$(echo "$METRICS" | jq -r '.summary.best_checkpoint.filename // "none"' 2>/dev/null)
    BEST_LOSS=$(echo "$METRICS" | jq -r '.summary.best_checkpoint.policy_loss // 0' 2>/dev/null)
    
    if [ "$BEST" != "none" ] && [ "$BEST" != "null" ]; then
        echo -e "  ${GREEN}⭐ $BEST (loss: $BEST_LOSS)${NC}"
    else
        echo -e "  ${YELLOW}No checkpoints yet${NC}"
    fi
    
    # Top 3 checkpoints
    echo "$METRICS" | jq -r '.checkpoints[:3][] | "  • \(.filename) (loss: \(.policy_loss))"' 2>/dev/null
    echo ""
    
    # Mini graph (last 10 values)
    echo -e "${BOLD}📈 Loss Trend (last 10):${NC}"
    LOSSES=$(echo "$METRICS" | jq -r '.metrics[-10:][].policy_loss // empty' 2>/dev/null | tr '\n' ' ')
    
    if [ -n "$LOSSES" ]; then
        echo -n "  "
        for loss in $LOSSES; do
            # Map loss to bar height (1-8)
            HEIGHT=$(echo "scale=0; 8 - ($loss * 4)" | bc -l 2>/dev/null | cut -d. -f1)
            [ "$HEIGHT" -lt 1 ] 2>/dev/null && HEIGHT=1
            [ "$HEIGHT" -gt 8 ] 2>/dev/null && HEIGHT=8
            CHARS="▁▂▃▄▅▆▇█"
            echo -n "${CHARS:$((HEIGHT-1)):1}"
        done
        echo ""
    else
        echo -e "  ${YELLOW}[No data yet]${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${NC}"
    echo -e "  Refreshing every 5s... Press Ctrl+C to stop"
    
    sleep 5
done
