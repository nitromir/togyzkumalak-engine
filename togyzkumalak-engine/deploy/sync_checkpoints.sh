#!/bin/bash
# ==============================================================================
# Auto-Sync Checkpoints from Vast.ai to Local Machine
# ==============================================================================
#
# Continuously syncs new checkpoints from remote server to local directory.
# Run this on YOUR LOCAL machine while training runs on Vast.ai.
#
# Usage:
#   ./sync_checkpoints.sh <ssh_connection> [local_dir] [interval]
#
# Examples:
#   ./sync_checkpoints.sh root@192.168.1.100 -p 22022
#   ./sync_checkpoints.sh "root@123.45.67.89 -p 12345" ./my_checkpoints 60
#
# Requirements:
#   - SSH access to remote server (key-based auth recommended)
#   - rsync installed on both machines
#
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SSH_CONNECTION="${1:-}"
LOCAL_DIR="${2:-./alphazero_checkpoints}"
SYNC_INTERVAL="${3:-30}"  # seconds
REMOTE_PATH="/workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine/models/alphazero"

# Validate arguments
if [ -z "$SSH_CONNECTION" ]; then
    echo -e "${RED}Error: SSH connection required${NC}"
    echo ""
    echo "Usage: $0 <ssh_connection> [local_dir] [interval]"
    echo ""
    echo "Examples:"
    echo "  $0 'root@123.45.67.89 -p 12345'"
    echo "  $0 'root@123.45.67.89 -p 12345' ./checkpoints 60"
    echo ""
    exit 1
fi

# Create local directory
mkdir -p "$LOCAL_DIR"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  🔄 AlphaZero Checkpoint Auto-Sync                           ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  Remote: ${NC}$SSH_CONNECTION:$REMOTE_PATH"
echo -e "${CYAN}║  Local:  ${NC}$LOCAL_DIR"
echo -e "${CYAN}║  Interval: ${NC}${SYNC_INTERVAL}s"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
if ssh $SSH_CONNECTION "echo 'Connected!'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH connection successful${NC}"
else
    echo -e "${RED}✗ SSH connection failed${NC}"
    echo "Make sure you can connect with: ssh $SSH_CONNECTION"
    exit 1
fi

# Counter
SYNC_COUNT=0
TOTAL_SYNCED=0

# Function to sync checkpoints
sync_checkpoints() {
    SYNC_COUNT=$((SYNC_COUNT + 1))
    TIMESTAMP=$(date '+%H:%M:%S')
    
    echo -e "\n${CYAN}[$TIMESTAMP] Sync #$SYNC_COUNT${NC}"
    
    # Get list of remote files
    REMOTE_FILES=$(ssh $SSH_CONNECTION "ls -la $REMOTE_PATH/*.pth.tar 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    
    if [ "$REMOTE_FILES" = "0" ]; then
        echo -e "${YELLOW}  No checkpoints found on remote server yet...${NC}"
        return
    fi
    
    # Sync using rsync
    echo -e "  Syncing $REMOTE_FILES checkpoint(s)..."
    
    # Use rsync if available, otherwise scp
    if command -v rsync &> /dev/null; then
        RESULT=$(rsync -avz --progress -e "ssh" \
            "$SSH_CONNECTION:$REMOTE_PATH/*.pth.tar" \
            "$LOCAL_DIR/" 2>&1)
        
        # Count new files
        NEW_FILES=$(echo "$RESULT" | grep -c "\.pth\.tar$" || echo "0")
    else
        # Fallback to scp
        scp $SSH_CONNECTION:"$REMOTE_PATH/*.pth.tar" "$LOCAL_DIR/" 2>/dev/null
        NEW_FILES=$(ls -1 "$LOCAL_DIR"/*.pth.tar 2>/dev/null | wc -l)
    fi
    
    # Also sync metrics file
    ssh $SSH_CONNECTION "cat $REMOTE_PATH/training_metrics.json 2>/dev/null" > "$LOCAL_DIR/training_metrics.json" 2>/dev/null || true
    
    # Report
    LOCAL_COUNT=$(ls -1 "$LOCAL_DIR"/*.pth.tar 2>/dev/null | wc -l || echo "0")
    LOCAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1 || echo "0")
    
    echo -e "${GREEN}  ✓ Local checkpoints: $LOCAL_COUNT (${LOCAL_SIZE})${NC}"
    
    # Show latest checkpoint
    LATEST=$(ls -t "$LOCAL_DIR"/*.pth.tar 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        LATEST_NAME=$(basename "$LATEST")
        LATEST_SIZE=$(ls -lh "$LATEST" | awk '{print $5}')
        echo -e "${GREEN}  ⭐ Latest: $LATEST_NAME ($LATEST_SIZE)${NC}"
    fi
}

# Function to show summary
show_summary() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  📊 Sync Summary${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    
    if [ -d "$LOCAL_DIR" ]; then
        echo -e "  Local checkpoints:"
        ls -lhS "$LOCAL_DIR"/*.pth.tar 2>/dev/null | head -10 | while read line; do
            echo "    $line"
        done
        
        TOTAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)
        TOTAL_FILES=$(ls -1 "$LOCAL_DIR"/*.pth.tar 2>/dev/null | wc -l)
        echo -e "\n  Total: $TOTAL_FILES files, $TOTAL_SIZE"
    fi
    
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Stopping sync...${NC}"; show_summary; exit 0' INT

# Main loop
echo -e "\n${GREEN}Starting auto-sync (Ctrl+C to stop)...${NC}"

while true; do
    sync_checkpoints
    
    # Countdown to next sync
    for i in $(seq $SYNC_INTERVAL -1 1); do
        printf "\r  Next sync in ${i}s...  "
        sleep 1
    done
    printf "\r                        \r"
done
