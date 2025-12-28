#!/usr/bin/env python3
"""
AlphaZero Training Monitor - Rich CLI Dashboard

Real-time monitoring of AlphaZero training with:
- Progress bars
- ASCII loss graphs
- Checkpoint status
- GPU utilization

Usage:
    python monitor.py                    # Monitor localhost:8000
    python monitor.py --host 192.168.1.1 # Monitor remote server
    python monitor.py --port 8080        # Custom port
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box
except ImportError:
    print("Installing rich library for beautiful CLI...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box

console = Console()


def create_ascii_graph(values: List[float], width: int = 40, height: int = 8, title: str = "") -> str:
    """Create ASCII line graph from values."""
    if not values or len(values) < 2:
        return "  [No data yet]"
    
    # Normalize values
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Sample values if too many
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values + [values[-1]] * (width - len(values))
    
    # Create graph
    graph_chars = "▁▂▃▄▅▆▇█"
    lines = []
    
    # Title line
    if title:
        lines.append(f"  {title}")
    
    # Value labels
    lines.append(f"  {max_val:.3f} ┐")
    
    # Graph body
    graph_line = "  "
    for val in sampled[:width]:
        normalized = (val - min_val) / range_val
        char_idx = min(int(normalized * (len(graph_chars) - 1)), len(graph_chars) - 1)
        graph_line += graph_chars[char_idx]
    
    lines.append(graph_line + " │")
    lines.append(f"  {min_val:.3f} ┘")
    lines.append(f"  {'─' * width}")
    lines.append(f"  iter 1{' ' * (width - 10)}iter {len(values)}")
    
    return "\n".join(lines)


def create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Create ASCII progress bar."""
    if total == 0:
        return "[" + "░" * width + "] 0%"
    
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent*100:.1f}%"


def format_time(seconds: float) -> str:
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_training_status(base_url: str) -> Dict:
    """Fetch training status from API."""
    try:
        # Get active sessions
        sessions_resp = requests.get(f"{base_url}/api/training/alphazero/sessions", timeout=5)
        sessions = sessions_resp.json().get("sessions", {})
        
        # Get metrics
        metrics_resp = requests.get(f"{base_url}/api/training/alphazero/metrics", timeout=5)
        metrics_data = metrics_resp.json()
        
        # Get GPU info
        gpu_resp = requests.get(f"{base_url}/api/training/alphazero/gpu-info", timeout=5)
        gpu_info = gpu_resp.json()
        
        # Get checkpoints
        checkpoints_resp = requests.get(f"{base_url}/api/training/alphazero/checkpoints", timeout=5)
        checkpoints = checkpoints_resp.json().get("checkpoints", [])
        
        # Find active task
        active_task = None
        for task_id, task in sessions.items():
            if isinstance(task, dict) and task.get("status") in ["running", "pending"]:
                active_task = task
                break
        
        return {
            "active_task": active_task,
            "metrics": metrics_data.get("metrics", []),
            "summary": metrics_data.get("summary", {}),
            "config": metrics_data.get("config", {}),
            "gpu_info": gpu_info,
            "checkpoints": checkpoints[:10],
            "connected": True
        }
    except requests.exceptions.RequestException as e:
        return {"connected": False, "error": str(e)}


def generate_dashboard(status: Dict, start_time: datetime) -> Panel:
    """Generate the main dashboard panel."""
    
    if not status.get("connected"):
        return Panel(
            f"[red]❌ Cannot connect to server[/red]\n\nError: {status.get('error', 'Unknown')}\n\nMake sure the server is running.",
            title="🦾 AlphaZero Monitor",
            border_style="red"
        )
    
    task = status.get("active_task")
    metrics = status.get("metrics", [])
    summary = status.get("summary", {})
    gpu_info = status.get("gpu_info", {})
    checkpoints = status.get("checkpoints", [])
    
    # Build content
    lines = []
    
    # === Header ===
    if task:
        current_iter = task.get("current_iteration", 0)
        total_iter = task.get("total_iterations", 100)
        progress = task.get("progress", 0)
        task_status = task.get("status", "unknown")
        elapsed = task.get("elapsed_time", 0)
        
        # Status indicator
        status_emoji = {"running": "🔄", "completed": "✅", "stopped": "⏹️", "error": "❌"}.get(task_status, "❓")
        
        lines.append(f"[bold cyan]Status:[/bold cyan] {status_emoji} {task_status.upper()}")
        lines.append("")
        
        # Progress bar
        progress_bar = create_progress_bar(current_iter, total_iter, 40)
        lines.append(f"[bold]Iteration:[/bold] {current_iter} / {total_iter}")
        lines.append(f"  {progress_bar}")
        lines.append("")
        
        # Time
        if elapsed > 0:
            eta = (elapsed / max(current_iter, 1)) * (total_iter - current_iter) if current_iter > 0 else 0
            lines.append(f"[bold]Time:[/bold] {format_time(elapsed)} elapsed | ETA: ~{format_time(eta)}")
        lines.append("")
    else:
        lines.append("[yellow]No active training session[/yellow]")
        lines.append("Start training from the web UI or API")
        lines.append("")
    
    # === GPU Info ===
    lines.append("[bold magenta]━━━ GPU Status ━━━[/bold magenta]")
    if gpu_info.get("cuda_available"):
        gpus = gpu_info.get("gpus", [])
        lines.append(f"[green]✓ CUDA {gpu_info.get('cuda_version', '?')} - {len(gpus)} GPU(s)[/green]")
        for gpu in gpus[:4]:  # Show max 4
            lines.append(f"  • {gpu.get('name', '?')} ({gpu.get('memory_total_gb', '?')} GB)")
    else:
        lines.append("[yellow]⚠ CPU Mode (No CUDA)[/yellow]")
    lines.append("")
    
    # === Metrics ===
    lines.append("[bold magenta]━━━ Training Metrics ━━━[/bold magenta]")
    
    if metrics:
        last = metrics[-1]
        policy_loss = last.get("policy_loss", 0)
        value_loss = last.get("value_loss", 0)
        win_rate = last.get("win_rate", 0) * 100
        
        # Color coding
        pl_color = "green" if policy_loss < 1.0 else "yellow" if policy_loss < 1.5 else "red"
        vl_color = "green" if value_loss < 0.1 else "yellow" if value_loss < 0.2 else "red"
        wr_color = "green" if win_rate > 55 else "yellow" if win_rate > 50 else "red"
        
        lines.append(f"  📉 Policy Loss: [{pl_color}]{policy_loss:.4f}[/{pl_color}]  (target: < 1.0)")
        lines.append(f"  📊 Value Loss:  [{vl_color}]{value_loss:.4f}[/{vl_color}]  (target: < 0.1)")
        lines.append(f"  🏆 Win Rate:    [{wr_color}]{win_rate:.1f}%[/{wr_color}]   (target: > 55%)")
        lines.append(f"  📦 Examples:    {last.get('total_examples', 0):,}")
        lines.append("")
        
        # Mini graphs
        if len(metrics) >= 3:
            lines.append("[bold magenta]━━━ Loss Trend ━━━[/bold magenta]")
            policy_values = [m.get("policy_loss", 0) for m in metrics]
            lines.append(create_ascii_graph(policy_values, width=40, title="Policy Loss"))
            lines.append("")
    else:
        lines.append("  [dim]No metrics yet - waiting for first iteration...[/dim]")
        lines.append("")
    
    # === Checkpoints ===
    lines.append("[bold magenta]━━━ Latest Checkpoints ━━━[/bold magenta]")
    if checkpoints:
        for i, cp in enumerate(checkpoints[:5]):
            name = cp.get("name", "?")
            size = cp.get("size_mb", 0)
            cp_metrics = cp.get("metrics", {})
            loss = cp_metrics.get("policy_loss", "?")
            accepted = cp_metrics.get("accepted", False)
            
            marker = "⭐" if i == 0 else "  "
            status_mark = "✓" if accepted else "✗"
            loss_str = f"{loss:.3f}" if isinstance(loss, float) else str(loss)
            
            lines.append(f"  {marker} [{status_mark}] {name} ({size:.1f}MB) loss:{loss_str}")
    else:
        lines.append("  [dim]No checkpoints yet[/dim]")
    lines.append("")
    
    # === Footer ===
    uptime = datetime.now() - start_time
    lines.append(f"[dim]Monitor uptime: {format_time(uptime.total_seconds())} | Press Ctrl+C to exit[/dim]")
    
    content = "\n".join(lines)
    
    return Panel(
        content,
        title="🦾 [bold cyan]AlphaZero Training Monitor[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    )


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Training Monitor")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", default=8000, type=int, help="Server port (default: 8000)")
    parser.add_argument("--interval", default=3, type=int, help="Refresh interval in seconds (default: 3)")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    start_time = datetime.now()
    
    console.clear()
    console.print(f"[bold cyan]Connecting to {base_url}...[/bold cyan]")
    
    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                status = get_training_status(base_url)
                dashboard = generate_dashboard(status, start_time)
                live.update(dashboard)
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
