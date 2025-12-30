#!/usr/bin/env python3
"""
AlphaZero Training Monitor - Rich CLI Dashboard

Real-time monitoring of AlphaZero training with:
- Progress bars
- ASCII loss graphs
- Checkpoint status
- REAL-TIME GPU utilization (All 16 GPUs)
- Real-time training logs

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
    from rich.columns import Columns
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
    from rich.columns import Columns
    from rich import box

console = Console()


def create_ascii_graph(values: List[float], width: int = 40, title: str = "") -> str:
    """Create ASCII line graph from values."""
    if not values or len(values) < 2:
        return "  [Waiting for more data...]"
    
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
    graph_chars = " ▂▃▄▅▆▇█"
    lines = []
    
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
    
    return "\n".join(lines)


def create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Create ASCII progress bar."""
    if total <= 0:
        return "[" + "░" * width + "] 0%"
    
    percent = min(1.0, current / total)
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
    """Fetch all necessary status info from API."""
    try:
        # 1. Sessions status
        sessions_resp = requests.get(f"{base_url}/api/training/alphazero/sessions", timeout=2)
        sessions = sessions_resp.json().get("sessions", {})
        
        # 2. Metrics (Loss, Win Rate)
        metrics_resp = requests.get(f"{base_url}/api/training/alphazero/metrics", timeout=2)
        metrics_data = metrics_resp.json()
        
        # 3. REAL-TIME GPU Utilization
        gpu_util_resp = requests.get(f"{base_url}/api/system/gpu-utilization", timeout=2)
        gpu_util = gpu_util_resp.json()
        
        # 3b. REAL-TIME CPU Utilization
        cpu_util = {"connected": False}
        try:
            cpu_util_resp = requests.get(f"{base_url}/api/system/cpu-utilization", timeout=2)
            cpu_util = cpu_util_resp.json()
        except:
            pass
        
        # 4. Logs
        logs_resp = requests.get(f"{base_url}/api/training/alphazero/logs?lines=50", timeout=2)
        logs_data = logs_resp.json()
        
        # Prefer the detailed training log if available
        display_logs = logs_data.get("training", [])
        if not display_logs:
            display_logs = logs_data.get("output", [])
        
        # Find active task
        active_task = None
        for task_id, task in sessions.items():
            if isinstance(task, dict) and task.get("status") in ["running", "pending"]:
                active_task = task
                active_task["id"] = task_id
                break
        
        return {
            "active_task": active_task,
            "metrics": metrics_data.get("metrics", []),
            "gpu_util": gpu_util.get("gpus", []),
            "cpu_util": cpu_util,
            "logs": display_logs[-30:], # Get last 30 lines
            "connected": True
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def generate_dashboard(status: Dict, start_time: datetime) -> Layout:
    """Generate the multi-pane layout dashboard."""
    layout = Layout()
    
    # Split main layout
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    
    # Split main into left (Status/Metrics) and right (GPU/Logs)
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    # Header
    layout["header"].update(
        Panel(
            Text("🚀 Togyzkumalak AlphaZero - 16x GPU Super-Training Monitor", justify="center", style="bold cyan"),
            box=box.ROUNDED, border_style="cyan"
        )
    )
    
    # Footer
    uptime = datetime.now() - start_time
    layout["footer"].update(
        Panel(
            Text(f"Uptime: {format_time(uptime.total_seconds())} | Refresh: 3s | Ctrl+C to Exit", justify="center", style="dim"),
            box=box.ROUNDED, border_style="dim"
        )
    )
    
    if not status.get("connected"):
        layout["main"].update(
            Panel(f"[red]❌ Connection Error:[/red] {status.get('error')}", title="Error", border_style="red")
        )
        return layout

    task = status.get("active_task")
    metrics = status.get("metrics", [])
    gpu_list = status.get("gpu_util", [])
    logs = status.get("logs", [])

    # === LEFT PANE: Progress & Metrics ===
    left_content = []
    
    # 1. System Load (Always visible)
    cpu = status.get("cpu_util", {})
    if cpu.get("status") == "ok":
        cpu_p = cpu.get("cpu_percent", 0)
        mem_p = cpu.get("memory_percent", 0)
        cores = cpu.get("cpu_count", "N/A")
        
        left_content.append(f"[bold cyan]System Load ({cores} cores):[/bold cyan]")
        left_content.append(f"  CPU: {create_progress_bar(int(cpu_p), 100, 25)}")
        left_content.append(f"  RAM: {create_progress_bar(int(mem_p), 100, 25)}")
        left_content.append("")

    # 2. Training Progress
    if task:
        curr = task.get("current_iteration", 0)
        total = task.get("total_iterations", 100)
        tid = task.get("id", "N/A")
        status_text = task.get("status_text", "Running")
        
        left_content.append(f"[bold yellow]Task ID:[/bold yellow] {tid}")
        left_content.append(f"[bold]Iteration:[/bold] {curr} / {total} [dim]({status_text})[/dim]")
        left_content.append(create_progress_bar(curr, total, 30))
    else:
        # If no active task, show last info from metrics
        last_iter = metrics[-1].get("iteration", "N/A") if metrics else "N/A"
        left_content.append("[bold red]⚠ NO ACTIVE TRAINING TASK[/bold red]")
        left_content.append(f"[dim]Last completed iteration: {last_iter}[/dim]")
        left_content.append("[dim]Start training in Jupyter to see progress.[/dim]")
    
    left_content.append("")
    
    # 3. Metrics
    if metrics:
        last = metrics[-1]
        p_loss = last.get("policy_loss", 0)
        v_loss = last.get("value_loss", 0)
        win_r = last.get("win_rate", 0) * 100
        
        left_content.append(f"[bold cyan]Metrics (Last Iter):[/bold cyan]")
        left_content.append(f"  • Policy Loss: [green]{p_loss:.4f}[/green]")
        left_content.append(f"  • Value Loss:  [green]{v_loss:.4f}[/green]")
        left_content.append(f"  • Win Rate:    [bold yellow]{win_r:.1f}%[/bold yellow]")
        left_content.append("")
        
        # Mini Graph
        if len(metrics) > 2:
            left_content.append("[bold cyan]Policy Loss Trend:[/bold cyan]")
            p_values = [m.get("policy_loss", 0) for m in metrics]
            left_content.append(create_ascii_graph(p_values, width=30))
    else:
        left_content.append("[dim]Waiting for first iteration metrics...[/dim]")

    layout["left"].update(Panel("\n".join(left_content), title="📊 Training Status", border_style="green"))

    layout["left"].update(Panel("\n".join(left_content), title="📊 Training Status", border_style="green"))

    # === RIGHT PANE: GPU & Logs ===
    # Split right into GPU (top) and Logs (bottom) - Logs get more space
    layout["right"].split_column(
        Layout(name="gpu", ratio=1),
        Layout(name="logs", ratio=3)
    )

    # GPU Utilization
    gpu_table = Table(box=None, padding=(0, 1), show_header=False)
    gpu_table.add_column("ID", justify="right", style="cyan")
    gpu_table.add_column("Bar", width=15)
    gpu_table.add_column("Load", justify="right")
    gpu_table.add_column("Mem", justify="right")

    if gpu_list:
        # Group GPUs into columns if there are many
        for g in gpu_list:
            idx = g.get("index", 0)
            util = g.get("utilization", 0)
            used = g.get("memory_used_mib", 0)
            total = g.get("memory_total_mib", 0)
            
            # Color based on load
            color = "green" if util < 30 else "yellow" if util < 70 else "bold red"
            
            # Small bar for GPU
            bar_width = 10
            filled = int(bar_width * (util / 100))
            bar = f"[{color}]" + "█" * filled + "[/]" + "░" * (bar_width - filled)
            
            gpu_table.add_row(
                f"G{idx}",
                bar,
                f"[{color}]{util}%[/{color}]",
                f"{used/1024:.1f}/{total/1024:.0f}G"
            )
        gpu_panel_content = gpu_table
    else:
        gpu_panel_content = Text("No GPU information available", style="dim")

    layout["gpu"].update(Panel(gpu_panel_content, title="🔥 GPU Real-Time Load", border_style="magenta"))

    # Logs - Styled like the user's screenshot
    log_text = Text()
    if logs:
        for line in logs:
            # Parse line: 2025-12-30 07:54:02,622 - INFO - Message
            if " - INFO - " in line:
                parts = line.split(" - INFO - ", 1)
                log_text.append(parts[0], style="dim")
                log_text.append(" - INFO - ", style="green")
                msg = parts[1]
                
                # Highlight important keywords
                if "Epoch" in msg:
                    log_text.append(msg + "\n", style="bold yellow")
                elif "Progress" in msg:
                    log_text.append(msg + "\n", style="cyan")
                elif "Arena" in msg:
                    log_text.append(msg + "\n", style="bold magenta")
                elif "Checkpoint" in msg:
                    log_text.append(msg + "\n", style="bold green")
                else:
                    log_text.append(msg + "\n")
            elif "ERROR" in line or "fail" in line.lower():
                log_text.append(line + "\n", style="bold red")
            else:
                log_text.append(line + "\n", style="dim")
    else:
        log_text.append("Waiting for logs...", style="dim")

    layout["logs"].update(Panel(log_text, title="📜 Training Log (Live Feed)", border_style="yellow"))

    return layout


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
        pass
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        time.sleep(5)


if __name__ == "__main__":
    main()
