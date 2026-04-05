#how to run me 

#cd ~/iiit_dharwad/src/drl_nav
#./run_demo.sh --stop

#./run_demo.sh
# =============================================================
#  run_demo.sh — One-Click Warehouse DRL Demo
# =============================================================
#
#  ./run_demo.sh              → full demo (Gazebo + Brain + Plot)
#  ./run_demo.sh --obstacles  → drop 3 random cylinders
#  ./run_demo.sh --obstacles 5
#  ./run_demo.sh --clear      → remove cylinders
#  ./run_demo.sh --stop       → kill everything

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="/tmp/drl_demo.pids"

# ── Colours ────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'
B='\033[1m'; R='\033[0;31m'; X='\033[0m'

# ── Terminal opener ────────────────────────────────────────────
open_term() {
    local title="$1"; local cmd="$2"
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="$title" -- bash -c "$cmd; exec bash" &
    else
        xterm -T "$title" -fa 'Monospace' -fs 11 \
              -bg '#0d1117' -fg '#c9d1d9' -geometry 130x35 \
              -e bash -c "$cmd; exec bash" &
    fi
    echo $!
}

# ══════════════════════════════════════════════════════════════
# --stop
# ══════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--stop" ]]; then
    echo -e "${R}Stopping demo...${X}"
    [[ -f "$LOCK_FILE" ]] && while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$LOCK_FILE"
    rm -f "$LOCK_FILE"
    pkill -f "gzserver" 2>/dev/null || true
    pkill -f "gzclient" 2>/dev/null || true
    pkill -f "ros2_node.py" 2>/dev/null || true
    echo -e "${G}Done.${X}"
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# --obstacles / --clear
# ══════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--obstacles" ]]; then
    COUNT="${2:-3}"
    bash -c "source /opt/ros/humble/setup.bash; cd $SCRIPT_DIR; python3 spawn_cylinder.py --count $COUNT"
    exit 0
fi
if [[ "${1:-}" == "--clear" ]]; then
    bash -c "source /opt/ros/humble/setup.bash; cd $SCRIPT_DIR; python3 spawn_cylinder.py --clear"
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# MAIN LAUNCH
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${B}${C}═══════════════════════════════════════════${X}"
echo -e "${B}${C}  Schrodinger's Bug — Warehouse DRL Demo  ${X}"
echo -e "${B}${C}═══════════════════════════════════════════${X}"
echo ""

rm -f "$LOCK_FILE"

# ────────────────────────────────────────────────────────────
# T1 — Gazebo + Robot (using OUR fixed launch, not stock TB3)
#
# demo_launch.py uses TimerAction(15s) so spawn_entity only
# fires AFTER gzserver has fully loaded the factory plugin.
# This eliminates the "/spawn_entity unavailable" error.
# ────────────────────────────────────────────────────────────
echo -e "${G}[1/3]${X} Launching Gazebo + Robot..."
echo -e "  ${Y}Uses demo_launch.py (15s delayed spawn — no race condition)${X}"

T1_CMD="source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
export ALSA_CARD=none
export SDL_AUDIODRIVER=dummy
export LIBGL_ALWAYS_SOFTWARE=1
export SVGA_VGPU10=0
cd $SCRIPT_DIR
echo '══════════════════════════════════════'
echo '  T1 — Gazebo World + Robot Spawn'
echo '══════════════════════════════════════'
echo ''
echo 'Starting Gazebo... Robot spawns 15s after gzserver loads.'
echo 'Please wait — this is normal.'
echo ''
ros2 launch $SCRIPT_DIR/demo_launch.py"

PID1=$(open_term "T1 — Gazebo" "$T1_CMD")
echo "$PID1" >> "$LOCK_FILE"
echo -e "  ${C}→ PID $PID1${X}"

# Wait for Gazebo + the 15s spawn timer + a bit extra
echo ""
echo -e "  ${Y}Waiting 35s for Gazebo to load + robot to spawn...${X}"
for i in $(seq 35 -1 1); do
    printf "\r  ${Y}%2ds remaining...${X}  " "$i"
    sleep 1
done
echo ""
echo ""

# ────────────────────────────────────────────────────────────
# T2 — DRL Brain
# ────────────────────────────────────────────────────────────
echo -e "${G}[2/3]${X} Launching DRL Brain (TD3 + Safety Supervisor)..."

T2_CMD="source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
if [ -f /opt/robot_env/bin/activate ]; then source /opt/robot_env/bin/activate; elif [ -f \$HOME/robot_env/bin/activate ]; then source \$HOME/robot_env/bin/activate; fi
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] {message}'
export RCUTILS_COLORIZED_OUTPUT=1
cd $SCRIPT_DIR
echo '══════════════════════════════════════'
echo '  T2 — DRL Brain (TD3 Agent)'
echo '══════════════════════════════════════'
echo ''
echo 'Robot navigates: Shelf → Dock → Shelf → Dock ...'
echo 'Safety Supervisor overrides if LiDAR < 22cm'
echo ''
python3 ros2_node.py"

PID2=$(open_term "T2 — DRL Brain" "$T2_CMD")
echo "$PID2" >> "$LOCK_FILE"
echo -e "  ${C}→ PID $PID2${X}"
sleep 2

# ────────────────────────────────────────────────────────────
# T3 — Training Proof Plot
# ────────────────────────────────────────────────────────────
echo -e "${G}[3/3]${X} Opening training proof plots..."

T3_CMD="if [ -f /opt/robot_env/bin/activate ]; then source /opt/robot_env/bin/activate; elif [ -f \$HOME/robot_env/bin/activate ]; then source \$HOME/robot_env/bin/activate; fi
cd $SCRIPT_DIR
python3 plot_results.py
echo 'Plot saved. Close this window when done.'"

PID3=$(open_term "T3 — Training Graphs" "$T3_CMD")
echo "$PID3" >> "$LOCK_FILE"

# ────────────────────────────────────────────────────────────
echo ""
echo -e "${B}${G}═══════════════════════════════════════════${X}"
echo -e "${B}${G}  All windows launched! 🚀${X}"
echo -e "${B}${G}═══════════════════════════════════════════${X}"
echo ""
echo -e "  ${C}./run_demo.sh --obstacles${X}   → drop random cylinders"
echo -e "  ${C}./run_demo.sh --clear${X}       → remove cylinders"
echo -e "  ${C}./run_demo.sh --stop${X}        → kill everything"
echo ""
