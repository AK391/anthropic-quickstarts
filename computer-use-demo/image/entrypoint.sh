#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh

python http_server.py > /tmp/server_logs.txt 2>&1 &

STREAMLIT_SERVER_PORT=8501 python -m streamlit run computer_use_demo/streamlit.py > /tmp/streamlit_stdout.log &

GRADIO_SERVER_PORT=7860 python computer_use_demo/gradio_app.py > /tmp/gradio_stdout.log &

echo "✨ Computer Use Demo is ready!"
echo "➡️  Open http://localhost:8080 in your browser to begin (Streamlit)"
echo "➡️  Open http://localhost:7860 in your browser for the Gradio interface"

# Keep the container running
tail -f /dev/null
