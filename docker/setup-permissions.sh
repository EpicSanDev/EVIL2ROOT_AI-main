#!/bin/bash
# Make entrypoint scripts executable
chmod +x docker/services/entrypoint-bot.sh
chmod +x docker/services/entrypoint-ai.sh
chmod +x docker/services/entrypoint-web.sh
echo "Docker entrypoint scripts are now executable."
