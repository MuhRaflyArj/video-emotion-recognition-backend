#!/bin/bash

# Update package list and install Docker if not present
if ! command -v docker &> /dev/null
then
    echo "Docker not found. Installing Docker for Debian..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg

    # Add Docker’s official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "Docker installed. Applying group changes with 'sg docker'..."
    exec sg docker "$(realpath "$0")" "$@"
    exit 0
else
    echo "Docker is already installed."
fi

docker build -t expression-recognition-app .
docker run -d --env-file .env -p 5000:5000 expression-recognition-app