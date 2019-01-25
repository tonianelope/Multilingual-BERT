#!/bin/sh

sudo apt-get update
sudo apt-get install tmux htop

## update to instlall for python 3.6 (e.g if 2.x installed)
python3 -V foo >/dev/null 2>&1 || {
    echo >&2 "Installing Python3 ...\n"
    sudo apt-get install python3.6
    sudo apt-get -y install python3-pip
}

# instal Anaconda
conda -V foo >/dev/null 2>&1 || {
    echo >&2 "Installing Anaconda ...\n"
    wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
    bash Anaconda3-5.3.1-Linux-x86_64.sh
}
