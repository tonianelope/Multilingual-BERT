#!/bin/sh

sudo apt-get update
sudp apt-get install tmux htop

if ! p_loc="$(type -p "$python")" || [[ -z $p_loc ]]; then
    sudo apt-get install python3.6
    sudo apt-get -y install python3-pip
else
    echo ">> Python already installed"
fi

# instal Anaconda
if [-f ./Anaconda3-5.3.1-Linux-x86_64.sh ]; then
   echo ">> Installing Anaconda..."
   wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
   bash Anaconda3-5.3.1-Linux-x86_64.sh
else
    echo ">> Anaconda already installed"
fi
