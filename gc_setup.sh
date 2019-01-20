#!/bin/sh

# Usage info
show_help(){
    cat <<EOF
        Usage: ${0##*/} [-hg]
        Setup environment for fastai

              -h       displays help
              -g       installs fastai with GPU enabled
EOF
}

case $1 in
    -h|-\?|--help)
        show_help
        exit
        ;;
    -g|-gpu)
        gpu=t
        echo "GPU version selected\n"
        ;;
    -c|-cpu)
        cpu=t
        echo "GPU version selected\n"
        ;;

esac

sudo apt-get update
sudp apt-get install tmux htop

if ! p_loc="$(type -p "$python")" || [[ -z $p_loc ]]; then
    sudo apt-get install python3.6
    sudo apt-get -y install python3-pip
else
    echo "Python already installed"
fi

# instal Anaconda
if [-f ./Anaconda3-5.3.1-Linux-x86_64.sh ]
   echo "Installing Anaconda"
   wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
   bash Anaconda3-5.3.1-Linux-x86_64.sh
else
    echo "Anaconda already installed"
fi

# install fastai v0.7  (https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652)
pip install pip -U
conda update conda

if "$gpu"; then
   echo "Installing Fastai GPU ..."
   git clone https://github.com/fastai/fastai.git
   cd fastai
   conda env create -f environment.yml
   echo "Activating Fastai env"
   conda activate fastai
else
    echo "Installing Fastai CPU ..."
    git clone https://github.com/fastai/fastai.git
    cd fastai
    conda env create -f environment-cpu.yml
    echo "Activating Fastai env"
    conda activate fastai-cpu
fi
