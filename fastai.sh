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

# install fastai v0.7  (https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652)
pip install pip -U
conda update conda

if [ "$gpu" ]; then
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

mkdir fastai-scripts
ln -s fastai/courses/dl2/imdb_scripts/* ./fastai-scripts/

mkdir -p  data/wt103/models data/wt103/tmp

wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/bwd_wt103.h5
wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/bwd_wt103_enc.h5
wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/fwd_wt103.h5
wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/fwd_wt103_enc.h5
wget -P ./data/wt103/tmp http://files.fast.ai/models/wt103/itos_wt103.pkl

