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
        cpu_ext=''
        echo "GPU version selected\n"
        ;;
    -c|-cpu)
        cpu_ext='-cpu'
        echo "CPU version selected\n"
        ;;

esac

# install fastai v0.7  (https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652)
pip install pip -U
conda update conda
conda install python=3.6
sudo ln -s /home/tonianelope/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh

env_file="environment${cpu_ext}.yml"
env_name="fastai${cpu_ext}"

echo "Installing Fastai v0.7 ..."
git clone https://github.com/fastai/fastai.git
cd fastai
conda env create -f env_file python=3.6
echo "Activating Fastai env"
conda activate env_name

# hack to add fastai to conda sys path (for python scripts)
mkdir -p /home/tonianelope/anaconda3/envs/"${env_name}"/python3.6/fastai/
cp -r /home/tonianelope/GermLM/fastai/courses/dl2/fastai/* /home/tonianelope/anaconda3/envs/"${env_name}"/python3.6/fastai/

cd courses/dl2/imdb_scripts/

mkdir -p data/wt103/models data/wt103/tmp

wget -P ./data/wt103/models/ http://files.fast.ai/models/wt103/bwd_wt103.h5
wget -P ./data/wt103/models/ http://files.fast.ai/models/wt103/bwd_wt103_enc.h5
wget -P ./data/wt103/models/ http://files.fast.ai/models/wt103/fwd_wt103.h5
wget -P ./data/wt103/models/ http://files.fast.ai/models/wt103/fwd_wt103_enc.h5
wget -P ./data/wt103/tmp/itos.pkl http://files.fast.ai/models/wt103/itos_wt103.pkl

curl -O http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz -C data/
