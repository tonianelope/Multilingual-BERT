mkdir -p  data/wt103/models data/wt103/tmp

wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/bwd_wt103.h5
wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/bwd_wt103_enc.h5
wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/fwd_wt103.h5
wget -r -P ./data/wt103/models http://files.fast.ai/models/wt103/fwd_wt103_enc.h5
wget -P ./data/wt103/tmp http://files.fast.ai/models/wt103/itos_wt103.pkl
