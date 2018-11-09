apt install -y language-pack-zh-hans git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> ~/.bash_profile

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile

mkdir -p ~/.pyenv/cache

pyenv install anaconda3-5.2.0

conda install opencv gdal
pip install msgpack tensorflow-gpu keras shapely bypy

# 百度云盘下载数据

git clone https://github.com/korewayume/dstl.git

# 安装cuda和cudnn
