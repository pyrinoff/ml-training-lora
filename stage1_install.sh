# delete previous python
sudo apt-get remove *python*
# install python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
python3.10 --version
# make python default
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
# install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
python3.10 -m pip --version
pip --version
# install setuptools
pip uninstall setuptools && pip install setuptools
# install requirements
pip install -r requirements.txt
