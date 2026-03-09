#!/bin/bash

# tar -xzf packages38_rev1.tar.gz
# tar -xzf  python38.tar.gz
# export PATH=$(pwd)/python/bin:$PATH
# export PYTHONPATH=$PWD/packages
# commented out the above 4 lines of code when trying to use new CHTC system
export HOME=$PWD
mkdir home
python3 chtc_main_P2.py $1
