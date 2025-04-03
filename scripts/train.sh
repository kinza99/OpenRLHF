#!/bin/bash

set -x

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "请提供脚本名称作为参数"
    echo "使用方法: $0 <脚本名称>"
    exit 1
fi

source /home/duhe/miniconda3/etc/profile.d/conda.sh
conda activate openrlhf

cd /cpfs01/user/duhe/OpenRLHF


./examples/scripts/"$1".sh