#!/bin/bash
# =====================================================
# 电价预测系统 - API版本一键运行脚本 (macOS/Linux)
# =====================================================

echo "====================================================="
echo "电价预测系统 - API版本"
echo "====================================================="
echo ""

# 设置Python路径（macOS/Linux通常使用python3）
PYTHON="python3"

# 检查Python是否存在
if ! command -v $PYTHON &> /dev/null; then
    echo "[ERROR] Python未找到: $PYTHON"
    echo "请安装Python3或修改脚本中的PYTHON变量"
    exit 1
fi

echo "[INFO] 使用Python: $PYTHON"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "====================================================="
echo "步骤 1/4: 获取API数据"
echo "====================================================="
echo ""
$PYTHON scripts/fetch_data.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 数据获取失败"
    exit 1
fi
echo ""
echo "[OK] 数据获取完成"
echo ""

echo "====================================================="
echo "步骤 2/4: 创建特征"
echo "====================================================="
echo ""
$PYTHON features/create_features.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 特征创建失败"
    exit 1
fi
echo ""
echo "[OK] 特征创建完成"
echo ""

echo "====================================================="
echo "步骤 3/4: 训练模型"
echo "====================================================="
echo ""
$PYTHON models/train_models.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 模型训练失败"
    exit 1
fi
echo ""
echo "[OK] 模型训练完成"
echo ""

echo "====================================================="
echo "步骤 4/4: 启动Web应用"
echo "====================================================="
echo ""
echo "正在启动Streamlit应用..."
echo "浏览器将自动打开 http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""
$PYTHON -m streamlit run app_api.py
