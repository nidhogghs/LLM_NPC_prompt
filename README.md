# LLM_NPC_prompt

基于 **Tencent Cloud Hunyuan（腾讯混元）Python SDK** 与 **Gradio** 的简易对话/提示工程示例工程。  
支持本地运行、调用腾讯云混元大模型接口，并通过 Gradio 提供一个轻量的 Web 界面。

## 功能特点

- 使用 `tencentcloud-sdk-python` 的混元客户端发起对话补全请求  
- 封装基础的会话与提示管理（`messages`）  
- 提供一键启动的 Gradio 界面，方便快速迭代提示词  
- 代码结构清晰，便于集成到你的项目中

## 环境要求

- Python 3.9+
- 已开通腾讯云账号并创建 **API 密钥**（SecretId/SecretKey）
- 已在控制台开通/申请对应的混元模型调用权限与配额

## 安装

```bash
# 1) 克隆本仓库
git clone https://github.com/nidhogghs/LLM_NPC_prompt.git
cd LLM_NPC_prompt

# 2) 创建并激活虚拟环境（可选）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3) 安装依赖
pip install -U pip
pip install -r requirements.txt
