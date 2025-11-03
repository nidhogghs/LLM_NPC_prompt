# -*- coding: utf-8 -*-
import os
import json
import types
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import gradio as gr

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models


# ===================== 基本配置（可按需修改） =====================
prompts_dir = "prompts"          # 人设XML文件夹
model_list_path = "models.json"  # 模型列表（仅名称的数组），不存在则回退到默认
default_models = ["hunyuan-a13b", "hunyuan-standard", "hunyuan-pro"]
log_dir = "logs"                 # 日志保存目录
default_temperature = 0.7
default_max_tokens = 512


# ===================== 工具函数 =====================
def ensure_dir(p: str | Path) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_models_from_file(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return default_models
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass
    return default_models

def scan_personas(dirpath: str | Path) -> List[str]:
    p = Path(dirpath)
    if not p.exists():
        return []
    # 返回相对 prompts_dir 的路径，支持子目录
    return [str(x.relative_to(p)) for x in p.glob("**/*.xml")]

def load_system_xml(full_path: str | Path) -> str:
    p = Path(full_path)
    if not p.exists():
        raise FileNotFoundError(f"System XML not found: {p.resolve()}")
    return p.read_text(encoding="utf-8")

def merge_personas(rel_paths: list[str] | None, base_dir: str | Path) -> str:
    """
    将多个人设XML按传入顺序合并为一个system文本。
    - 用 XML 注释标注来源文件与顺序；
    - 若为空，返回空字符串。
    """
    if not rel_paths:
        return ""
    parts = []
    for rel in rel_paths:
        full = Path(base_dir, rel)
        if not full.exists():
            raise FileNotFoundError(f"System XML not found: {full.resolve()}")
        xml = full.read_text(encoding="utf-8")
        parts.append(f"<!-- BEGIN: {rel} -->\n{xml}\n<!-- END: {rel} -->")
    header = (
        "<!--\n"
        "  Multiple persona XML merged.\n"
        "  NOTE: Later files override earlier ones when rules conflict.\n"
        "-->\n"
    )
    return header + "\n\n".join(parts)

def make_client():
    # 需要环境变量：TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY
    cred = credential.Credential(
        os.getenv("TENCENTCLOUD_SECRET_ID"),
        os.getenv("TENCENTCLOUD_SECRET_KEY"),
    )
    http_profile = HttpProfile()
    http_profile.endpoint = "hunyuan.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    return hunyuan_client.HunyuanClient(cred, "", client_profile)

def call_hunyuan_chat(client, model_name: str, messages: List[Dict[str, str]],
                      temperature: float, max_tokens: int) -> str:
    req = models.ChatCompletionsRequest()
    params = {
        "Model": model_name,
        "Messages": messages,
        "Temperature": float(temperature),
        "MaxTokens": int(max_tokens),
        # "Stream": True,  # 若需要流式，可开启并改造UI回调
    }
    req.from_json_string(json.dumps(params, ensure_ascii=False))
    resp = client.ChatCompletions(req)

    # 兼容流式/非流式
    if isinstance(resp, types.GeneratorType):
        chunks = []
        for event in resp:
            try:
                data = json.loads(event["Data"])
                delta = data.get("Choices", [{}])[0].get("Delta", {}).get("Content", "")
                chunks.append(delta)
            except Exception:
                pass
        return "".join(chunks)
    else:
        data = json.loads(resp.to_json_string())
        return data["Choices"][0]["Message"]["Content"]

def format_dialogue_as_text(history: List[Dict[str, str]],
                            model_name: str,
                            persona_file: str | list[str] | None,
                            started_at: datetime,
                            ended_at: datetime) -> str:
    lines = []
    lines.append("=== Goblin Chat Log ===")
    lines.append(f"Model: {model_name}")
    if isinstance(persona_file, list):
        lines.append("PersonaXMLs:")
        for p in persona_file:
            lines.append(f"  - {Path(prompts_dir, p).resolve()}")
    else:
        lines.append(f"PersonaXML: {Path(prompts_dir, persona_file).resolve() if persona_file else 'None'}")
    lines.append(f"StartedAt: {started_at.isoformat(timespec='seconds')}")
    lines.append(f"EndedAt:   {ended_at.isoformat(timespec='seconds')}")
    lines.append("=" * 28)
    lines.append("")
    for i, msg in enumerate(history):
        role = msg.get("Role", "unknown")
        content = msg.get("Content", "")
        lines.append(f"[{i:03d}] Role: {role}")
        lines.append(content)
        lines.append("-" * 28)
    return "\n".join(lines) + "\n"

def save_dialogue(history: List[Dict[str, str]], model_name: str, persona_file: str | list[str] | None,
                  started_at: datetime, ended_at: datetime) -> Path:
    ensure_dir(log_dir)
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    fname = f"goblin_chat_{stamp}.txt"
    path = Path(log_dir) / fname
    text = format_dialogue_as_text(history, model_name, persona_file, started_at, ended_at)
    path.write_text(text, encoding="utf-8")
    return path


# ===================== 会话状态与逻辑 =====================
def init_state() -> Dict[str, Any]:
    """初始化会话状态。"""
    return {
        "client": None,
        "history": [],            # [{"Role": "...", "Content": "..."}]
        "started_at": None,
        "model": None,
        "persona_file": [],       # 记录使用的人设文件列表
        "saved": False,
    }

def start_session(model_name: str, persona_rel_paths: list[str] | None,
                  temperature: float, max_tokens: int):
    """
    点击“开始会话”：
    - 初始化client
    - 合并多个人设XML并作为 system 注入
    - 清空 UI 聊天框，返回欢迎语与 state
    """
    state = init_state()
    try:
        client = make_client()
    except Exception as e:
        return gr.update(value=f"【错误】SDK初始化失败：{e}"), [], state

    history = []
    try:
        system_xml = merge_personas(persona_rel_paths, prompts_dir)
        if system_xml.strip():
            history.append({"Role": "system", "Content": system_xml})
    except Exception as e:
        return gr.update(value=f"【错误】读取/合并人设失败：{e}"), [], state

    state["client"] = client
    state["history"] = history
    state["started_at"] = datetime.now()
    state["model"] = model_name
    state["persona_file"] = persona_rel_paths or []
    state["saved"] = False

    welcome = "会话已开始。现在可以在下方输入框对话。"
    return welcome, [], state

def chat_reply(user_message: str, chat_history_ui: List[List[str]],
               state: Dict[str, Any], temperature: float, max_tokens: int):
    """
    把用户消息写入 Hunyuan 历史，调用模型，写回助手消息。
    返回：(错误文本或None, 更新后的UI聊天记录, 更新后的state)
    """
    if not state or not state.get("client"):
        return "【错误】请先点击“开始会话”", chat_history_ui, state

    client = state["client"]
    model_name = state["model"]
    history = state["history"]

    # 追加用户消息
    history.append({"Role": "user", "Content": user_message})

    try:
        assistant_out = call_hunyuan_chat(
            client=client,
            model_name=model_name,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except TencentCloudSDKException as e:
        # 回滚用户这条，避免污染上下文
        history.pop()
        return f"【SDK错误】{e}", chat_history_ui, state
    except Exception as e:
        history.pop()
        return f"【错误】{e}", chat_history_ui, state

    # 写回历史
    history.append({"Role": "assistant", "Content": assistant_out})
    state["history"] = history

    # 更新UI
    chat_history_ui = chat_history_ui + [[user_message, assistant_out]]
    return None, chat_history_ui, state

def on_send(user_message, chatbot, state, temperature, max_tokens):
    """
    发送按钮/回车 的回调：
    - 传递到 chat_reply
    - 返回：清空输入框、更新后的chatbot、state
    """
    chatbot = chatbot or []
    err, updated_chat, state = chat_reply(
        user_message, chatbot, state, temperature, max_tokens
    )
    if err:
        updated_chat = chatbot + [[user_message, err]]
    return "", updated_chat, state

def end_and_save(state: Dict[str, Any]):
    """
    点击“结束并保存”：
    - 保存整个 history 到 logs/
    """
    if not state or not state.get("history"):
        return "当前没有会话内容可保存。"

    if state.get("saved"):
        return "本次会话已保存，无需重复保存。"

    try:
        save_path = save_dialogue(
            history=state["history"],
            model_name=state.get("model") or "unknown",
            persona_file=state.get("persona_file"),
            started_at=state.get("started_at") or datetime.now(),
            ended_at=datetime.now(),
        )
        state["saved"] = True
        return f"已保存到：{save_path.resolve()}"
    except Exception as e:
        return f"保存失败：{e}"

def refresh_personas():
    """重新扫描 prompts/ 下的人设XML文件（多选）。"""
    files = scan_personas(prompts_dir)
    if not files:
        return gr.update(choices=[], value=[]), "未在 prompts/ 下发现 .xml 文件"
    # 默认不选；也可改为 value=[files[0]]
    return gr.update(choices=files, value=[]), f"共发现 {len(files)} 个XML。"


# ===================== 构建 UI =====================
def build_ui():
    model_options = load_models_from_file(model_list_path)
    persona_files = scan_personas(prompts_dir)

    with gr.Blocks(title="Goblin Panel (Hunyuan)") as demo:
        gr.Markdown("## Goblin 面板：选择模型 + 多个人设（合并为System）+ 对话 + 自动保存")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="模型（Model）",
                choices=model_options,
                value=model_options[0]
            )
            persona_multi = gr.CheckboxGroup(
                label="人设（可多选，按勾选顺序合并）",
                choices=persona_files,
                value=[],
            )
            refresh_button = gr.Button("重新扫描人设")

        with gr.Row():
            temperature = gr.Slider(0.0, 2.0, value=default_temperature, step=0.05, label="温度（Temperature）")
            max_tokens = gr.Slider(64, 4096, value=default_max_tokens, step=64, label="最大输出Tokens（MaxTokens）")

        start_msg = gr.Markdown("")
        with gr.Row():
            start_btn = gr.Button("开始会话", variant="primary")
            end_btn = gr.Button("结束并保存")
        save_info = gr.Markdown("")

        # 聊天区域（兼容各种 Gradio 版本）
        chatbot = gr.Chatbot(label="对话窗口（Chat）")
        with gr.Row():
            user_input = gr.Textbox(
                label="发送消息（Enter 发送）",
                show_label=False,
                placeholder="输入内容后回车或点发送",
                lines=2
            )
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话（仅UI）")

        # 全局状态
        state = gr.State(init_state())

        # 事件绑定
        start_btn.click(
            start_session,
            inputs=[model_dropdown, persona_multi, temperature, max_tokens],
            outputs=[start_msg, chatbot, state]
        )

        end_btn.click(end_and_save, inputs=[state], outputs=[save_info])

        refresh_button.click(
            refresh_personas,
            inputs=[],
            outputs=[persona_multi, start_msg]
        )

        send_btn.click(
            on_send,
            inputs=[user_input, chatbot, state, temperature, max_tokens],
            outputs=[user_input, chatbot, state]
        )
        user_input.submit(
            on_send,
            inputs=[user_input, chatbot, state, temperature, max_tokens],
            outputs=[user_input, chatbot, state]
        )

        clear_btn.click(lambda: [], inputs=None, outputs=chatbot)

        gr.Markdown(
            """
            **使用说明**
            1. 准备 `prompts/` 文件夹，放置人设 XML（支持子目录）；可多选合并为一条 system。
            2. （可选）在同目录放 `models.json`（形如 `["hunyuan-a13b","hunyuan-pro"]`），否则使用内置列表。
            3. 设置环境变量 `TENCENTCLOUD_SECRET_ID` / `TENCENTCLOUD_SECRET_KEY`。
            4. 选择模型与人设（可多选），点 **开始会话**；在下方输入框发送消息。
            5. 结束后点 **结束并保存**，日志默认保存在 `logs/`。
            - 冲突规则：多个人设合并时，**后勾选**的规则文本在后，语义冲突时可视为“后者优先”。
            """
        )

    return demo


# ===================== 入口 =====================
if __name__ == "__main__":
    ensure_dir(prompts_dir)
    ensure_dir(log_dir)
    ui = build_ui()
    # 本机访问：默认 http://127.0.0.1:7860
    ui.launch()
