# -*- coding: utf-8 -*-
import os
import json
import types
from pathlib import Path
from datetime import datetime

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models


# ====== 根据你的项目路径修改 ======
XML_PATH = "prompts/goblin.xml"   # 系统提示词 XML 路径
MODEL_NAME = "hunyuan-a13b"       # 可换：hunyuan-standard / hunyuan-pro
MAX_TOKENS = 512
TEMPERATURE = 0.7
LOG_DIR = "logs"                  # 对话日志保存目录（会自动创建）


def load_system_xml(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"System XML not found: {p.resolve()}")
    return p.read_text(encoding="utf-8")


def make_client():
    # 从环境变量读取密钥：TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY
    cred = credential.Credential(
        os.getenv("TENCENTCLOUD_SECRET_ID"),
        os.getenv("TENCENTCLOUD_SECRET_KEY"),
    )
    httpProfile = HttpProfile()
    httpProfile.endpoint = "hunyuan.tencentcloudapi.com"

    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile

    return hunyuan_client.HunyuanClient(cred, "", clientProfile)


def chat_once(client, messages):
    """调用一次 ChatCompletions，返回 assistant 文本。"""
    req = models.ChatCompletionsRequest()

    params = {
        "Model": MODEL_NAME,
        "Messages": messages,
        "Temperature": TEMPERATURE,
        "MaxTokens": MAX_TOKENS,
        # "Stream": True,  # 需要流式再打开
    }
    req.from_json_string(json.dumps(params, ensure_ascii=False))

    resp = client.ChatCompletions(req)

    # 兼容官方的流式/非流式处理
    if isinstance(resp, types.GeneratorType):  # 流式响应
        chunks = []
        for event in resp:
            try:
                data = json.loads(event["Data"])
                delta = data.get("Choices", [{}])[0].get("Delta", {}).get("Content", "")
                print(delta, end="", flush=True)
                chunks.append(delta)
            except Exception:
                pass
        print()
        return "".join(chunks)
    else:  # 非流式
        data = json.loads(resp.to_json_string())
        return data["Choices"][0]["Message"]["Content"]


def ensure_log_dir() -> Path:
    """确保日志目录存在，返回 Path 对象。"""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def default_log_path(start_time: datetime) -> Path:
    """根据会话开始时间生成默认日志文件路径。"""
    stamp = start_time.strftime("%Y%m%d_%H%M%S")
    fname = f"goblin_chat_{stamp}.txt"
    return ensure_log_dir() / fname


def format_dialogue_as_text(history, model_name: str, xml_path: str,
                            started_at: datetime, ended_at: datetime) -> str:
    """把对话 history 格式化为纯文本（含元信息与system内容）。"""
    lines = []
    lines.append("=== Goblin Chat Log ===")
    lines.append(f"Model: {model_name}")
    lines.append(f"XML: {Path(xml_path).resolve()}")
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


def save_dialogue(history, model_name: str, xml_path: str,
                  started_at: datetime, ended_at: datetime, path: Path = None) -> Path:
    """保存对话到 TXT 文件，返回保存路径。"""
    if path is None:
        path = default_log_path(started_at)

    text = format_dialogue_as_text(history, model_name, xml_path, started_at, ended_at)
    path.write_text(text, encoding="utf-8")
    return path


def main():
    history = []  # 放在最外层，便于 finally 中访问
    session_started_at = datetime.now()
    saved = False  # 防重复保存

    try:
        # 1) 读取系统提示词（XML）
        system_xml = load_system_xml(XML_PATH)

        # 2) 初始化客户端
        client = make_client()

        # 3) 初始化对话历史（第一条是 system）
        history = [
            {"Role": "system", "Content": system_xml}
        ]

        print("Goblin chat ready. Type 'exit' to quit.\n")
        while True:
            try:
                user_inp = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Info] Session interrupted.")
                break

            if user_inp.lower() in ("exit", "quit"):
                print("Bye.")
                break

            # 追加用户发言
            history.append({"Role": "user", "Content": user_inp})

            # 调用一次模型
            try:
                assistant_out = chat_once(client, history)
            except TencentCloudSDKException as e:
                print(f"[SDK Error] {e}")
                # 删除刚刚追加的用户消息，避免把失败轮写进上下文
                history.pop()
                continue

            # 打印并写回上下文
            print(f"Goblin: {assistant_out}\n")
            history.append({"Role": "assistant", "Content": assistant_out})

            # 可选：滑动窗口，避免上下文过长（保留最近8轮 user/assistant，对 system 不动）
            MAX_TURNS = 8
            non_sys = [m for m in history if m["Role"] != "system"]
            if len(non_sys) > 2 * MAX_TURNS:
                history = [history[0]] + non_sys[-2 * MAX_TURNS:]

    except FileNotFoundError as e:
        print(e)
    except TencentCloudSDKException as err:
        print(err)
    finally:
        # 退出时自动保存
        if history:
            try:
                save_path = save_dialogue(
                    history=history,
                    model_name=MODEL_NAME,
                    xml_path=XML_PATH,
                    started_at=session_started_at,
                    ended_at=datetime.now(),
                )
                saved = True
                print(f"[Saved] Dialogue saved to: {save_path.resolve()}")
            except Exception as se:
                print(f"[Save Error] {se}")
        else:
            print("[Info] No dialogue to save.")


if __name__ == "__main__":
    main()
