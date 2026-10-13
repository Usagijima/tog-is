# -*- coding: utf-8 -*-
"""
第七步：vLLM 启动后的 smoke test（方案第 19 节 + 20.4 节容量测量）

前提：vLLM 服务已经在本机 8000 端口运行：
    vllm serve Qwen/Qwen3-4B-Instruct-2507-AWQ --max-model-len 8192 --port 8000

本脚本做三件事：
  1. 发一条真实请求，确认模型能生成中文
  2. 测输出吞吐（token/秒）——代入方案 20.4 节公式判断一个月跑不跑得完
  3. 把结果写入 data/smoke_test.json（实验记录素材）

用法：
    python step7_smoke.py
"""

import json
import os
import time
import urllib.error
import urllib.request

DATA_DIR = "data"
URL = "http://localhost:8000/v1/chat/completions"


def served_model_id():
    """向服务器询问它实际服务的模型名，避免名字对不上。"""
    with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=30) as resp:
        models = json.loads(resp.read())["data"]
    if not models:
        raise RuntimeError("服务器上没有加载任何模型，请先在 Bionic 里加载 Qwen3-4B")
    print(f"  服务器上的模型: {[m['id'] for m in models]}")
    return models[0]["id"]


def chat(messages, max_tokens=256, temperature=0.2):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    n_tokens, text, t0 = 0, "", time.time()
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"服务器返回 {e.code}: {body}")
    with resp_ctx as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            text += delta
            if delta:
                n_tokens += 1  # 流式逐 chunk 计数（近似值，够用）
    elapsed = time.time() - t0
    return text, n_tokens, elapsed


MODEL_ID = None  # main() 里自动探测


def main():
    global MODEL_ID
    # ---- 0. 探测服务器上的模型 ----
    print("[0/3] 探测服务器模型 ...")
    MODEL_ID = served_model_id()

    # ---- 1. 生成测试 ----
    print("[1/3] 发送测试请求 ...")
    text, n, t = chat([{"role": "user", "content": "用一句话解释什么是'选择性披露'"}])
    print(f"  模型回答: {text.strip()[:100]}")
    print(f"  耗时 {t:.1f} 秒, 约 {n} 个输出 token, 吞吐约 {n / t:.1f} token/s")
    if not text.strip():
        print("  [FAIL] 模型没有返回内容")

    # ---- 2. 批量压测（近似实验负载：短输入 + 短输出）----
    print("[2/3] 模拟实验负载批量测速（4 条并发×3 轮） ...")
    import threading

    results = []

    def worker():
        _, nn, tt = chat(
            [{"role": "user", "content": "把这句话改写成 JSON：我推荐行动 a_03"}],
            max_tokens=64,
        )
        results.append((nn, tt))

    threads = []
    for _ in range(3):
        for _ in range(4):
            th = threading.Thread(target=worker)
            th.start()
            threads.append(th)
        for th in threads[-4:]:
            th.join()
    total_tokens = sum(r[0] for r in results)
    wall = max(r[1] for r in results) if results else 0
    # 并发吞吐 = 总 token / 最慢一条的耗时（近似）
    throughput = total_tokens / wall if wall else 0
    print(f"  并发吞吐约 {throughput:.1f} token/s (4 并发)")

    # ---- 3. 代入方案 20.4 节容量公式 ----
    print("[3/3] 容量估算 ...")
    T_out, T_in = 30_720_000, 245_760_000          # 方案 20.3 节 token 上限
    r_d = throughput                                  # 实测 decode 吞吐（近似）
    r_p = 1500                                        # 输入吞吐先按经验值，实测后替换
    hours = 1.3 / 3600 * (T_out / r_d + T_in / r_p)
    print(f"  按 decode={r_d:.0f} token/s 估算, 全量约需 {hours:.0f} GPU 小时 (预算 400)")
    verdict = "可行 ✓" if hours <= 400 else "超预算, 需降档或租云端卡"
    print(f"  结论: {verdict}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "smoke_test.json"), "w", encoding="utf-8") as f:
        json.dump({"single_token_s": round(n / t, 1),
                   "concurrent_4_token_s": round(throughput, 1),
                   "est_gpu_hours": round(hours, 1),
                   "gpu": "RTX 5060 8GB", "model": "Qwen3-4B-Instruct-2507-AWQ",
                   "datetime": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    print("结果已写入 data/smoke_test.json")


if __name__ == "__main__":
    main()
