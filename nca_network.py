"""
nca_network.py

NCA（Neural Cellular Automata）的更新ルールによる3ノードLLMネットワーク。
各ノードは隣接ノードの直前出力のみを入力として自分の回答を更新する。

依存: httpx, scipy
モデル: Ollama qwen2.5:3b (http://localhost:11434)
タスク: world_consistency（矛盾検出）
"""

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
NUM_NODES = 3
NUM_STEPS = 5


def call_llm(prompt: str) -> str:
    """Ollamaにhttpxで同期リクエストを送り、テキストを返す。stream=False。"""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"].strip()


def initial_response(node_id: int, task_input: str) -> str:
    """ステップ0: 隣接情報なし、タスク入力だけで初期回答を生成。"""
    prompt = (
        f"You are Node {node_id} in a 3-node reasoning network.\n"
        f"Task: Detect any logical contradiction in the following statements.\n"
        f"Statements: {task_input}\n"
        f"Respond with: CONSISTENT or CONTRADICTION, followed by one sentence explanation."
    )
    return call_llm(prompt)


def nca_update(node_id: int, task_input: str, neighbor_outputs: list[str]) -> str:
    """NCA的更新ステップ。隣接ノードの直前出力を見て自分の回答を更新する。"""
    prompt = (
        f"You are Node {node_id} in a 3-node reasoning network.\n"
        f"Task: Detect any logical contradiction in the following statements.\n"
        f"Statements: {task_input}\n"
        f"\n"
        f"Your neighbors' previous answers:\n"
        f"- Left neighbor: {neighbor_outputs[0]}\n"
        f"- Right neighbor: {neighbor_outputs[1]}\n"
        f"\n"
        f"Update your answer based on your neighbors' reasoning.\n"
        f"Respond with: CONSISTENT or CONTRADICTION, followed by one sentence explanation."
    )
    return call_llm(prompt)


def get_neighbors(node_id: int) -> tuple[int, int]:
    """リング状の左隣・右隣ノードIDを返す。"""
    left = (node_id - 1) % NUM_NODES
    right = (node_id + 1) % NUM_NODES
    return left, right


def aggregate_verdict(outputs: list[str]) -> str:
    """最終ステップの全ノード出力から多数決で最終判定を返す。"""
    counts = {"CONTRADICTION": 0, "CONSISTENT": 0}
    for output in outputs:
        upper = output.upper()
        if "CONTRADICTION" in upper:
            counts["CONTRADICTION"] += 1
        elif "CONSISTENT" in upper:
            counts["CONSISTENT"] += 1
    if counts["CONTRADICTION"] > counts["CONSISTENT"]:
        return "CONTRADICTION"
    elif counts["CONSISTENT"] > counts["CONTRADICTION"]:
        return "CONSISTENT"
    return "UNKNOWN"


def run_nca_network(task_input: str) -> dict:
    """
    メイン実行関数。
    3ノードがNCA的更新ルールで回答を更新し、最終判定を返す。
    """
    steps = []

    # ステップ0: 初期回答
    print(f"Step 0: generating initial responses...")
    current_outputs = []
    for node_id in range(NUM_NODES):
        resp = initial_response(node_id, task_input)
        print(f"  Node {node_id}: {resp}")
        current_outputs.append(resp)
    steps.append({"step": 0, "outputs": list(current_outputs)})

    # NCA更新ステップ
    for step in range(1, NUM_STEPS + 1):
        print(f"Step {step}: updating...")
        prev_outputs = list(current_outputs)  # 同期更新のためコピー
        new_outputs = []
        for node_id in range(NUM_NODES):
            left, right = get_neighbors(node_id)
            neighbor_out = [prev_outputs[left], prev_outputs[right]]
            resp = nca_update(node_id, task_input, neighbor_out)
            print(f"  Node {node_id}: {resp}")
            new_outputs.append(resp)
        current_outputs = new_outputs
        steps.append({"step": step, "outputs": list(current_outputs)})

    final_verdict = aggregate_verdict(current_outputs)

    return {
        "task_input": task_input,
        "steps": steps,
        "final_verdict": final_verdict,
        "num_steps": NUM_STEPS,
    }


if __name__ == "__main__":
    test_cases = [
        # 矛盾あり
        "The cat is alive. The cat is dead. The box has never been opened.",
        # 矛盾なし
        "It rained yesterday. The ground is wet. The forecast predicted rain.",
        # 矛盾あり
        "John is taller than Mary. Mary is taller than John. They are the same height.",
    ]

    for i, task in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {task}")
        result = run_nca_network(task)
        for step in result["steps"]:
            print(f"  Step {step['step']}: {step['outputs']}")
        print(f"  Final verdict: {result['final_verdict']}")
