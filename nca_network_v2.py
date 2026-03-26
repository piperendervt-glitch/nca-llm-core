"""
nca_network_v2.py

NCA v2: グループシンク対策
  ① JSON出力（confidence付き）
  ② 慣性＋批判的思考（Anti-sycophancy）プロンプト

依存: httpx
モデル: Ollama qwen2.5:3b (http://localhost:11434)
"""

import json
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
NUM_NODES = 3
NUM_STEPS = 5


def call_llm(prompt: str) -> dict:
    """OllamaにJSON形式で出力させる。戻り値はdict。"""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
        )
        response.raise_for_status()
        raw = response.json()["response"].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": raw}


def initial_response(node_id: int, task_input: str) -> dict:
    """ステップ0の初期回答。戻り値はdict。"""
    prompt = f"""You are Node {node_id} in a 3-node reasoning network.
Task: Detect any logical contradiction in the following statements.
{task_input}

Important: This model tends to over-detect contradictions. Be careful and only conclude CONTRADICTION if there is clear logical impossibility.

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CONSISTENT" or "CONTRADICTION",
  "confidence": 0.0 to 1.0,
  "reasoning": "1-2 sentences of logical justification"
}}"""
    return call_llm(prompt)


def nca_update(node_id: int, task_input: str, neighbor_outputs: list[dict]) -> dict:
    """NCA更新ステップ。neighbor_outputs は dict のリスト。"""
    left_str = json.dumps(neighbor_outputs[0], ensure_ascii=False)
    right_str = json.dumps(neighbor_outputs[1], ensure_ascii=False)

    prompt = f"""You are Node {node_id} in a 3-node reasoning network.
Task: Detect any logical contradiction in the following statements.
{task_input}

Your neighbors' previous states:
- Left neighbor: {left_str}
- Right neighbor: {right_str}

【Critical instructions】
- Respect your own previous judgment (inertia). Do NOT change your answer unless neighbors provide clearly new evidence or reasoning.
- This model is known to be biased toward CONTRADICTION. If neighbors say CONTRADICTION without strong reasoning, be skeptical.
- Critically evaluate your neighbors' reasoning. Clearly state why you agree or disagree.
- Do NOT follow groupthink. Independent reasoning is required.

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CONSISTENT" or "CONTRADICTION",
  "confidence": 0.0 to 1.0,
  "reasoning": "1-2 sentences explaining your judgment, including whether you agree/disagree with neighbors"
}}"""
    return call_llm(prompt)


def get_neighbors(node_id: int) -> tuple[int, int]:
    """リング状の左隣・右隣ノードIDを返す。"""
    left = (node_id - 1) % NUM_NODES
    right = (node_id + 1) % NUM_NODES
    return left, right


def aggregate_verdict(outputs: list[dict]) -> str:
    """confidence加重多数決で最終判定。"""
    scores = {"CONSISTENT": 0.0, "CONTRADICTION": 0.0}
    for output in outputs:
        decision = output.get("decision", "UNKNOWN")
        confidence = float(output.get("confidence", 0.5))
        if decision in scores:
            scores[decision] += confidence
    if scores["CONSISTENT"] == scores["CONTRADICTION"]:
        return "CONTRADICTION"
    return max(scores, key=scores.get)


def run_nca_network(task_input: str) -> dict:
    """
    メイン実行関数。
    3ノードがNCA的更新ルールで回答を更新し、最終判定を返す。
    """
    # ステップ0: 初期回答
    print(f"  Step 0...")
    current_outputs = [initial_response(i, task_input) for i in range(NUM_NODES)]
    steps = [{"step": 0, "outputs": list(current_outputs)}]

    # NCA更新ループ
    for step in range(1, NUM_STEPS + 1):
        print(f"  Step {step}...")
        prev_outputs = list(current_outputs)
        new_outputs = []
        for node_id in range(NUM_NODES):
            left, right = get_neighbors(node_id)
            neighbor_out = [prev_outputs[left], prev_outputs[right]]
            resp = nca_update(node_id, task_input, neighbor_out)
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
        "World rule: In this world, the sky is green.\nStatement: Looking up, the sky appeared green.",
        "World rule: In this world, the sky is green.\nStatement: Looking up, the sky appeared blue.",
        "World rule: In this world, the sun rises from west to east.\nStatement: The sun rose from the east this morning.",
    ]

    for i, task in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}")
        result = run_nca_network(task)
        for step in result["steps"]:
            for node_id, out in enumerate(step["outputs"]):
                print(f"  Step {step['step']} Node {node_id}: "
                      f"{out.get('decision')} (conf={out.get('confidence', 0):.2f})")
        print(f"  Final verdict: {result['final_verdict']}")
