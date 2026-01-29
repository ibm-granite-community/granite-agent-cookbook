# Testing Your Agent with Test-Driven Development (TDD)

Agents feel powerful in demos—but they often fail silently in production.

A small prompt tweak, a tool schema change, or a model upgrade (for example, migrating from **Granite 3.8** to **Granite 4**) can subtly break behavior without throwing an error. Traditional unit tests don’t catch this.

This guide shows how to apply **test-driven development (TDD)** to AI agents so that:

* Your agent’s behavior is explicit
* Regressions are caught early
* Model or prompt changes are safer

You’ll learn how to build a **lightweight evaluation harness** for a function-calling agent.

> **Estimated time:** 15 minutes
> **Prerequisites:** Basic Python, familiarity with tool/function-calling agents

A complete runnable example is available in the companion notebook:
[test_driven_agent_demo.ipynb](test_driven_agent_demo.ipynb).

---

## Why agent testing is different

Classic software is deterministic. Agents are not.

An agent:

* Is **stateful**
* Is **probabilistic**
* Makes decisions across **multiple steps**
* Can fail in ways that still “look reasonable” to a human

That means:

* “It worked yesterday” is not a guarantee
* Model upgrades (e.g., Granite 3 → Granite 4) can change behavior
* Prompt changes can break tool usage without obvious errors

**Agent TDD gives you guardrails.**

---

## The Agent TDD loop

Agent TDD follows the same philosophy as classic TDD—but the unit of testing is *behavior*, not functions.

```
1. Write a failing test
2. Run the agent
3. Evaluate tool usage + final response
4. Fix prompts / tools / logic
5. Repeat
```

Over time, this becomes a **regression suite** for your agent.

---

## Defining test cases (the foundation)

Agent tests work best when they’re **structured data**, not free-form text.

Each test defines:

* The input
* The expected tool calls (trajectory)
* The expected response characteristics

### Example 1: Single-turn, single tool

```json
{
  "name": "get_weather_only",
  "input": "What is the weather in Boston?",
  "expected_tool_calls": [
    {
      "tool_name": "get_weather",
      "tool_parameters": {
        "city": "Boston",
        "country": "USA"
      }
    }
  ],
  "expected_response_contains": "Sunny"
}
```

This test fails if:

* The wrong tool is called
* The parameters are wrong
* The agent skips the tool entirely

---

### Example 2: Single-turn, multiple tools

```json
{
  "name": "weather_and_hotel",
  "input": "What is the weather in Boston and find me a hotel?",
  "expected_tool_calls": [
    {
      "tool_name": "get_weather",
      "tool_parameters": {"city": "Boston", "country": "USA"}
    },
    {
      "tool_name": "find_hotel",
      "tool_parameters": {
        "city": "Boston",
        "country": "USA",
        "max_budget_per_night": 200
      }
    }
  ],
  "expected_response_contains": "Sunny"
}
```

This ensures:

* The agent reasons correctly
* The order of actions makes sense
* No tool is skipped or hallucinated

---

### Example 3: Multi-turn conversation (real-life scenario)

Think of an **internal operations agent**:

```json
{
  "name": "retrieve_then_email_receipt",
  "turns": [
    {
      "input": "Retrieve my booking called nyc_trip_oct.",
      "expected_tool_calls": [
        {
          "tool_name": "retrieve_booking",
          "tool_parameters": {"booking_name": "nyc_trip_oct"}
        }
      ],
      "expected_response_contains": "Retrieved"
    },
    {
      "input": "Email this receipt to my manager for approval.",
      "expected_tool_calls": [
        {
          "tool_name": "send_email",
          "tool_parameters": {
            "to": "manager@example.com",
            "subject": "Travel approval"
          }
        }
      ],
      "expected_response_contains": "emailed"
    }
  ]
}
```

Each turn is evaluated independently—so failures are easier to diagnose.

---

## Evaluating tool-call trajectories

Trajectory evaluation answers one question:

> Did the agent do *the right thing*?

### Exact match (recommended default)

Use this when correctness matters (writes, deletes, approvals):

```python
def trajectory_match_exact(actual, expected):
    actual_norm = [
        {"tool_name": c.tool_name, "tool_parameters": c.tool_parameters}
        for c in actual
    ]
    return actual_norm == expected
```

### Containment match (use carefully)

Useful when some parameters are flexible (timestamps, optional fields):

```python
def trajectory_match_containment(actual, expected):
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if a.tool_name != e["tool_name"]:
            return False
        for k, v in e.get("tool_parameters", {}).items():
            if a.tool_parameters.get(k) != v:
                return False
    return True
```

**Rule of thumb:**
Start strict → relax only when you have a strong reason.

---

## Evaluating final responses

For early-stage agents, keep this simple.

```python
def response_match(actual, expected_contains):
    return expected_contains.lower() in actual.lower()
```

You can later add:

* Semantic similarity
* LLM-as-a-judge
* Structured outputs

But **don’t start there**. Trajectory correctness matters more.

---

## Tracking non-functional metrics (often overlooked)

Agents can “work” while still getting worse.

Track these early:

| Metric      | Why it matters                      |
| ----------- | ----------------------------------- |
| Latency     | Detects slow reasoning chains       |
| Token usage | Catches prompt bloat and cost creep |

This becomes critical when scaling or switching models.

---

## Minimal test runner

```python
def run_tests(agent, tests):
    results = []
    for test in tests:
        output = agent.run(test["input"])
        results.append({
            "name": test["name"],
            "trajectory_ok": trajectory_match_exact(
                output.tool_calls,
                test["expected_tool_calls"]
            ),
            "response_ok": response_match(
                output.final_response,
                test["expected_response_contains"]
            ),
            "latency_ms": round(output.latency_ms, 2),
            "total_tokens": output.total_tokens,
        })
    return results
```

Each test produces a **clear, debuggable signal**.

---

## Why this matters for Granite 4

Granite 4 is:

* More literal
* More sensitive to prompt structure
* Less forgiving of ambiguous instructions

Agent TDD:

* Makes failures visible
* Prevents silent regressions
* Gives confidence during model or prompt migrations

---

## Summary

| Step | Action                                         |
| ---- | ---------------------------------------------- |
| 1    | Define agent behavior as structured test cases |
| 2    | Validate tool-call trajectories                |
| 3    | Validate final responses                       |
| 4    | Track latency and token usage                  |
| 5    | Grow a regression suite over time              |

**Start small.**
A handful of well-chosen tests will save you far more time than ad-hoc debugging in production.
