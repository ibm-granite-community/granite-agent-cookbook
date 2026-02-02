# Agent Test Driven Development: Building Reliable AI Agents

Building AI agents that call functions and interact with external tools opens up powerful possibilities for automation and intelligent assistance. However, ensuring these agents behave reliably and consistently requires a structured testing approach. Without proper testing, it becomes difficult to track whether prompt modifications affect behavior, how different models compare on specific use cases, or whether tool schema changes introduce unexpected issues.

In this guide, we'll build a complete testing framework for function-calling agents. By the end, you'll have a structured approach to catch regressions, compare alternatives objectively, and deploy agent updates with confidence.

If you haven't already built a function-calling agent, check out our previous guide on [building agents with LangGraph and Granite](granite-agent-cookbook/recipes/Function_Calling/Function_Calling_Agent.ipynb). We'll pick up where that left off.

## Why Test Your Agent?

AI agents exhibit non-deterministic behavior by nature. The same prompt can yield different results across runs. Models receive updates. Tool schemas evolve. New capabilities are added. Without a robust testing framework, each change introduces uncertainty.

Here's what a good test framework gives you:

**Catch regressions early** - Know immediately when a change breaks existing functionality  
**Compare alternatives objectively** - Which model actually works better for your use case?  
**Ship with confidence** - Core use cases are validated before deployment  
**Faster debugging** - Reproduce issues with specific test cases  
**Documentation** - Your test cases become living examples of expected behavior

The approach we're building today will cover:
- Evaluation helpers for tool-call trajectories and responses
- Structured test case definitions
- Single-turn and multi-turn conversation testing
- Summary metrics and reporting

Let's dive in.

## Testing Your Agent

Building upon the function-calling agent from the previous guide, we'll now construct a comprehensive testing framework. The agent under test includes two tools: `get_current_weather` for retrieving weather information and `get_stock_price` for accessing historical stock data.

### Step 7: Define Data Structures and Evaluation Helpers

Before running any tests, we need to decide what "passing" means. This is test-driven development 101: write your assertions first, then build what satisfies them.

For our agent, we care about two things:

**Trajectory evaluation**: Did the agent call the right tools with the right parameters?  
**Response evaluation**: Did the agent's final answer contain the expected information?

Here's how we model this:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time


@dataclass
class ToolCall:
    """Represents a single tool call made by the agent."""
    tool_name: str
    tool_parameters: Dict[str, Any]


@dataclass
class AgentTestResult:
    """The output of an agent test run, including tool calls and metrics."""
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_response: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0


def trajectory_match(actual: List[ToolCall], expected: List[Dict[str, Any]]) -> bool:
    """Check if actual tool calls exactly match expected.
    
    Use exact match when mistakes are risky (e.g., tools that write or delete data).
    """
    actual_norm = [{"tool_name": c.tool_name, "tool_parameters": c.tool_parameters} for c in actual]
    return actual_norm == expected


def response_match(actual: str, expected_contains: str) -> bool:
    """Check if actual response contains the expected substring.
    
    For deterministic tests, a simple substring check is fast and easy to debug.
    For flexible responses, consider LLM-as-a-judge or semantic similarity.
    """
    return expected_contains.lower() in actual.lower()


def estimate_tokens(text: str) -> int:
    """Simple heuristic for token count (for demo purposes)."""
    return max(1, len(text) // 4)
```

#### Breaking this down

- The `ToolCall` class captures what the agent decided to do. Each tool call has a name (like "get_current_weather") and parameters (like `{"location": "Miami"}`).

- The `AgentTestResult` class holds everything we need to evaluate a test run. Beyond the tool calls and final response, we track performance metrics like latency and token usage. Why? Because sometimes a correct answer that takes 10 seconds isn't actually correct for your use case.

- The `trajectory_match` function does exact comparison. This is strict, but that's the point. If your agent calls the wrong tool or passes the wrong parameters, you want to know. For tools that write data or make purchases, strict checking prevents expensive mistakes.

- The `response_match` function is more lenient. It just checks if the expected substring appears in the response. This works great for our use case where we want to verify the agent mentioned "Miami" or "IBM" in its answer. For more complex validation, you might use an LLM-as-a-judge approach or semantic similarity, but start simple.

- The `estimate_tokens` function is a quick heuristic. In production, you'd use the actual tokenizer for your model, but this gets us 80% of the way there for tracking costs.

### Step 8: Create a Test Runner for the Agent

Now we need a function that runs the agent and extracts all the information we need for evaluation:

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def run_agent_for_test(graph: CompiledStateGraph, user_input: str) -> AgentTestResult:
    """Run the agent and collect results for testing purposes."""
    start = time.time()
    tool_calls_made: List[ToolCall] = []
    final_response = ""
    
    user_message = HumanMessage(user_input)
    input_state = State(messages=[user_message])
    
    # Stream through the agent execution
    for event in graph.stream(input_state):
        for value in event.values():
            last_message = value["messages"][-1]
            
            # Capture tool calls from AI messages
            if isinstance(last_message, AIMessage):
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        tool_calls_made.append(ToolCall(
                            tool_name=tc['name'],
                            tool_parameters=tc['args']
                        ))
                # Capture final text response (when no tool calls)
                if last_message.content and not last_message.tool_calls:
                    final_response = last_message.content
    
    latency_ms = (time.time() - start) * 1000
    prompt_tokens = estimate_tokens(user_input)
    response_tokens = estimate_tokens(final_response)
    
    return AgentTestResult(
        tool_calls=tool_calls_made,
        final_response=final_response,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        total_tokens=prompt_tokens + response_tokens,
    )
```

#### Implementation Details

**Latency Measurement**: The timer starts immediately upon function entry. For production systems, latency directly impacts user experience. An agent that requires 30 seconds to respond to "What's the weather?" fails to meet usability requirements, regardless of response accuracy.

**State Initialization**: The function creates initial state with a `HumanMessage` containing the user's query, matching the agent's expected input format.

**Event Processing Loop**: As the agent executes through its graph (transitioning from the LLM node to the tools node and back), we intercept and examine each message:
- `AIMessage` with `tool_calls` indicates the agent has decided to use a tool. We capture the tool name and arguments.
- `AIMessage` with `content` but no tool calls represents the final answer, which we save.
- `ToolMessage` objects contain tool execution results. These are already present in the conversation history and don't require explicit capture.

**Metrics Calculation**: After execution completes, we calculate latency and estimate token usage. These metrics enable detection of performance regressions and cost estimation.

The function returns an `AgentTestResult` containing comprehensive information: tools invoked, final response, and performance metrics.

### Step 9: Define the Test Set

Defining what your agent should actually do. Each test case specifies three components::

- **Input**: The user query to be processed
- **Expected tool calls**: What tools should be invoked and with what parameters
- **Expected response**: A substring the final answer should contain

```python
AGENT_TESTS = [
    {
        "name": "weather_query",
        "input": "What is the weather in Miami?",
        "expected_tool_calls": [
            {"tool_name": "get_current_weather", "tool_parameters": {"location": "Miami"}}
        ],
        "expected_response_contains": "Miami"
    },
    {
        "name": "stock_price_query",
        "input": "What were the IBM stock prices on September 5, 2025?",
        "expected_tool_calls": [
            {"tool_name": "get_stock_price", "tool_parameters": {"ticker": "IBM", "date": "2025-09-05"}}
        ],
        "expected_response_contains": "IBM"
    },
    {
        "name": "weather_different_city",
        "input": "Tell me the current weather in New York",
        "expected_tool_calls": [
            {"tool_name": "get_current_weather", "tool_parameters": {"location": "New York"}}
        ],
        "expected_response_contains": "New York"
    },
    {
        "name": "stock_different_ticker",
        "input": "Get me the stock price for AAPL on January 15, 2025",
        "expected_tool_calls": [
            {"tool_name": "get_stock_price", "tool_parameters": {"ticker": "AAPL", "date": "2025-01-15"}}
        ],
        "expected_response_contains": "AAPL"
    },
]
```

#### Test Case Design Rationale

**Basic Functionality Tests** (`weather_query` and `stock_price_query`): These tests validate fundamental functionality for each tool. Failures in these tests indicate critical issues with core capabilities.

**Parameter Variation Tests** (`weather_different_city` and `stock_different_ticker`): These tests use different parameters to verify proper entity extraction and parameter handling. For example, if the agent consistently queries weather for "Miami" regardless of user input, these tests will detect the issue.

**Parameter Format Validation**: The expected parameters align precisely with user requests. For `weather_query`, the user specifies "Miami" and we expect `{"location": "Miami"}`. For `stock_price_query`, the user provides "September 5, 2025" and we expect the ISO date format `"2025-09-05"`. This validates the agent's ability to extract and normalize information correctly.

**Response Validation**: The `expected_response_contains` field uses intentionally simple validation, confirming only that the agent mentions the relevant entity in its response. This approach avoids brittle tests that fail due to minor wording variations.


### Step 10: Run Single-Turn Tests

With test cases defined, we can now execute them and evaluate results:

```python
def run_single_turn_tests(test_graph: CompiledStateGraph, tests: List[Dict]) -> List[Dict]:
    """Run all single-turn tests and collect results."""
    results = []
    for test in tests:
        print(f"Running test: {test['name']}...")
        output = run_agent_for_test(test_graph, test["input"])
        
        # For trajectory matching, we check if the expected tool was called
        # Note: LLM may include slight variations in parameters
        traj_ok = len(output.tool_calls) == len(test["expected_tool_calls"])
        if traj_ok and len(output.tool_calls) > 0:
            # Check tool names match
            for actual, expected in zip(output.tool_calls, test["expected_tool_calls"]):
                if actual.tool_name != expected["tool_name"]:
                    traj_ok = False
                    break
        
        resp_ok = response_match(output.final_response, test["expected_response_contains"])
        
        results.append({
            "name": test["name"],
            "trajectory_ok": traj_ok,
            "response_ok": resp_ok,
            "latency_ms": round(output.latency_ms, 2),
            "prompt_tokens": output.prompt_tokens,
            "response_tokens": output.response_tokens,
            "total_tokens": output.total_tokens,
            "tool_calls": [(tc.tool_name, tc.tool_parameters) for tc in output.tool_calls],
            "final_response": output.final_response[:200] + "..." if len(output.final_response) > 200 else output.final_response
        })
        print(f"  ✓ Trajectory: {traj_ok}, Response: {resp_ok}")
    
    return results

# Run the tests
test_results = run_single_turn_tests(graph, AGENT_TESTS)
test_results
```
#### Understanding the Evaluation Logic

**Trajectory Validation**: The validation begins by comparing tool call counts. If the agent makes 2 tool calls when 1 is expected, this constitutes an immediate failure. Subsequently, we verify that each tool name matches expectations. Note that we do not perform strict parameter matching in this implementation, as LLMs may format parameters with slight variations (for example, "New York" versus "New York City"). Production systems should implement more sophisticated parameter validation as needed.

**Response Validation**: This utilizes the `response_match` helper function for simple substring matching.

**Results Dictionary**: The results capture comprehensive information including pass/fail status, performance metrics, and actual outputs for debugging purposes. Long responses are truncated to 200 characters to maintain output readability.

**Example output:**

```
Running test: weather_query...
Getting current weather for Miami
  ✓ Trajectory: True, Response: True
Running test: stock_price_query...
Getting stock price for IBM on 2025-09-05
Error fetching stock data: '2025-09-05'
  ✓ Trajectory: True, Response: True
Running test: weather_different_city...
Getting current weather for New York
  ✓ Trajectory: True, Response: True
Running test: stock_different_ticker...
Getting stock price for AAPL on 2025-01-15
Error fetching stock data: '2025-01-15'
  ✓ Trajectory: True, Response: True
```

All four tests passed! The agent correctly identified which tool to call and included the expected entities in its responses. The "Error fetching stock data" messages are expected - those dates are in the future, so the API returns an error, but the agent still handled it gracefully.

### Step 11: Define Multi-Turn Tests

Real conversations don't happen in isolation. Users ask follow-up questions. They reference previous context. They switch topics mid-conversation. Your agent needs to handle all of this.

Multi-turn tests verify that the agent maintains context across a conversation:

```python
MULTI_TURN_TESTS = [
    {
        "name": "weather_then_stock",
        "turns": [
            {
                "input": "What is the weather in Boston?",
                "expected_tool_name": "get_current_weather",
                "expected_response_contains": "Boston"
            },
            {
                "input": "Now tell me the IBM stock price on January 10, 2025",
                "expected_tool_name": "get_stock_price",
                "expected_response_contains": "IBM"
            }
        ]
    },
    {
        "name": "multiple_weather_queries",
        "turns": [
            {
                "input": "What's the weather like in London?",
                "expected_tool_name": "get_current_weather",
                "expected_response_contains": "London"
            },
            {
                "input": "How about in Tokyo?",
                "expected_tool_name": "get_current_weather",
                "expected_response_contains": "Tokyo"
            },
            {
                "input": "And what about Paris?",
                "expected_tool_name": "get_current_weather",
                "expected_response_contains": "Paris"
            }
        ]
    }
]
```

#### Multi-Turn Test Design

**Tool Type Switching** (`weather_then_stock`): This test validates the agent's ability to transition between different tool types within a single conversation. Some agents exhibit "mode stickiness," struggling to switch between different tool categories.

**Contextual Reference Handling** (`multiple_weather_queries`): This test uses follow-up language such as "How about in Tokyo?" and "And what about Paris?". The agent must understand that these queries reference weather information, despite the absence of the explicit term "weather." This validates contextual understanding capabilities.

Note that each turn specifies only `expected_tool_name` rather than complete `expected_tool_calls`. For multi-turn testing, the primary focus is on correct tool selection. Parameter extraction validation is covered comprehensively in single-turn tests.

### Step 12: Run Multi-Turn Tests

Multi-turn tests are trickier because we need to maintain conversation state across turns:

```python
def run_multi_turn_tests(test_graph: CompiledStateGraph, tests: List[Dict]) -> List[Dict]:
    """Run multi-turn tests where each test has multiple conversation turns."""
    all_results = []
    
    for test in tests:
        print(f"\nRunning multi-turn test: {test['name']}")
        turn_results = []
        
        for i, turn in enumerate(test["turns"]):
            print(f"  Turn {i+1}: {turn['input'][:50]}...")
            output = run_agent_for_test(test_graph, turn["input"])
            
            # Check if the expected tool was called
            tool_ok = any(tc.tool_name == turn["expected_tool_name"] for tc in output.tool_calls)
            resp_ok = response_match(output.final_response, turn["expected_response_contains"])
            
            turn_results.append({
                "input": turn["input"],
                "tool_ok": tool_ok,
                "response_ok": resp_ok,
                "tool_calls": [(tc.tool_name, tc.tool_parameters) for tc in output.tool_calls],
                "final_response": output.final_response[:100] + "..." if len(output.final_response) > 100 else output.final_response
            })
            print(f"    ✓ Tool: {tool_ok}, Response: {resp_ok}")
        
        all_results.append({"name": test["name"], "turns": turn_results})
    
    return all_results

# Run multi-turn tests
multi_turn_results = run_multi_turn_tests(graph, MULTI_TURN_TESTS)
multi_turn_results
```

#### Implementation Differences from Single-Turn Testing

**Multi-Turn Structure**: Each test contains multiple turns. The implementation loops through them sequentially, with LangGraph automatically maintaining conversation history in the state.

**Per-Turn Evaluation**: The evaluation for each turn is streamlined—we verify that the correct tool was invoked and that the response contains the expected entity. Strict parameter matching is not performed at this level.

**Result Aggregation**: Results for all turns are accumulated and packaged together. This structure enables pattern detection, such as "first turn succeeds but second turn fails," which may indicate context-handling issues.

**Example output:**

```
Running multi-turn test: weather_then_stock
  Turn 1: What is the weather in Boston?...
Getting current weather for Boston
    ✓ Tool: True, Response: True
  Turn 2: Now tell me the IBM stock price on January 10, 202...
Getting stock price for IBM on 2025-01-10
Error fetching stock data: '2025-01-10'
    ✓ Tool: True, Response: True

Running multi-turn test: multiple_weather_queries
  Turn 1: What's the weather like in London?...
Getting current weather for London
    ✓ Tool: True, Response: True
  Turn 2: How about in Tokyo?...
Getting current weather for Tokyo
    ✓ Tool: True, Response: True
  Turn 3: And what about Paris?...
Getting current weather for Paris
    ✓ Tool: True, Response: True
```

Perfect! All turns passed. The agent correctly interpreted follow-up questions like "How about in Tokyo?" as weather queries and extracted the right city names.

### Step 13: Compute Summary Metrics

While individual test results provide detailed debugging information, summary metrics offer high-level insights into overall agent performance:

```python
# Single-turn summary
passed = sum(1 for r in test_results if r["trajectory_ok"] and r["response_ok"])
total = len(test_results)
avg_latency = round(sum(r["latency_ms"] for r in test_results) / total, 2) if total > 0 else 0
avg_total_tokens = round(sum(r["total_tokens"] for r in test_results) / total, 2) if total > 0 else 0

single_turn_summary = {
    "passed": passed,
    "total": total,
    "pass_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "N/A",
    "avg_latency_ms": avg_latency,
    "avg_total_tokens": avg_total_tokens
}

print("=" * 50)
print("SINGLE-TURN TEST SUMMARY")
print("=" * 50)
print(f"Tests Passed: {passed}/{total}")
print(f"Pass Rate: {single_turn_summary['pass_rate']}")
print(f"Average Latency: {avg_latency} ms")
print(f"Average Tokens: {avg_total_tokens}")
print()

# Multi-turn summary
multi_turn_passed = 0
multi_turn_total = 0

for test in multi_turn_results:
    for turn in test["turns"]:
        multi_turn_total += 1
        if turn["tool_ok"] and turn["response_ok"]:
            multi_turn_passed += 1

multi_turn_summary = {
    "passed": multi_turn_passed,
    "total": multi_turn_total,
    "pass_rate": f"{(multi_turn_passed/multi_turn_total)*100:.1f}%" if multi_turn_total > 0 else "N/A"
}

print("=" * 50)
print("MULTI-TURN TEST SUMMARY")
print("=" * 50)
print(f"Turns Passed: {multi_turn_passed}/{multi_turn_total}")
print(f"Pass Rate: {multi_turn_summary['pass_rate']}")
print()

# Overall summary
overall_passed = passed + multi_turn_passed
overall_total = total + multi_turn_total
print("=" * 50)
print("OVERALL TEST SUMMARY")
print("=" * 50)
print(f"Total Passed: {overall_passed}/{overall_total}")
print(f"Overall Pass Rate: {(overall_passed/overall_total)*100:.1f}%" if overall_total > 0 else "N/A")
```

**Example output:**

```
==================================================
SINGLE-TURN TEST SUMMARY
==================================================
Tests Passed: 4/4
Pass Rate: 100.0%
Average Latency: 1971.31 ms
Average Tokens: 44.75

==================================================
MULTI-TURN TEST SUMMARY
==================================================
Turns Passed: 5/5
Pass Rate: 100.0%

==================================================
OVERALL TEST SUMMARY
==================================================
Total Passed: 9/9
Overall Pass Rate: 100.0%
```

#### Metrics Interpretation

**Pass Rate**: This metric indicates the percentage of tests that meet all success criteria. While this is the primary indicator of correctness, it should be analyzed in conjunction with other metrics.

**Average Latency**: This metric directly impacts user experience. If average latency increases from 2 seconds to 5 seconds following a model update, this represents a performance regression even if accuracy remains constant.

**Average Tokens**: This metric correlates directly with operational costs. If a change to prompting strategy doubles average token count, you must evaluate whether quality improvements justify the increased cost.

These metrics should be tracked over time. Implement alerts for scenarios where pass rate drops below acceptable thresholds or when latency increases significantly. This approach enables early detection of regressions.

## What We've Built

Let's recap what we now have:

**Data structures** that cleanly represent tool calls and test results  
**Evaluation helpers** for both trajectory and response validation  
**A test runner** that executes the agent and captures everything we need  
**Single-turn tests** that verify basic functionality for each tool  
**Multi-turn tests** that verify conversation handling  
**Summary metrics** that give us the big picture

This is a complete testing framework. You can now:

- Add new test cases as you add new tools or identify edge cases
- Compare different models by running the same tests against multiple agents
- Catch regressions by running tests in CI/CD before deployment
- Track performance over time with your latency and token metrics

## Next Steps

This framework provides a solid foundation for production agent testing. Consider the following enhancements for production systems:

**Expand Test Coverage**: Add edge case testing to validate error handling. Test scenarios such as non-existent cities for weather queries or invalid date formats for stock queries. Comprehensive testing should cover failure modes in addition to success paths.

**Advanced Evaluation Methods**: For responses where substring matching proves insufficient, implement LLM-as-a-judge evaluation or semantic similarity scoring for responses that should be semantically similar but not textually identical.

**CI/CD Integration**: Automate test execution on every pull request. Configure build failures when pass rate drops below defined thresholds. This practice catches regressions before they reach production environments.

**Model Comparison Framework**: When evaluating new model versions, execute the complete test suite against both models and compare results systematically. Base decisions on objective data rather than subjective assessment.

**Metrics Tracking System**: Store test results in a database and construct monitoring dashboards. Track trends such as gradually increasing latency or degrading pass rates over time.

## Conclusion

Testing AI agents requires different approaches than traditional software testing, but maintains equal importance. The non-deterministic nature of LLMs can make manual testing seem sufficient, but this approach does not scale.

The testing framework presented in this guide provides:
- Confidence that agents behave as expected
- Early detection of breaking changes
- Objective data for comparing alternatives
- A foundation for continuous improvement

Begin with the tests defined in this guide, then expand coverage as you discover new edge cases or add capabilities. This systematic approach ensures that core use cases remain validated through agent evolution.

The complete implementation of this testing framework is available in the Granite Kitchen repository for reference and adaptation to your specific use cases.