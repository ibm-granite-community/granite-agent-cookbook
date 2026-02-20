# Agent Test Driven Development: Evaluating AI Agents

Building AI agents that call functions and interact with external tools opens up powerful possibilities for automation and intelligent assistance. However, as these agents move from prototype to production, a critical question emerges: how do you know your agent is working correctly?

This guide introduces the concepts and approaches for evaluating function-calling agents. Rather than diving straight into code, we'll explore the fundamentals of agent evaluation - what to test, how to define success criteria, and which validation approaches work best for different scenarios. By understanding these concepts, you'll be equipped to design robust evaluation strategies for your own agents.

If you haven't already built a function-calling agent, check out our previous guide on [building agents with LangGraph and Granite](../Function_Calling/Function_Calling_Agent.ipynb) to get started.

## Why Test Your Agent?

In traditional software development, Test-Driven Development (TDD) has become a cornerstone practice. You write tests that define expected behavior, then build code that satisfies those tests. This same principle applies to AI agents, but with added complexity due to their non-deterministic nature.

Bringing confidence that your agent does what is expected is not optional, it's a necessity for productizing agents. Here's why systematic evaluation matters:

**Regression Detection** - Know immediately when prompt changes, model updates, or tool modifications break existing functionality
**Objective Comparison** - Make data-driven decisions when comparing models, prompting strategies, or architectural approaches
**Production Readiness** - Deploy with confidence knowing core use cases work reliably
**Debugging Efficiency** - Reproduce and diagnose issues systematically rather than relying on manual testing
**Living Documentation** - Test cases serve as executable specifications of how your agent should behave

The stakes are even higher than traditional software. An agent that calls the wrong API, extracts incorrect parameters, or misinterprets user intent can have real-world consequences - financial transactions, data modifications, or incorrect information delivery.

## What Can You Test in an Agent?

Before diving into evaluation strategies, it's important to understand what aspects of agent behavior can be tested. Different agent architectures require different evaluation approaches.

- **Tool Calling**: For function-calling agents, the most critical aspect is whether the agent selects the correct tool and extracts appropriate parameters from user queries. A weather agent that calls a stock price API instead of a weather API has fundamentally failed, regardless of how well-written its final response is.

- **Final Responses**: After tools execute and return results, the agent synthesizes this information into a natural language response. Even if the correct tool was called, the response might be unclear, incomplete, or fail to address the user's actual question.

- **Retrieval Quality**: For Retrieval-Augmented Generation (RAG) agents, evaluation must consider whether the agent retrieves relevant documents and effectively uses them to answer questions. Retrieving documents is only useful if they're actually relevant to the query.

- **Code Validation**: Code generation agents require validation that generated code is syntactically correct, executes successfully, and produces expected outputs. A code snippet that looks correct but contains subtle bugs can make code maintenance difficult.

- **Multi-Step Reasoning**: Complex agents often need to chain multiple operations together. Evaluation must verify not just individual steps, but that the overall sequence achieves the intended goal.

In this guide, we'll focus primarily on evaluating tool-calling agents, as they represent a common and foundational pattern. The principles discussed here extend naturally to other agent types.

## Evaluating Tool-Calling Agents

Tool-calling agents make decisions about which functions to invoke and what parameters to pass. Effective evaluation requires testing both dimensions: the agent's decision-making process and its final outputs.

### Defining Test Cases

A well-defined test case specifies three components:

**User Input**: The query or prompt that triggers the agent
**Expected Behavior**: What the agent should do in response
**Success Criteria**: How to determine if the agent performed correctly

For a weather agent with `get_current_weather` and `get_stock_price` tools, a basic test case might look like:

```text
Input: "What is the weather in Miami?"
Expected Tool: get_current_weather
Expected Parameters: {"location": "Miami"}
Expected Response Content: Should mention Miami and weather conditions
```

### Types of Test Cases

Different test patterns validate different capabilities. A comprehensive evaluation suite should include multiple types.

#### Single-Step Cases

Single-step test cases verify that the agent can handle straightforward requests that require exactly one tool call.

- **Basic functionality**: Does the agent invoke the correct tool for simple, unambiguous requests?

  ```text
  Input: "Get me the stock price for IBM"
  Expected Tool: get_stock_price
  Expected Parameters: {"ticker": "IBM"}
  Expected Response Content: Should mention IBM and the current stock price
  ```

- **Parameter extraction**: Can the agent extract multiple parameters correctly?

  ```text
  Input: "What were Apple stock prices on January 15, 2025?"
  Expected Tool: get_stock_price
  Expected Parameters: {"ticker": "AAPL", "date": "2025-01-15"}
  Expected Response Content: Should mention Apple/AAPL and the stock price for 2025-01-15
  ```

- **Parameter normalization**: Does the agent normalize inputs to expected formats?

  ```text
  Input: "Weather in NYC"
  Expected Tool: get_current_weather
  Expected Parameters: {"location": "New York City"}
  Expected Response Content: Should mention New York City and weather conditions
  ```

- **Ambiguity handling**: How does the agent behave when inputs are unclear?

  ```text
  Input: "What about Apple?"
  Expected Behavior: Ask for clarification (stock price vs. weather vs. something else), or make a clearly stated assumption
  Success Criteria: Agent does not call an unrelated tool with fabricated parameters
  ```

#### Multi-Step Cases

Multi-step test cases require the agent to chain multiple tool calls to accomplish a goal.

- **Sequential operations**: The agent must perform operations in order, where each step depends on previous results.

  ```text
  Input: "Get the weather in Boston, then find the stock price for Tesla"
  Step 1 Expected Tool: get_current_weather
  Step 1 Expected Parameters: {"location": "Boston"}
  Step 2 Expected Tool: get_stock_price
  Step 2 Expected Parameters: {"ticker": "TSLA"}
  Success Criteria: Both tool calls occur in order with correct parameters
  ```

- **Conditional logic**: The agent must make decisions based on intermediate results.

  ```text
  Input: "If the weather in Seattle is rainy, get me the stock price for umbrella companies"
  Step 1 Expected Tool: get_current_weather
  Step 1 Expected Parameters: {"location": "Seattle"}
  Step 2 Expected Behavior: If Step 1 indicates rain, then either (a) ask a clarification question to identify a company/ticker, or (b) call get_stock_price with a clearly justified ticker
  Success Criteria: Agent does not skip Step 1; Step 2 is consistent with the weather result and avoids made-up parameters
  ```

#### Multi-Turn Cases

Multi-turn test cases verify that the agent maintains context across a conversation.

- **Follow-up questions**: Can the agent understand references to previous context?

  ```text
  Turn 1 Input: "What's the weather in London?"
  Turn 1 Expected Tool: get_current_weather
  Turn 1 Expected Parameters: {"location": "London"}
  Turn 1 Expected Response Content: Should mention London and weather conditions

  Turn 2 Input: "How about in Tokyo?"
  Turn 2 Expected Tool: get_current_weather
  Turn 2 Expected Parameters: {"location": "Tokyo"}
  Turn 2 Expected Response Content: Should mention Tokyo and weather conditions
  Success Criteria: Turn 2 correctly inherits the intent (weather) from Turn 1
  ```

- **Topic switching**: Can the agent handle conversation shifts?

  ```text
  Turn 1 Input: "What's the weather in Boston?"
  Turn 1 Expected Tool: get_current_weather
  Turn 1 Expected Parameters: {"location": "Boston"}
  Turn 1 Expected Response Content: Should mention Boston and weather conditions

  Turn 2 Input: "Now tell me the IBM stock price"
  Turn 2 Expected Tool: get_stock_price
  Turn 2 Expected Parameters: {"ticker": "IBM"}
  Turn 2 Expected Response Content: Should mention IBM and stock price information
  Success Criteria: Turn 2 switches tools correctly (agent doesn't stay in weather intent)
  ```

- **Anaphora resolution**: Can the agent resolve pronouns and references?

  ```text
  Turn 1 Input: "Get me Apple's stock price"
  Turn 1 Expected Tool: get_stock_price
  Turn 1 Expected Parameters: {"ticker": "AAPL"}
  Turn 1 Expected Response Content: Should mention Apple/AAPL and stock price information

  Turn 2 Input: "What was it yesterday?"
  Turn 2 Expected Tool: get_stock_price
  Turn 2 Expected Parameters: {"ticker": "AAPL", "date": "<yesterday>"}
  Turn 2 Expected Response Content: Should provide Apple/AAPL stock price for yesterday
  Success Criteria: "it" correctly resolves to Apple stock (not weather or another entity)
  ```

### Evaluation Metrics and Validation

Metrics provide quantitative measures of agent performance. Different metrics capture different aspects of quality.

### Metrics

- **Tool selection accuracy**
  - What percentage of test cases result in the correct tool being called?
  - Often the most critical metric for tool-calling agents.

- **Parameter extraction accuracy**
  - When the right tool is called, are the parameters correct?
  - Partial credit can be meaningful (e.g., right city but wrong units is better than calling the wrong tool).

  *Note: For an example of representing tool calls as an AST for standardized evaluation, see the "Metrics" section here: <https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics>*

- **Response quality**
  - Does the final response appropriately communicate the tool results to the user?
  - Harder to measure automatically, but critical for user experience.

- **Latency**
  - How long does the agent take to respond?
  - Even correct answers that arrive too slowly may fail production requirements.

- **Token efficiency**
  - How many tokens does the agent consume per request?
  - Directly impacts operational costs.

- **Success rate over conversation length**
  - Do multi-turn conversations degrade in quality as they get longer?
  - Some agents perform well initially but lose context after several turns.

- **Error recovery**
  - When tool calls fail, does the agent handle errors gracefully?
  - A good agent should inform the user and potentially suggest alternatives.

### Validation mechanisms

- **Programmatic matching for tool calls**
  - Tool call validation is often programmatic because the structure is well-defined.
  - **Exact matching**: Use for high-stakes tools (writes data, makes purchases). Tool name and required parameters must match exactly.
  - **Fuzzy matching**: Use for lower-stakes tools where some flexibility is acceptable (e.g., “New York” vs “New York City”).
  - **Schema validation**: Ensure parameters conform to expected types and constraints (e.g., ticker is 1–5 uppercase letters).
  - **LLM-as-a-judge for tool calls**: When parameters are dynamic or context-dependent, deterministic matching becomes brittle. An LLM judge can evaluate whether the selected tool and extracted parameters are semantically correct given the user's intent. This is especially useful for tools that accept free-text arguments (e.g., a `search_documents` tool with a `query` parameter), date/time expressions that can be phrased many ways, or parameters derived from multi-turn context where the "correct" value isn't a single fixed string. The judge prompt should include the user query, the expected tool behavior, and the actual tool call, then ask whether the call is appropriate. This adds cost per evaluation but scales better than enumerating every valid parameter variant.
  - Choose strict vs. lenient matching based on your use case. Prefer deterministic checks for well-structured parameters and reserve LLM-as-a-judge for parameters where valid values are open-ended or hard to enumerate.

- **Response evaluation (LLM-as-a-judge and alternatives)**
  - Tool calls are structured; final responses are natural language and harder to evaluate deterministically.
  - **Substring matching**: Check expected keywords appear (fast, deterministic, limited coverage).
  - **Semantic similarity**: Compare to reference responses using embeddings (allows phrasing variation).
  - **LLM-as-a-judge**: Score accuracy/completeness/helpfulness given user query, tool results, and agent response (flexible, but adds cost and non-determinism).
  - **Human evaluation**: Best for critical use cases or calibration, but doesn’t scale.
  - A pragmatic approach combines multiple mechanisms: programmatic matching for tool calls, lightweight response checks, and periodic human review.

## Performance Metrics and Tracking

Beyond pass/fail status for individual test cases, tracking metrics over time reveals trends and regressions.

**Pass Rate Trends**: If your pass rate drops from 95% to 85% after a model update, that's a signal to investigate, even if 85% seems acceptable in isolation.

**Latency Distribution**: Average latency matters, but so does variance. An agent with 2-second average latency but occasional 30-second outliers creates a poor user experience.

**Cost Trends**: Token usage multiplied by pricing determines operational costs. Track this metric to ensure costs remain sustainable as usage scales.

**Error Type Distribution**: Categorize failures by type (wrong tool, wrong parameters, poor response quality, etc.). This helps prioritize improvements. If 80% of failures are parameter extraction errors, that's where to focus optimization.

Establish baselines for these metrics with your initial agent version, then monitor for significant deviations. Automated alerts when metrics cross thresholds enable rapid response to issues.

## Evaluation Approaches Not Covered

This guide focuses on tool-calling agents, but other agent types require specialized evaluation approaches:

**Retrieval Evaluation for RAG Agents**: RAG systems need metrics like retrieval precision (what percentage of retrieved documents are relevant) and recall (what percentage of relevant documents were retrieved). The [RAGAS framework](https://docs.ragas.io/) provides comprehensive evaluation metrics for RAG applications.

**Code Execution Testing**: Code generation agents require running generated code in sandboxed environments and verifying outputs match expected results, along with static analysis for security vulnerabilities.

**Long-Context Evaluation**: Agents working with very long conversations or documents need specialized tests for maintaining coherence and accuracy as context length increases.

## Practical Considerations

As you design your evaluation strategy, keep these practical considerations in mind:

**Start Simple**: Begin with a small set of critical test cases using basic validation mechanisms. Expand coverage as you understand your agent's failure modes better.

**Version Control Your Tests**: Test cases are as important as your code. Track them in version control so you can correlate agent changes with evaluation results.

**Automate Where Possible**: Manual testing doesn't scale. Automate test execution so you can run your suite frequently—ideally on every code change.

**Balance Coverage and Efficiency**: Comprehensive coverage is ideal, but running 10,000 test cases on every change might be impractical. Identify a core set of smoke tests that run on every change, with more extensive testing on a schedule.

**Iterate on Your Evaluation Strategy**: Your first evaluation approach won't be perfect. As you discover new failure modes or edge cases, add tests that would have caught them. Your evaluation suite should evolve alongside your agent.

## Next Steps

Now that you understand the concepts behind agent evaluation, you're ready to implement testing for your own agents.

**Implement a Test Framework**: The companion code example in [Function_Calling_Agent_TDD.ipynb](./Function_Calling_Agent_TDD.ipynb) demonstrates these concepts in practice with a working implementation for tool-calling agents built with LangGraph and Granite.

**Expand Test Coverage**: Start with basic test cases covering core functionality, then add edge cases, error conditions, and multi-turn scenarios as you identify gaps.

**Integrate with Development Workflow**: Run tests automatically on pull requests to catch regressions before they reach production. Fail builds when critical tests fail or when metrics degrade significantly. For a deeper dive into CI/CD pipelines tailored for AI agents, stay tuned for our upcoming guide on CI/CD for Agents.

**Compare Models and Approaches**: Use your evaluation suite to make objective decisions when comparing different models, prompting strategies, or architectural patterns. Let data guide your choices.

**Monitor Production Performance**: Beyond development testing, track the same metrics in production. Real user interactions often reveal issues that test suites miss.

## Conclusion

Evaluation is not a one-time activity but an ongoing practice. As your agent evolves, your evaluation strategy should evolve with it. New features require new tests. Discovered bugs should translate into test cases that prevent regressions.

The investment in robust evaluation pays dividends throughout your agent's lifecycle. It enables confident iteration, catches issues early, and provides the data needed to make informed decisions about model selection, architecture, and feature development.

Most importantly, systematic evaluation is what separates prototype demonstrations from production-ready systems. If you can't measure whether your agent works correctly, you can't confidently deploy it to real users.

Start with the fundamentals covered in this guide, implement testing incrementally, and continuously refine your approach based on what you learn. The path to production-ready agents begins with the confidence that comes from thorough evaluation.
