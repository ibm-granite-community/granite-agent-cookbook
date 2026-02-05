# Agent Test Driven Development: Evaluating AI Agents

Building AI agents that call functions and interact with external tools opens up powerful possibilities for automation and intelligent assistance. However, as these agents move from prototype to production, a critical question emerges: how do you know your agent is working correctly?

This guide introduces the concepts and approaches for evaluating function-calling agents. Rather than diving straight into code, we'll explore the fundamentals of agent evaluation - what to test, how to define success criteria, and which validation approaches work best for different scenarios. By understanding these concepts, you'll be equipped to design robust evaluation strategies for your own agents.

If you haven't already built a function-calling agent, check out our previous guide on [building agents with LangGraph and Granite](../Function_Calling/Function_Calling_Agent.ipynb) to get started.

## Why Test Your Agent?

In traditional software development, Test-Driven Development (TDD) has become a cornerstone practice. You write tests that define expected behavior, then build code that satisfies those tests. This same principle applies to AI agents, but with added complexity due to their non-deterministic nature.

Bringing confidence that your agent does what is expected is not optional, it's a necessity for productionizing agents. Here's why systematic evaluation matters:

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

- **Code Validation**: Code generation agents require validation that generated code is syntactically correct, executes successfully, and produces expected outputs. A code snippet that looks correct but contains subtle bugs is worse than no code at all.

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

```
Input: "What is the weather in Miami?"
Expected Tool: get_current_weather
Expected Parameters: {"location": "Miami"}
Expected Response Content: Should mention Miami and weather conditions
```

The key is specificity. Vague expectations like "should work correctly" don't help you identify failures. Instead, define concrete, measurable criteria.

### Types of Test Cases

Different test patterns validate different capabilities. A comprehensive evaluation suite should include multiple types.

#### Single-Step Cases

Single-step test cases verify that the agent can handle straightforward requests that require exactly one tool call.

**Basic Functionality**: Does the agent invoke the correct tool for simple, unambiguous requests?
```
Input: "Get me the stock price for IBM"
Expected: Calls get_stock_price with ticker="IBM"
```

**Parameter Extraction**: Can the agent extract multiple parameters correctly?
```
Input: "What were Apple stock prices on January 15, 2025?"
Expected: Calls get_stock_price with ticker="AAPL", date="2025-01-15"
```

**Parameter Normalization**: Does the agent normalize inputs to expected formats?
```
Input: "Weather in NYC"
Expected: Calls get_current_weather with location="New York City"
(Tests that the agent expands abbreviations appropriately)
```

**Ambiguity Handling**: How does the agent behave when inputs are unclear?
```
Input: "What about Apple?"
Expected: Might ask for clarification or make reasonable assumptions
```

#### Multi-Step Cases

Multi-step test cases require the agent to chain multiple tool calls to accomplish a goal.

**Sequential Operations**: The agent must perform operations in order, where each step depends on previous results.
```
Input: "Get the weather in Boston, then find the stock price for Tesla"
Expected: 
  Step 1: Calls get_current_weather with location="Boston"
  Step 2: Calls get_stock_price with ticker="TSLA"
```

**Conditional Logic**: The agent must make decisions based on intermediate results.
```
Input: "If the weather in Seattle is rainy, get me the stock price for umbrella companies"
Expected: Calls get_current_weather first, then conditionally calls get_stock_price
```

#### Multi-Turn Cases

Multi-turn test cases verify that the agent maintains context across a conversation.

**Follow-up Questions**: Can the agent understand references to previous context?
```
Turn 1: "What's the weather in London?"
Expected: Calls get_current_weather with location="London"

Turn 2: "How about in Tokyo?"
Expected: Calls get_current_weather with location="Tokyo"
(Tests that agent understands "How about" refers to weather)
```

**Topic Switching**: Can the agent handle conversation shifts?
```
Turn 1: "What's the weather in Boston?"
Expected: Calls get_current_weather with location="Boston"

Turn 2: "Now tell me the IBM stock price"
Expected: Calls get_stock_price with ticker="IBM"
(Tests that agent doesn't get stuck in "weather mode")
```

**Anaphora Resolution**: Can the agent resolve pronouns and references?
```
Turn 1: "Get me Apple's stock price"
Expected: Calls get_stock_price with ticker="AAPL"

Turn 2: "What was it yesterday?"
Expected: Calls get_stock_price with ticker="AAPL", date=<yesterday>
(Tests that agent understands "it" refers to Apple stock)
```

### Evaluation Metrics

Metrics provide quantitative measures of agent performance. Different metrics capture different aspects of quality.

1. **Tool Selection Accuracy**: What percentage of test cases result in the correct tool being called? This is often the most critical metric for tool-calling agents.

2. **Parameter Extraction Accuracy**: When the right tool is called, are the parameters correct? Partial credit might be awarded calling the weather API with the right city but wrong units is better than calling the wrong API entirely.

3. **Response Quality**: Does the final response appropriately communicate the tool results to the user? This is harder to measure automatically but critical for user experience.

4. **Latency**: How long does the agent take to respond? Even correct answers that arrive too slowly fail to meet production requirements.

5. **Token Efficiency**: How many tokens does the agent consume per request? This directly impacts operational costs.

6. **Success Rate Over Conversation Length**: Do multi-turn conversations degrade in quality as they get longer? Some agents perform well initially but lose context after several turns.

7. **Error Recovery**: When tool calls fail, does the agent handle errors gracefully? A good agent should inform the user and potentially suggest alternatives.

### Validation Mechanisms

Once you've defined test cases and metrics, you need mechanisms to validate whether the agent passed or failed.

#### Programmatic Matching for Tool Calls

Tool call validation can often be done programmatically since the structure is well-defined.

**Exact Matching**: For high-stakes tools (those that write data, make purchases, etc.), exact matching is appropriate. The tool name must match exactly, and all required parameters must be present with correct values.

**Fuzzy Matching**: For lower-stakes tools, some flexibility might be acceptable. "New York" vs "New York City" might both be valid for a weather query. Date formats "2025-01-15" vs "January 15, 2025" represent the same information.

**Schema Validation**: Beyond checking values, ensure that parameters conform to expected types and constraints. A stock ticker should be a string of 1-5 uppercase letters, not a number or lowercase text.

The choice between strict and lenient matching depends on your use case. An agent booking flights needs exact matching. An agent fetching weather for general conversation might allow more flexibility.

#### LLM-as-a-Judge for Response Evaluation

While tool calls have clear structure, final responses are natural language and harder to evaluate automatically.

**Substring Matching**: The simplest approach checks whether expected keywords appear in the response. If testing a weather query for Miami, verify "Miami" appears in the response. This is fast and deterministic but catches only obvious failures.

**Semantic Similarity**: Compare the agent's response to reference responses using embedding models. Responses that are semantically similar to good examples score higher. This allows for natural variation in phrasing while ensuring key information is present.

**LLM-as-a-Judge**: Use another LLM to evaluate response quality. Provide the judge with the user query, tool results, and agent response, then ask it to score factors like accuracy, completeness, and helpfulness. This is the most flexible approach but introduces its own non-determinism and cost.

**Human Evaluation**: For the most critical use cases or when developing your evaluation strategy, human review remains the gold standard. Humans can catch subtle issues that automated approaches miss, though this doesn't scale well.

A pragmatic approach combines multiple validation mechanisms. Use programmatic matching for tool calls, substring matching for basic response validation, and periodic human review to catch edge cases and inform improvements to automated evaluation.

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

**Integrate with Development Workflow**: Run tests automatically on pull requests to catch regressions before they reach production. Fail builds when critical tests fail or when metrics degrade significantly.

**Compare Models and Approaches**: Use your evaluation suite to make objective decisions when comparing different models, prompting strategies, or architectural patterns. Let data guide your choices.

**Monitor Production Performance**: Beyond development testing, track the same metrics in production. Real user interactions often reveal issues that test suites miss.

## Conclusion

Evaluation is not a one-time activity but an ongoing practice. As your agent evolves, your evaluation strategy should evolve with it. New features require new tests. Discovered bugs should translate into test cases that prevent regressions.

The investment in robust evaluation pays dividends throughout your agent's lifecycle. It enables confident iteration, catches issues early, and provides the data needed to make informed decisions about model selection, architecture, and feature development.

Most importantly, systematic evaluation is what separates prototype demonstrations from production-ready systems. If you can't measure whether your agent works correctly, you can't confidently deploy it to real users.

Start with the fundamentals covered in this guide, implement testing incrementally, and continuously refine your approach based on what you learn. The path to production-ready agents begins with the confidence that comes from thorough evaluation.