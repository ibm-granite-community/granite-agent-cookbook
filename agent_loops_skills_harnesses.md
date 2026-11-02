# From “Hello, World” to Working Software: Agent Loops, Skills and Harnesses

Ever wondered how an LLM that answers questions becomes an agent that takes actions, writes code and works through complex tasks?

The difference is not just a better prompt. It is also the system around the model: how it executes actions, receives feedback, accesses reusable knowledge and operates within constraints. These are harness-level capabilities.

In this blog, we will explore three connected concepts:

- **Agent loops:** the feedback cycle.
- **Skills:** reusable instructions and resources.
- **Agent harnesses:** the machinery that brings everything together.

To make these ideas concrete, let’s follow one fictional task:

> Add a greeting feature to an existing Python application. It should greet users by name, handle blank names, support English and Spanish, follow the repository’s conventions, and pass its tests. Do not add dependencies or change unrelated files.

It sounds simple. But there is a difference between producing a greeting function and delivering a working feature.

*The code, repository details and execution traces below are illustrative, not results from a real application.*

## 1. A Plain LLM: A Plausible Answer

Suppose we send the task to an LLM in a single prompt, without repository access or tools. It might return:

```python
def greet(name):
    return f"Hello, {name}!"
```

That is a reasonable start. It greets someone by name.

But does it solve our problem?

```text
Input: "John"      → "Hello, John!"     Works
Input: "   "      → "Hello,    !"      Wrong
Spanish           → Not implemented   Missing
Repository fit    → Not checked       Unknown
Repository tests  → Not run           Unverified
```

The model has produced a plausible answer, but our request was not simply “write a greeting function”. It was “add a feature to an existing application”.

We need to understand where the function belongs, how localization works, what blank names should produce and how the repository verifies changes.

What if we stopped treating the first answer as the final answer? This is where Agent Loops come into picture.

## 2. Agent Loops: Act, Observe, Adjust

Like loops in programming languages, an agent loop has an entry point, a task to perform and an exit condition. It applies the same principles to an LLM-powered system.

An agent loop gives the model repeated opportunities to choose an action, receive its result and decide what to do next. A common pattern is to invoke the model, execute its requested tools, feed the observations back into context and repeat until a stop condition is met.

Conceptually:

```text
Assemble context
      ↓
Invoke the model
      ↓
Execute a permitted action
      ↓
Observe the result
      ↓
Update context and repeat
```

The important addition is **feedback from the environment**.

Generating another answer is not the same as checking whether the previous answer worked. Reading a file or running a test gives the agent something concrete to respond to.

For our greeting feature, an illustrative trace could look like this:

```text
User:
Add the greeting feature.

Agent → Read the existing implementation and tests.
Agent → Edit greeting.py.
Agent → Run the relevant tests.

Tool result:
Blank-name test failed.
Expected: "Hello, friend!"
Received: "Hello,    !"

Agent → Trim surrounding whitespace and handle empty names.
Agent → Run the tests again.

Tool result:
Spanish greeting test failed.
Expected: "¡Hola, Ana!"
Received: "Hello, Ana!"

Agent → Add Spanish support using the existing localization mechanism.
Agent → Run the tests again.

Tool result:
All specified tests pass.

Agent → Summarise the changes and verification.
```

Notice what changed: **an observation changed the next action**.

The first implementation was wrong. The useful part of the loop was not that the agent avoided mistakes, but that it could observe a mistake and act on it.

### A loop needs an exit

For our minimal design, let’s choose three controls:

1. **A maximum number of iterations**, so the agent cannot retry indefinitely.
2. **A permitted tool set**, limited to the actions this task needs.
3. **A completion check**, requiring evidence that the requested behavior works.

These are our design choices, not universal requirements.

In this example, completion would mean more than “the model says it is done”. We would check that the feature covers both languages and blank names, the relevant tests pass, and the diff contains no unrelated changes or new dependencies.

If the iteration limit is reached, the agent should report what remains unresolved rather than claim success.

### Feedback is not model retraining

Here, “learning from the attempt” means using new observations in the next decision. It does not mean the loop updates the model’s weights. The harness supplies additional information through context.

The agent can now correct mistakes. But why should it rediscover our greeting rules every time?

That brings us to skills.

## 3. Skills: Package the Procedure Once

Skills package specialized instructions and supporting resources in a reusable format. Under the Agent Skills specification, a skill is a directory containing at least a `SKILL.md` file, with optional scripts, references and assets.

Think of the distinction this way:

> The loop helps the agent respond to what just happened.  
> The skill gives it a procedure we have already written down.

For our fictional application, we could create:

```text
greeting-feature/
├── SKILL.md
├── references/
│   └── greeting-rules.md
├── scripts/
│   └── check_greeting.py
└── assets/
    └── greeting-cases.json
```

In our design:

- `SKILL.md` explains when to use the skill and the workflow to follow.
- `greeting-rules.md` records approved languages, fallback names and formatting.
- `check_greeting.py` provides an optional verification helper.
- `greeting-cases.json` stores reusable input/output examples.

The supporting folders are optional; `SKILL.md` is the required entry point.

### What goes in `SKILL.md`?

The specification requires YAML frontmatter followed by Markdown instructions.

Our example could look like this:

```markdown
---
name: greeting-feature
description: >-
  Implement or modify greeting features with localization,
  blank-name handling and repository-compatible tests.
  Use when adding user greetings or changing supported
  greeting languages.
---

# Greeting feature workflow

1. Read the existing implementation and repository instructions.
2. Read references/greeting-rules.md.
3. Reuse the application's existing localization mechanism.
4. Trim surrounding whitespace from names.
5. Use the approved fallback for blank names.
6. Add tests for supported languages and blank names.
7. Run the relevant tests and report their actual results.
8. Do not add dependencies or modify unrelated files.
```

### How does the agent discover it?

The `name` identifies the skill, while the `description` explains what it does and when it applies. The specification's progressive-disclosure pattern makes this metadata available before the full instructions are loaded.

Our description includes useful triggers: greetings, localization, blank names and supported languages.

“Helps with Python” would be much less useful. It does not tell the agent which Python tasks this procedure is intended for.

### What does it load?

When the skill is activated, the agent loads the full `SKILL.md`. Supporting references and resources are loaded as needed, rather than putting every available document into the initial context. This is **progressive disclosure**.

For our example:

```text
Discover:
"This skill handles greeting features."

Activate:
Read the greeting workflow.

Load supporting context:
Read the approved greeting rules.

Use resources when needed:
Consult test cases or run the verification helper.
```

This keeps the initial context focused while making more detail available when the task calls for it.

### What makes it portable?

The shared directory structure and file format make skills reusable across compatible implementations. However, executable scripts still depend on the receiving agent's supported tools, languages and environment.

A Python verification script is not useful if the receiving environment cannot execute Python. Portability of the package does not guarantee portability of every execution capability.

Read more about [Skills and Specifications](https://agentskills.io/home).

### Skills guide; controls enforce

There is an important distinction here.

In our skill, “do not modify unrelated files” is an instruction. If we want a hard restriction, our harness must implement it through execution controls.

Likewise, if our project requires tests before implementation, we should state that explicitly in the workflow. We should not expect the agent to infer an unstated development policy.

Skills are a place to package procedures. Harness-level hooks and middleware provide mechanisms for deterministic checks and execution behavior.

The loop closes the feedback cycle. The skill supplies reusable knowledge.

But something still has to run both.

## 4. Agent Harnesses: Bring the System Together

A useful framing, [presented by Vivek Trivedy](https://x.com/Vtrivedy10/status/2031408954517971368), is:

### Agent = Model + Harness

The model is one part of the agent. The harness is the surrounding code, configuration and execution logic, including tools, state, orchestration and controls.

This also clears up a common confusion:

**The harness is not the model plus everything around it. The agent is.**

The harness is the “everything around it” part.

### We already had a minimal harness

Remember the loop in Section 2?

Something had to call the model, execute file operations, run tests and return their output. That machinery was already a minimal harness. Tool execution and feedback loops do not happen inside a raw model by themselves.

So this section does not introduce orchestration for the first time. It expands our minimal harness into a richer system.

### Designing a harness for our greeting feature

Here is a proposed configuration:

| Component | Role in our example |
| --- | --- |
| Base instructions | Establish general coding behavior and truthful reporting |
| Repository instructions | Explain project conventions and test commands |
| Tools | Read files, edit code, inspect diffs and execute tests |
| Skill loading | Make `greeting-feature` available when relevant |
| Context management | Preserve requirements, decisions and useful observations |
| Execution controls | Restrict edits and block dependency changes |
| Completion checks | Require evidence that the requested feature works |
| Session log | Record actions and results for inspection |

These are design choices for our example, not a claim that every harness automatically supplies them.

We can also separate the layers by their jobs:

- **Repository instructions:** What conventions apply here?
- **Skills:** What reusable procedure fits this task?
- **Tools:** What actions can the agent take?
- **Hooks and controls:** What checks or restrictions run outside the model’s judgment?
- **Session records:** What happened during execution?

Harness designs can include these kinds of instructions, tools, middleware and state-management mechanisms.

### Putting it all together

Let’s assume our fictional repository uses `unittest`, stores greeting translations in `greeting.py`, and documents its conventions in `AGENTS.md`.

Our proposed harness permits changes only to the greeting implementation and its tests.

An illustrative execution could look like this:

```text
User:
Add the greeting feature.
Support English and Spanish, handle blank names,
follow repository conventions, and pass the tests.
Do not add dependencies or change unrelated files.

Harness:
Load base instructions and repository instructions.
Expose permitted tools and available skill metadata.
Apply the iteration budget and execution controls.

Agent → Read greeting.py and tests/test_greeting.py.

Tool result:
The application already has a localization mapping.
The repository uses unittest.
Greeting behavior belongs in greeting.py.

Agent → Activate greeting-feature.
Agent → Read references/greeting-rules.md.

Tool result:
English fallback: "friend"
Spanish fallback: "amigo"
Trim surrounding whitespace.
Preserve the existing default language.

Agent → Add tests for:
        - named users in English
        - named users in Spanish
        - empty names
        - whitespace-only names
        - names with surrounding whitespace

Agent → Run the relevant tests.

Tool result:
New tests fail against the existing implementation.

Agent → Update greeting.py using the existing mapping.
Agent → Run the relevant tests again.

Tool result:
All greeting tests pass.

Agent → Run the repository's full test command.

Tool result:
The full suite passes.

Agent → Inspect the diff.

Harness completion checks:
Only permitted files changed.
No dependencies added.
Required cases covered.
Verification results recorded.

Agent:
Implemented English and Spanish greetings,
including blank-name fallbacks.
Updated the greeting tests.
The relevant tests and full suite passed.
No dependencies or unrelated files were changed.
```

The model still chooses actions. But our design gives those actions context, execution boundaries and observable results.

The harness is not another clever prompt. It is the system that makes prompts, tools, skills and feedback usable together.

## 5. Real Harness Examples

You do not have to build all this machinery yourself. OpenCode, Hermes and Pi offer different approaches to the surrounding agent system. Their documented capabilities illustrate different design priorities.

### [OpenCode](https://opencode.ai): A Coding-Focused Environment

OpenCode is an open-source coding agent available through terminal, IDE and desktop interfaces, with support for multiple model providers.

Its two main built-in agents are:

- **Build:** the default development agent, with all tools enabled.
- **Plan:** a restricted planning and analysis agent. By default, file edits and shell commands require approval. It is therefore more precise to call it restricted than unconditionally read-only.

You can switch between primary agents using the Tab key. OpenCode also documents three built-in subagents:

- **General:** for complex research and multi-step work.
- **Explore:** for fast, read-only codebase exploration.
- **Scout:** for read-only external documentation and dependency-source research, including inspecting repositories in a managed cache.

Its toolset includes file reading, editing, writing, pattern searches and shell execution, with permissions configurable to allow, deny or request approval.

For our greeting task, a possible workflow would be to use Plan to inspect the repository and propose changes, then Build to implement and verify them.

The useful lesson is that **planning and execution can have different permissions**.

### [Hermes](https://hermes-agent.nousresearch.com): Reuse Knowledge Across Sessions

Hermes is an open-source agent from Nous Research whose documentation emphasizes persistent memory, skill creation from experience and skill improvement during use.

Its documented capabilities include:

- Persistent memory and cross-session recall.
- Agent Skills-compatible procedural knowledge.
- Isolated subagents for delegated work.
- Scheduled automation.
- Messaging-platform integrations.
- Multiple terminal backends.
- Web tools and MCP integration.
- Batch processing and trajectory export for research workflows.

For our example, the interesting possibility is not just fixing today’s greeting feature. It is retaining a reusable procedure for similar work later.

We might configure a workflow that saves the approved greeting rules as a skill after review, so future tasks can reuse them.

That is a different feedback cycle:

```text
Within a task:
Act → observe → adjust.

Across tasks:
Capture a useful procedure → retain it → reuse it.
```

### [Pi](https://pi.dev): A Minimal Harness You Can Shape

Pi describes itself as a minimal, extensible agent harness. It can inspect files, run commands, edit content and work through multi-step tasks from a terminal.

Its customization mechanisms include skills, prompt templates, extensions and shareable packages. It also supports session branching, configurable context management and embedding through an SDK.

Rather than prescribing every workflow, Pi keeps its core minimal and provides extension mechanisms for features such as planning, subagents and permission gates.

For our greeting example, we could propose a small setup:

- Repository instructions for conventions.
- The `greeting-feature` skill.
- An extension that protects unrelated paths.
- A completion check tied to the repository’s tests.

One caution matters: Pi's documentation states that its tools run with the permissions of the Pi process. Project trust controls do not themselves sandbox tool calls.

**Minimal does not mean automatically isolated.** Execution safety still needs deliberate configuration.

### Which approach fits?

For our fictional task, I would frame the choice around the workflow rather than declare a universal winner:

- Consider **OpenCode** for a coding-focused environment with distinct planning and execution roles.
- Consider **Hermes** when persistent procedures and cross-session workflows are central.
- Consider **Pi** when you want a small core that you can customize around your own process.

The question is not just “Which model should I use?”

It is also: **What surrounding system does this task need?**

## Conclusion: We Did Not Make the Model Better at Saying Hello

We started with a plausible function:

```python
def greet(name):
    return f"Hello, {name}!"
```

Then we added three layers:

| Concept | What it contributes |
| --- | --- |
| Agent loop | Feedback that can change the next action |
| Skill | A reusable procedure and supporting knowledge |
| Harness | The execution system, context and controls |

The task stayed the same. What changed was how the system approached it.

We did not make the model better at saying hello. We gave it a way to act, observe the consequences, reuse relevant knowledge and work within a controlled environment.

**The loop closes the feedback cycle. The skill packages the procedure. The harness brings the working system together.**

---
