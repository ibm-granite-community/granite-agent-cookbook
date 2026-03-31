# Context Management for AI Agents

## 1. Introduction: Context Engineering

The context window is an agent's entire working memory. Unlike a human, an agent cannot glance at a whiteboard or flip back through handwritten notes. Every piece of information it needs : task description, tool schemas, conversation history, retrieved documents, prior tool outputs must fit within a fixed token budget, and everything that's in there influences every output.

---

## 2. Anatomy of an Agent's Context

### 2.1 Three Context Layers

**Model Context**

The live token stream the LLM sees at each call: system prompt, conversation turns, tool definitions, and tool results. This is the only thing that actually influences the model's next output. It is ephemeral by default, reset on every new call unless you explicitly carry it forward.

**Tool / Store Context**

State that lives outside the model but can be injected on demand: a LangGraph state object, an in-memory key-value store, a Postgres-backed long-term memory, a vector database. Think of this as the agent's filing cabinet.

**Life-cycle Context**

Everything that persists across agent runs: user preferences, prior task summaries, learned facts. In LangGraph this maps to a persistent store that survives thread boundaries.

---

## 3. Why Context Management is Hard: The Four Failure Modes

> Sources:
>
> - <https://www.trychroma.com/research/context-rot#needle-in-a-haystack-extension>
> - *Context Rot: How Increasing Input Tokens Impacts LLM Performance*

### Failure Mode 1: Context Poisoning

A hallucination or incorrect inference enters the context and is treated as ground truth on subsequent turns. Because the model's outputs feed back into its own input, the error compounds. The longer the context, the more opportunities for poisoned information to be referenced and reinforced.

### Failure Mode 2: Context Distraction

When context grows so large that the model over-focuses on accumulated history and neglects what it learned during training. The Gemini team observed this directly with their Pokémon-playing agent: beyond roughly 100,000 tokens, the model stopped generating novel plans and started repeating actions from its history. This is not a context window limit, it is a reasoning quality limit that kicks in well before the hard ceiling.

### Failure Mode 3: Context Confusion

Superfluous or irrelevant information in the context window actively degrades response quality. Chroma's research makes this concrete: even a single distractor : a passage topically related to the question but that doesn't answer it measurably reduces accuracy. Adding four distractors compounds this degradation further. The model feels compelled to use all available context, and irrelevant tokens pull it off target.

### Failure Mode 4: Context Clash

Contradictory information accumulates in the context. Tightly coupled multi-agent systems are especially prone to this: if two subagents independently write different sections of a report, the combined output will be incoherent because each made assumptions the other violated. Clash also arises when new tool results contradict earlier ones that are still in the history.

---

## 4. Strategy Overview

| Strategy | Fixes |
| --- | --- |
| FIFO / Rolling Window | Distraction |
| Compaction / summarization | Distraction |
| Dynamic Tool Selection | Confusion |
| Context Pruning | Confusion, Poisoning |
| Structured Note-Taking | Distraction, Clash |
| Filesystem as Scratchpad | Distraction, Clash |
| Semantic Compression | Distraction, Confusion |
| RAG / Vector Retrieval | Confusion |
| Sub-agent Architectures | Clash, Distraction |

---

### Strategy 1 — FIFO / Rolling Window

**Failure mode addressed:** Context Distraction

The simplest context management strategy: keep only the N most recent messages. When the message list exceeds the limit, the oldest messages are dropped. First-in, first-out.

This directly combats Context Distraction. By capping history length you prevent the context from growing to the point where the model over-focuses on accumulated turns. The trade-off is that information from early in the conversation is permanently lost.

| Advantages | Limitations |
| --- | --- |
| ✅ Trivial to implement | ❌ Permanently discards early context |
| ✅ Zero infrastructure required | ❌ Invisible to the agent, it cannot know what was lost |
| ✅ Predictable token budget | ❌ Poor fit for tasks requiring long-range coherence |

---

### Strategy 2 — Compaction / Summarization

**Failure mode addressed:** Context Distraction

Instead of silently dropping old messages, compaction uses an LLM to generate a summary of the conversation so far, then replaces the raw history with that summary. The summary becomes the new "base" for future turns. When the window approaches capacity, it compacts, replacing the full history with a structured summary.

The key distinction from pruning (Strategy 4): summarization condenses all content into a shorter form. Pruning removes only the irrelevant content. Use summarization when all the history is broadly relevant but verbose. Use pruning when some parts are clearly irrelevant.

> ⚠️ **Warning: Information Loss Risk**
>
> Cognition and Manus warn that summarization must be done carefully to avoid losing critical information. Cognition uses a fine-tuned model specifically for this step to ensure key events are preserved. Manus actively discourages summarization and prefers context offloading (Strategy 5) instead, because offloading avoids the information loss problem entirely.
>
> If you use summarization, treat the compaction model as a first-class component that needs its own evaluation and prompt engineering.

| Advantages | Limitations |
| --- | --- |
| ✅ Preserves compressed form of all history | ❌ Risk of information loss , fine-tune or carefully prompt the compaction model |
| ✅ Agent sees a coherent narrative, not an abrupt cutoff | ❌ Adds latency and cost at compaction time |
| ✅ Works well for long-running conversational agents | ❌ Manus recommends offloading over summarization for tool-heavy agents |

---

### Strategy 3 — Dynamic Tool Selection

**Failure mode addressed:** Context Confusion

Every tool definition you bind to the LLM consumes tokens and introduces potential confusion. Research from the RAG-MCP paper found that above 30 tools with overlapping descriptions, tool selection accuracy degrades sharply; above 100 tools, failure is near-certain. The "Less is More" paper showed Llama 3.1 8B fails a benchmark with 46 tools but succeeds with 19.

The solution: embed all tool descriptions in a vector store and retrieve only the semantically relevant tools at each LLM call. The RAG-MCP team found this yielded up to 3x improvement in tool selection accuracy.

| Advantages | Limitations |
| --- | --- |
| ✅ Directly prevents Context Confusion from overlapping tool definitions | ❌ Requires maintaining embedded tool index |
| ✅ Reduces prompt length and cost per call | ❌ Retrieval can miss tools for novel query types |
| ✅ Scales to large tool registries (100+ tools) | ❌ Adds one embedding lookup per LLM call |

---

### Strategy 4 — Context Pruning

**Failure mode addressed:** Context Confusion, Context Poisoning

Pruning removes only the parts of a retrieved document or tool output that are irrelevant to the current task. Where summarization condenses all content, pruning is surgical, it removes only the parts that are irrelevant, leaving the relevant parts intact. This maps directly to the Chroma finding that even a single distractor degrades accuracy; pruning prevents distractors from entering the context in the first place.

| Advantages | Limitations |
| --- | --- |
| ✅ Directly prevents Context Confusion and Poisoning | ❌ Adds LLM call per tool output (or dedicated model overhead) |
| ✅ Preserves exact wording, no risk of paraphrase errors | ❌ May over-prune if the request is ambiguous |
| | ❌ Pruning model needs evaluation for your specific domain |

---

### Strategy 5 — Structured Note-Taking (Context Offloading to State)

**Failure modes addressed:** Context Distraction, Context Clash

Instead of letting tool outputs pile up in the message list, the agent actively writes key findings to a structured scratchpad, a dedicated key in LangGraph state and reads from it selectively. The message list stays lean; the scratchpad holds the accumulating knowledge.

This mirrors how humans work: we take notes during research, then synthesize those notes into a final output.

> **Manus: `todo.md` as Recitation**
>
> Manus tasks involve 50+ tool calls and are extremely token-heavy. Their key innovation: the agent creates a `todo.md` plan at the start and continuously rewrites it throughout task execution. This is not just note-taking, it is *recitation*. By rewriting the plan at each step, the agent is forced to re-contextualize where it is in the task, which keeps it on track even as the raw context fills up.

| Advantages | Limitations |
| --- | --- |
| ✅ Agent remains aware of its own notes without bloating the message list | ❌ Requires the agent to use the write tool, prompt engineering is essential |
| ✅ Rewriting the plan (Manus pattern) keeps the agent on track for long tasks | ❌ Scratchpad is within-session only (see Strategy 6 for cross-session persistence) |
| ✅ Avoids information loss unlike summarization, the agent chooses what to keep | ❌ Poor agent instruction can result in under-used scratchpad |

---

### Strategy 6 — Filesystem as Scratchpad

**Failure modes addressed:** Context Distraction, Context Clash

When tasks are long enough to exceed even a well-managed context window, the filesystem becomes the memory system. The agent writes plans, intermediate results, and findings to files; reads them back selectively; and uses `grep` or search to locate specific information without loading entire files.

This is the approach taken by the Anthropic multi-agent research system for tasks that could exceed 200,000 tokens. Manus uses this as their preferred alternative to summarization avoiding the information loss risk entirely.

| Advantages | Limitations |
| --- | --- |
| ✅ No information loss : full content is always recoverable | ❌ Requires filesystem access or object storage |
| ✅ Works for tasks that span multiple agent runs | ❌ Agent must be prompted to use file tools consistently |
| ✅ Selective reading (grep) keeps context focused on what's needed now | ❌ File management adds coordination overhead for multi-agent systems |
| ✅ Preferred by Manus over summarization for tool-heavy agents | |

---

### Strategy 7 — Semantic Compression

**Failure modes addressed:** Context Distraction, Context Confusion

Semantic compression goes further than summarization. Rather than summarizing conversation turns, it identifies the minimal set of concepts and facts that must be preserved for future reasoning, and encodes them in a structured, compact form, sometimes as structured JSON, sometimes as a knowledge graph, sometimes as condensed prose with explicit preservation rules.

Cognition (Devin) uses a fine-tuned model for this step, trained specifically to preserve the events and facts that matter for software engineering tasks.

| Advantages | Limitations |
| --- | --- |
| ✅ Preserves semantically important information even under aggressive compression | ❌ No standard LangChain abstraction : must build from scratch |
| ✅ Can be tuned per domain (coding, research, customer service) | ❌ Requires careful prompt engineering or fine-tuning per domain |
| ✅ Produces a structured, inspectable summary | ❌ Hardest to evaluate, information loss may be subtle |

---

### Strategy 8 — RAG / Vector Retrieval

**Failure mode addressed:** Context Confusion

Retrieval-Augmented Generation is the act of selectively retrieving relevant information from an external corpus and injecting it into the context window at call time. Rather than pre-loading all knowledge, the agent fetches only what is relevant to the current query.

RAG directly addresses Context Confusion: only semantically relevant chunks enter the context, so the model is not distracted by irrelevant passages. Chroma's LongMemEval results showed that models given focused, relevant context (~300 tokens) vastly outperformed models given the full 113k-token history, even though the full history contained all the necessary information.

| Advantages | Limitations |
| --- | --- |
| ✅ Only relevant information enters the context | ❌ Retrieval quality depends on embedding model and chunking strategy |
| ✅ Scales to arbitrarily large knowledge bases | ❌ Can miss relevant information if the query phrasing doesn't match chunk content |
| ✅ Mature ecosystem : many vector store backends available | ❌ Adds infrastructure complexity (vector store, embedding pipeline) |

---

### Strategy 9 — Sub-agent Architectures (Context Quarantine)

**Failure modes addressed:** Context Clash, Context Distraction

Context Quarantine means isolating different parts of a task in separate LLM context windows : each subagent sees only its own thread. This prevents Context Clash and enables each subagent to operate with a clean, focused context.

A multi-agent setup with a lead agent and subagents often outperforms single-agent on research benchmarks largely because each subagent can pursue an independent research direction without accumulating irrelevant context from other directions. With 10 subagents each having a 200k-token window, the system can process 2 million tokens of total information.

> ⚠️ **Coordination Warning:** Constrain subagents to *information gathering*, not decision making. Let a single agent do all the synthesis and writing at the end. If the subtasks must be tightly coordinated to produce a coherent whole, keep them in a single agent.

| Advantages | Limitations |
| --- | --- |
| ✅ Prevents Context Clash between independent research threads | ❌ High orchestration complexity |
| ✅ Massive effective context window (N subagents × model context) | ❌ Tightly coupled tasks will produce contradictions |
| ✅ Each subagent can have its own specialized tool loadout and prompt | ❌ Coordination overhead adds latency |
| ✅ Anthropic: 90.2% improvement over single-agent on research benchmarks | ❌ Debugging multi-agent traces is significantly harder |

---

## 5. Deployment Considerations

### Strategy Stack by Scenario

| Scenario | Recommended Stack |
| --- | --- |
| Short conversational agent (< 20 turns) | Strategy 1 Rolling Window |
| Long-running chatbot (100+ turns) | Strategy 2 (Compaction) + Strategy 5 (Note-Taking) |
| Tool-heavy research agent | Strategy 4 (Pruning) + Strategy 5 (Note-Taking) + Strategy 6 (Filesystem) |
| Agent with large tool registry (30+ tools) | Strategy 3 (Dynamic Tool Selection) |
| Knowledge-intensive Q&A agent | Strategy 8 (RAG) + Strategy 4 (Pruning) |
| Parallel research / breadth-first tasks | Strategy 9 (Sub-agents) information gathering only |
| Long-running multi-session agent | Strategy 6 (Filesystem) + persistent Store |

> When evaluating your agent's context management, do not rely solely on Needle-in-a-Haystack (NIAH) benchmarks. NIAH measures a narrow capability `lexical retrieval` but models that score near-perfectly on NIAH can still degrade badly on semantic reasoning tasks as context grows. Build evaluation sets that reflect your actual task distribution, include distractor content, and test at context lengths beyond your expected operating range.

---

## 6. Conclusion

Context management is the hardest part of building reliable agents. A few themes cut across all strategies:

- **Context is not free.** Every token influences the model's output. Treat context budget like memory budget.
- **Know your failure mode.** Poisoning, Distraction, Confusion, and Clash require different mitigations.
- **Information gathering and decision making should be separated.** Sub-agents work well for the former; single agents work better for the latter.
- **Summarization loses information.** Cognition fine-tunes a model for it; Manus avoids it entirely. Offloading to files or a scratchpad is safer.
- **Benchmarks understate the problem.** NIAH scores are misleading. Test at your actual context lengths with actual distractors.
- **Structure your context.** Typed LangGraph state makes pruning, summarization, and selective retrieval dramatically easier.

---

## References

| # | Source | Link |
| --- | --- | --- |
| 1 | Anthropic Engineering, Effective Context Engineering for AI Agents | <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents> |
| 2 | LangChain Blog, Context Engineering for Agents | <https://blog.langchain.com/context-engineering-for-agents/> |
| 3 | How Contexts Fail and How to Fix Them | <https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html> |
| 4 | An Agentic Case Study: Playing Pokémon with Gemini | <https://www.dbreunig.com/2025/06/17/an-agentic-case-study-playing-pok%C3%A9mon-with-gemini.html> |
| 5 | Context Rot: How Increasing Input Tokens Impacts LLM Performance | <https://www.trychroma.com/research/context-rot#needle-in-a-haystack-extension> |