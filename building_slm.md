# Techniques for Building Agentic Applications with Small Language Models

Building AI agents is no longer the exclusive domain of organizations with massive compute budgets. Agentic systems — where a model plans, calls tools, reflects on results, and iterates — can trigger dozens of model invocations per user query. At frontier-model pricing, those costs compound fast. Small Language Models (SLMs) are proving that size isn't everything: they deliver faster responses, lower costs per call, and more predictable behavior in the constrained, repetitive tasks that make up most of an agent's work.

This guide covers the techniques you need to build effective agentic applications with SLMs. Rather than viewing them as compromised versions of their larger siblings, we'll examine how SLMs can be the optimal choice for agent workloads — and how to structure your system so each component uses the right-sized model.

## What is a Small Language Model?

Small Language Models are AI systems designed to process and generate natural language, but with a significantly reduced parameter count compared to their large counterparts. The distinction isn't just about size - it reflects a fundamental difference in design philosophy and use case optimization.

**Parameter Range** - SLMs typically contain between a few million to around 10-20 billion parameters. To put this in context, models like Llama 3.2 1B or Phi-4-Mini 3.8B fall squarely in SLM territory, while GPT-4 operates with hundreds of billions or potentially trillions of parameters. The exact threshold varies by definition, but anything between 100 million and 10 billion parameters is generally considered "small" in today's landscape.

**Architecture** - Like their larger relatives, SLMs are built on transformer architectures and neural networks. However, they employ various model compression techniques to maintain capability while reducing size. Recent innovations like IBM's Granite 4.0 hybrid models combine Mamba-2 state space models with transformers, achieving over 70% memory reduction compared to pure transformer architectures of similar capability.

**Training Approach** - SLMs are often created through knowledge distillation, where a smaller "student" model learns from a larger "teacher" model. This process transfers the essential patterns and capabilities without requiring the full parameter space. Additionally, SLMs may be trained on more focused, domain-specific datasets rather than attempting to capture all of human knowledge.

**Specialization vs. Generalization** - This is perhaps the most important conceptual difference. While LLMs aim for broad, general-purpose intelligence, SLMs excel at targeted applications. They trade breadth for depth, becoming specialists rather than generalists.

## When and Why to Use SLMs

The decision to use an SLM isn't about settling for less - it's about optimizing for your actual requirements. Several factors make SLMs not just viable, but preferable in many scenarios.

**Agent Applications** - This is the most compelling case for SLMs today. A single user query to an agentic system may trigger 5–20 model calls: intent classification, tool selection, parameter extraction, result summarization, self-critique, and re-planning. At frontier-model rates, this compounds into significant cost per interaction. SLMs let you run these loops affordably — often at 10–30x lower cost per call — without sacrificing the latency that makes agents feel responsive. Most steps in an agentic loop are constrained, repetitive tasks (classify, extract, route) where a well-prompted SLM matches or exceeds a larger model's accuracy.

**Cost Considerations** - The economics are compelling. Running a 3B parameter model can be 10-30x cheaper than operating a 405B parameter model. This isn't just about inference costs - training, fine-tuning, and operational expenses all scale down proportionally. For applications requiring high throughput or continuous operation, these savings compound dramatically. A customer service chatbot handling millions of interactions monthly could see cost reductions from thousands to hundreds of dollars.

**Hardware Considerations** - SLMs democratize AI deployment. They can run on consumer-grade GPUs, single A100s, or even CPUs in some cases. Where LLMs might require multi-GPU setups with 80GB+ VRAM per card, an SLM like Granite 4.0 H-Micro runs comfortably on a 15GB T4 GPU. This means organizations can deploy AI capabilities without investing in specialized infrastructure, and development teams can experiment locally without cloud dependencies.

**Edge Device Considerations** - The real revolution happens at the edge. SLMs enable on-device AI for smartphones, IoT devices, and embedded systems. Models like Llama 3.2 1B or Apple's on-device 3B model can run entirely on mobile hardware, processing data locally without network latency or privacy concerns. This unlocks use cases from offline language translation to real-time voice assistants that work in network-constrained environments.

**Scope of Applications** - When your application has well-defined boundaries, SLMs shine. Consider these scenarios: structured data extraction from documents, classification tasks with known categories, domain-specific question answering, code completion for specific frameworks, or sentiment analysis. These tasks don't require the open-ended reasoning of frontier models - they need reliable, fast execution within a constrained domain. An SLM fine-tuned for medical record processing will outperform a general LLM at that specific task while costing a fraction to operate.

The strategic insight is recognizing that most steps in an agentic workflow are repetitive and scoped — requiring efficiency and predictability over creative brilliance.

## How to Identify an SLM Model

Distinguishing SLMs from LLMs requires looking beyond marketing materials to technical specifications.

**Parameter Count** - This is the most straightforward indicator. Check the model card or documentation for total parameters. Generally: under 1B = nano models, 1-7B = small models, 7-20B = medium-small models, 20B+ starts entering LLM territory. However, architecture matters - a 7B MoE (Mixture of Experts) model with only 1B active parameters behaves more like a traditional 1B model in terms of inference.

**Model Families to Know** - Several prominent SLM families are worth tracking: Phi series from Microsoft (Phi-4-Mini at 3.8B), Llama 3.2 series (1B and 3B variants), Qwen 2.5 series (ranging from 0.5B to 7B), Gemma models from Google (2B and 9B), IBM Granite 4.0 family (350M nano to 32B small with 9B active), SmolLM2 from HuggingFace (135M to 1.7B), and Ministral from Mistral AI (3B and 8B).

**Memory Footprint** - Practical identification: if the model loads in under 4GB of VRAM at float16 precision, it's definitely an SLM. Models requiring 8-16GB are in the small-to-medium range. Anything demanding 24GB+ is firmly LLM territory.

**Intended Use Case** - Model cards often explicitly state whether the model is designed for edge deployment, specific task types, or resource-constrained environments - strong indicators of SLM positioning.

## SLMs vs LLMs: Understanding the Trade-offs

The choice between SLMs and LLMs isn't binary - it's about matching capabilities to requirements. Here's a comprehensive comparison:

| Dimension | Small Language Models (SLMs) | Large Language Models (LLMs) |
| --------- | ---------------------------- | ---------------------------- |
| **Flexibility** | Domain-specific, excels within trained scope. Less adaptable to novel task types without fine-tuning. | Highly flexible, handles diverse tasks out-of-box. Strong zero-shot generalization across domains. |
| **Cost** | 10-30x lower inference costs. Minimal infrastructure requirements. Fast, cheap fine-tuning ($100s vs $10,000s). | High operational costs, especially at scale. Requires significant GPU infrastructure. Expensive to fine-tune. |
| **Accuracy (Task-Specific)** | Superior accuracy on domain-specific tasks when properly fine-tuned. Can match or exceed LLMs in narrow domains. | Strong general accuracy but may underperform specialized SLMs in specific domains. Often "good enough" without tuning. |
| **Accuracy (General)** | Limited performance on tasks outside training scope. Weaker open-ended reasoning and cross-domain transfer. | Excels at complex reasoning, multi-step problems, and tasks requiring broad knowledge. |
| **Deployment** | Runs on consumer hardware, mobile devices, edge infrastructure. Easy to replicate and scale horizontally. | Requires specialized hardware (multi-GPU setups). Cloud-dependent for most use cases. Limited deployment flexibility. |
| **Latency** | Sub-100ms response times common. Real-time processing feasible. No need for heavy parallelization. | Higher latency due to size. Requires batching and optimization for reasonable speeds. Can't match SLM response times. |
| **Development Cycle** | Fast iteration: fine-tune in hours or days. Easy experimentation. Quick to deploy updates. | Slow iteration: weeks for fine-tuning. Expensive experimentation. Deployment requires careful planning. |
| **Context Window** | Often limited (2K-8K tokens typical). Some recent models support 32K-128K tokens. | Large context windows (32K-200K+ tokens). Better for long document processing. |
| **Robustness** | More prone to errors on ambiguous inputs or adversarial examples. Requires careful input validation. | More robust to edge cases and ambiguous queries. Better error handling out-of-box. |

**The Heterogeneous Approach** - The most sophisticated agent systems don't choose one or the other exclusively. They use SLMs for routine operations (classification, extraction, tool selection) and invoke LLMs selectively for complex reasoning or open-ended tasks. Think of SLMs as the efficient workers and LLMs as expert consultants — you don't need the consultant for every step in the loop.

## Techniques for Working with SLMs

Operating SLMs effectively in agentic systems requires different techniques than prompting large models. Smaller models need more explicit guidance and contextual support.

### Prompting Tips for SLMs

Prompting discipline matters more with SLMs than with frontier models. Smaller models have less latent knowledge and weaker emergent reasoning — they rely on pattern matching against examples and precise instructions rather than abstract understanding. That predictability is an asset: when you provide clear structure, SLMs follow it reliably.

> For comprehensive SLM-specific prompting techniques, see [Prompting SLMs for Agentic Applications: Best Practices](https://github.com/ibm-granite-community/granite-agent-cookbook/blob/main/model_prompting_best_practices.md)

A few key principles:

- **Be explicit about role and output format.** Start with "You are an expert X" and end with a concrete format constraint ("Return only valid JSON", "Choose exactly one of: [A, B, C]").
- **Always include examples.** For any task, include 3–10 input/output demonstrations. SLMs rely on pattern matching against these far more than LLMs do.
- **Use retrieval-augmented few-shot selection.** Rather than fixed examples, dynamically select the most semantically similar examples from a larger pool. This outperforms static examples across diverse inputs.

Example — few-shot entity extraction with a 3B model:

```text
Extract person names and locations from the following texts.

Text: "John Smith visited Paris last summer."
Output: {"persons": ["John Smith"], "locations": ["Paris"]}

Text: "The conference in Tokyo was attended by Dr. Sarah Chen and Prof. Michael Brown."
Output: {"persons": ["Dr. Sarah Chen", "Prof. Michael Brown"], "locations": ["Tokyo"]}

Text: "Alice went to London and met Bob at the museum."
Output: {"persons": ["Alice", "Bob"], "locations": ["London"]}

Now extract from: "Mary Johnson and her colleague traveled to Berlin for the summit."
Output:
```

### Retrieval-Augmented Generation (RAG)

RAG is particularly powerful for SLMs because it compensates for their limited parametric knowledge by providing relevant context at inference time.

**Why RAG is Critical for SLMs** - SLMs can't memorize vast knowledge bases like LLMs. But they can effectively process and reason over provided context. RAG gives them just-in-time knowledge, transforming them from limited specialists into adaptable experts within their domain.

**Implementation Pattern** - The workflow: user query comes in, embedding model converts query to vector representation, vector database retrieves top-k most relevant documents, retrieved content is injected into SLM prompt as context, and SLM generates response grounded in provided evidence.

**Optimizing RAG for SLMs** - Keep retrieved context concise (2-3 documents, 1000-2000 tokens total) since SLMs have limited context windows. Use reranking to ensure only the most relevant excerpts are included. Implement hierarchical chunking - retrieve larger document chunks first, then extract the most relevant sub-sections to fit within token limits.

Structure RAG prompts carefully:

```text
Answer the question using ONLY the provided context. If the answer isn't in the context, say "I don't have enough information."

Context:
{retrieved_documents}

Question: {user_question}

Answer:
```

**RAG + Few-Shot Combination** - For best results with SLMs, combine RAG with few-shot prompting. Show examples of how to properly cite sources and acknowledge uncertainty:

```text
Context: "The Granite 4.0 models use a hybrid Mamba-2 architecture."
Question: "What architecture does Granite 4.0 use?"
Answer: "Based on the provided context, Granite 4.0 models use a hybrid Mamba-2 architecture."

Context: "IBM released Granite models under Apache 2.0 license."
Question: "What is the pricing for Granite?"
Answer: "The context doesn't provide pricing information, but notes the models are released under Apache 2.0 license, suggesting they are open source."
```

### Tool Calling & Structured Function Invocations

Modern SLMs like Granite 4.0, Phi-4-Mini, and Llama 3.2 support native tool calling, but effective use requires careful design.

**Function Definition Strategy** - Keep tool sets minimal. An SLM should have 5-15 tools maximum for any single workflow. More tools increase selection errors. Make function signatures explicit and unambiguous. Use descriptive names, clear parameter types, and comprehensive docstrings.

**Schema-First Design** - Define strict JSON schemas for tool parameters. Many SLMs support constrained decoding - they can be forced to output valid JSON matching your schema, eliminating parse errors. Use guided decoding libraries like Outlines or XGrammar to guarantee schema conformance.

Example for Granite 4.0:

```python
tools = [
    {
        "name": "search_products",
        "description": "Search product catalog by name or category",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {
                    "type": "string", 
                    "enum": ["electronics", "clothing", "books"],
                    "description": "Product category filter"
                }
            },
            "required": ["query"]
        }
    }
]

# Model generates: {"name": "search_products", "arguments": {"query": "laptop", "category": "electronics"}}
```

**Error Handling** - SLMs make more tool selection mistakes than LLMs. Implement validation layers: check if tool exists before execution, validate parameters against schemas before calling, provide clear error messages back to the model if validation fails, and limit retry attempts to prevent loops.

**Tool Documentation** - Include usage examples in tool descriptions. This acts as in-context learning for tool use:

```json
{
    "name": "calculate_price",
    "description": "Calculate final price with tax and discounts. Example: calculate_price(base_price=100, tax_rate=0.08, discount=0.1) returns 97.20"
}
```

### Orchestration & Subagent Patterns

Agentic workflows benefit enormously from distributing tasks across multiple SLMs rather than relying on a single large model.

**Subagent Architecture** - A main coordinator (which can be a lightweight SLM or even rule-based routing) delegates subtasks to specialized SLM workers. Each worker has specific tools and domain expertise. Example system for customer support: intent_classifier SLM (1B model, determines query type), product_expert SLM (3B model, fine-tuned on product documentation), billing_expert SLM (3B model, fine-tuned on billing policies), and escalation_writer SLM (7B model, drafts complex responses).

**Benefits Over Monolithic Agents** - Isolation prevents cross-domain confusion. The billing expert never sees product questions, improving accuracy. Parallel execution where multiple experts can work simultaneously on different aspects of a complex query. Independent iteration allowing you to fine-tune or replace one expert without touching others. Cost optimization by using appropriately-sized models (use 1B for simple classification, 7B only when needed).

**Handoff Protocols** - Define clear handoff contracts. When one subagent completes its task, it returns structured output that the next agent can consume:

```json
{
    "task": "intent_classification",
    "result": {"intent": "billing_question", "confidence": 0.92},
    "next_agent": "billing_expert",
    "context": {"user_id": "12345", "question": "Why was I charged twice?"}
}
```

**Verification Cascades** - Chain multiple SLMs for verification. A primary SLM generates output, a verifier SLM checks for correctness (fact-checking, logic validation), and only confident outputs pass through. Low-confidence cases fall back to a larger model or human review. This dramatically improves reliability without running expensive models on every request.

### Programmatic Complexity Over LM Complexity

A key insight for SLM deployment: you can often achieve better results by adding conventional code logic rather than expecting the model to handle everything.

**State Machines and Business Rules** - Don't ask an SLM to remember multi-turn conversation state. Maintain state externally in your application. Use traditional conditional logic for deterministic decisions: "if user is premium tier, apply discount" doesn't need ML. Only invoke the SLM for tasks requiring language understanding or generation.

**Pre and Post-Processing** - Input validation and sanitization happen in code before the SLM sees data. Output parsing and verification happen in code after the SLM responds. Let the model do language, let code do logic.

Example pattern:

```python
def handle_query(user_input: str):
    # Code: Input validation
    if len(user_input) < 5:
        return "Please provide more details."
    
    # Code: Intent detection (could use simple keyword matching)
    if "refund" in user_input.lower():
        agent = billing_agent
    else:
        agent = general_agent
    
    # SLM: Language understanding and generation
    response = agent.generate(user_input)
    
    # Code: Output validation
    if not is_valid_response(response):
        response = fallback_response()
    
    return response
```

This hybrid approach plays to each component's strengths - predictable code execution for logic, flexible language models for language.

For a deeper look at patterns that blend code logic and model calls in production agents, see [Mellea](https://mellea.ai/).

## Fine-Tuning Consideration

As you build your agent application, you may eventually reach a point where prompting, RAG, and careful tool design aren't enough to hit your accuracy target for a specific task. At that point, fine-tuning is worth considering — and with SLMs, it's far more accessible than most developers expect.

**Fine-tuning should be a last resort.** Exhaust prompting techniques, RAG integration, and tool design first. Many applications never need it. Fine-tuning adds operational overhead (maintaining a custom model, retraining when data drifts) that prompting-based approaches avoid.

**When it makes sense:** If a specific step in your agent loop — say, classifying support tickets or extracting structured fields from domain documents — consistently falls short despite good prompting, a small fine-tuning run on labeled examples can push accuracy past what a general model achieves.

**Why it's accessible with SLMs:** Parameter-efficient fine-tuning with LoRA/QLoRA means you can adapt a 3B Granite model on a single T4 GPU in a matter of hours, not days. The Apache 2.0 license on IBM Granite models means no restrictions on fine-tuned derivatives. You typically need only 100–500 labeled examples to see meaningful improvement on a well-scoped task.

For code examples and step-by-step setup, see the [IBM Granite Fine-Tuning with Unsloth guide](https://www.ibm.com/granite/docs/fine-tune/unsloth).

## Practical Example: Document Classification System

Let's walk through a concrete example comparing LLM and SLM approaches for a document classification system.

**The Requirement** - Classify incoming customer support tickets into categories: Technical Issue, Billing Question, Feature Request, General Inquiry, Complaint.

**LLM Approach (Baseline)**

```python
# Using GPT-4 (hypothetical)
prompt = f"""Classify the following support ticket into one category:
- Technical Issue
- Billing Question  
- Feature Request
- General Inquiry
- Complaint

Ticket: {ticket_text}

Category:"""

response = llm.generate(prompt)
# Cost: ~$0.03 per 1K tokens, ~500 tokens per ticket = $0.015/ticket
# Latency: ~800ms average
# Accuracy: 94%
```

At 10,000 tickets/month: $150 in API costs, dedicated infrastructure not required but cloud dependency, excellent accuracy with no tuning needed.

**SLM Approach (Optimized)**

```python
# Using Granite 4.0 H-Micro (3B) fine-tuned on 500 labeled tickets
few_shot_examples = """
Example 1:
Ticket: "I can't log into my account, keep getting error 500"
Category: Technical Issue

Example 2:
Ticket: "I was charged twice this month"  
Category: Billing Question

Example 3:
Ticket: "Can you add dark mode to the app?"
Category: Feature Request

"""

prompt = f"""{few_shot_examples}
Now classify:
Ticket: {ticket_text}
Category:"""

response = slm.generate(prompt)
# Cost: ~$0.001 per 1K tokens self-hosted = $0.0005/ticket
# Latency: ~120ms average
# Accuracy: 96% (after fine-tuning)
```

At 10,000 tickets/month: $5 in compute costs (self-hosted), one-time $200 fine-tuning cost, 97% accuracy after optimization, runs on single GPU or even CPU.

**Key Improvements in SLM Version:**

1. Few-shot examples provide clear patterns
2. Fine-tuning on domain data improved accuracy beyond the LLM
3. Structured output format reduces parsing errors
4. 30x cost reduction with better latency
5. No external API dependency

**When to Still Use LLM:**
If ticket requires multi-step reasoning ("My billing is wrong BECAUSE you didn't apply the discount I was promised"), the ticket references complex account history requiring synthesis, or the ticket is adversarial/deliberately confusing.

## Next Steps

Ready to put these techniques into practice? The best way to learn is by doing. Here's how to get started building your first SLM-powered agent:

**Start with Granite 4.0** - IBM's Granite models provide an excellent entry point for production SLM work. They're open source (Apache 2.0), well-documented, and designed specifically for enterprise use cases. The hybrid Mamba-2 architecture means you get strong performance without the typical memory overhead of pure transformer models. Granite natively supports structured JSON output, tool calling, and RAG-optimized inference — exactly what agentic applications need.

**Hands-On Tutorial** - For a comprehensive, practical guide to running and fine-tuning Granite models, check out [How to run and fine-tune IBM Granite AI models for your projects](https://allthingsopen.org/articles/how-to-run-and-fine-tune-ibm-granite-ai-models). This tutorial walks you through:

- Setting up your local development environment
- Running Granite models for AI-powered coding assistance
- Fine-tuning on custom datasets for domain-specific tasks
- Implementing time series forecasting with Granite
- Deploying optimized models to production

**Build Your First SLM Agent** - Choose a well-scoped problem: a classification step, a data extraction task, or a routing layer. Start with a baseline using few-shot prompting, build an evaluation dataset (even 50-100 examples is enough), iterate on prompt engineering and RAG integration, and only consider fine-tuning if baseline performance isn't sufficient.

**Measure Everything** - As you experiment, maintain rigorous evaluation practices. Track accuracy, latency, cost, and reliability. The data you collect will guide your optimization efforts and prove ROI to stakeholders. Remember: production AI is an engineering discipline, not just a research problem.

The techniques in this guide aren't theoretical — they're battle-tested approaches from real-world deployments. Your first SLM agent might feel constrained compared to the creative possibilities of frontier models. But when you see the cost savings, deployment flexibility, and reliable performance across dozens of model calls per query, you'll understand why SLMs are becoming the workhorses of production agentic AI.

## Conclusion

The future of production AI agents isn't about choosing between SLMs and LLMs - it's about deploying each where they excel. SLMs have evolved from "budget alternatives" to purpose-built tools that often outperform their larger siblings on the scoped, repetitive tasks that make up most of an agent's work.

The techniques covered here — prompting with examples, RAG augmentation, strategic tool calling, subagent architectures, and programmatic complexity — transform SLMs into reliable, efficient components. Combined with code-side logic to handle deterministic decisions, you get agent systems that are faster, cheaper, and more maintainable than LLM-only approaches.

Start with a single step in your agent loop. Build proper evaluation infrastructure. Iterate. And remember: most steps in an agentic workflow don't need frontier reasoning — they need reliable execution in constrained domains. That's exactly where SLMs dominate.

The era of practical, deployable AI agents at scale has arrived. It's smaller than you think.

---

**Further Reading**

- [IBM Granite 4.0 Documentation](https://github.com/ibm-granite/granite-4.0-language-models)
- [IBM Granite Fine-Tuning with Unsloth](https://www.ibm.com/granite/docs/fine-tune/unsloth)
- [Small Language Models for Agentic Systems (Research Paper)](https://arxiv.org/abs/2510.03847)
- [Prompt Engineering Guide - Few-Shot Learning](https://www.promptingguide.ai/techniques/fewshot)
- [RAG vs Fine-tuning vs Prompt Engineering](https://www.ibm.com/think/topics/rag-vs-fine-tuning-vs-prompt-engineering)
- [Mellea — Programmatic Agent Patterns](https://mellea.ai/)