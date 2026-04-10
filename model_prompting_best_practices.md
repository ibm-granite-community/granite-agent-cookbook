# Prompting SLMs for Agentic Applications: Best Practices for Accurate Agent Responses

## Introduction

When working with Small Language Models (SLMs)—models in agent systems and applications, effective prompting becomes critical. Unlike their larger counterparts, SLMs have limited capacity and require more precise, well-structured prompts to deliver accurate and reliable responses. This guide explores best practices for crafting system prompts and tool definitions that maximize the performance of smaller LLMs within agent frameworks.

## What Are Small Language Models (SLMs)?

For this guide, SLMs refer to models with:

- **Parameter counts**: Typically 1B-13B parameters
- **Context windows**: Usually 2K-8K tokens (vs. 32K-128K+ for larger models)
- **Use cases**: Edge deployment, cost-sensitive applications, specialized domains
- **Resources**: Can usually fit on 1 single compute node. For example, on one retail level GPU.

## System Prompt Design Principles

### Keep It Concise and Clear

Smaller models have limited context windows and processing capacity. Every token in your system prompt matters. Focus on:

- **Direct instructions**: State what the model should accomplish
- **Essential context only**: Include only information directly relevant to the task
- **Simple language**: Use straightforward vocabulary and sentence structures
- **Logical organization**: Present information in a clear, sequential manner
- **Minimal formatting**: Avoid emojis and decorative characters that add no semantic value. Use markdown (bold, bullets) only when it adds clarity

### Prioritize Task Oriented Guidelines

Frame instructions using positive action words rather than negative prohibitions.
This approach is more effective because:

- Positive framing tells the model **what to do** rather than what to avoid
- It reduces ambiguity and cognitive load on the model
- It provides clear direction for desired behavior

Negative instructions often backfire—much like saying *"Don't look behind you"* makes someone immediately turn around, telling a model what *not* to do can inadvertently encourage that exact behavior.

**Example of Negative vs. Positive Framing:**

| Avoid (Negative Framing) | Prefer (Positive Framing) |
| ---------------------------- | ---------------------------- |
| "Do not provide verbose explanations. Do not include unnecessary details." | "Provide concise explanations. Focus on essential details." |
| "Don't forget to validate input parameters." | "Validate all input parameters before processing." |
| "Never use deprecated functions in your code." | "Use current, supported functions in your code." |
| "Don't make assumptions about user requirements." | "Clarify user requirements before proceeding." |
| "Avoid writing code without proper error handling." | "Include comprehensive error handling in all code." |

### Structure for Success

Organize your system prompt with clear sections:

1. **Role Definition**: Establish the model's identity and purpose
2. **Capabilities**: List what the model can do
3. **Guidelines**: Provide behavioral instructions using positive framing
4. **Format Requirements**: Specify output structure expectations
5. **Examples**: Include 1-2 concrete examples when helpful (few-shot prompting)

Few-shot prompting provides concrete examples that demonstrate desired behavior patterns. This is particularly effective for smaller models that benefit from explicit demonstrations.

#### Few-shot prompting: Information Extraction Example

This example demonstrates how few-shot examples significantly improve structured output quality.

**Task**: Extract a person's name, age, occupation, and location from text and return it as JSON.

---

##### Approach 1: Zero-Shot (No Examples)

**System Prompt:**

```markdown
You are a technical assistant specializing in information extraction.

## Core Capabilities
- Extract person_name, age, occupation and location from a given text.
- Return the information in JSON format.
- If a given text does not contain the required information, leave the value empty.

## Response Format
Return only valid JSON.
```

**Input:**

```text
Sarah Chen is a 34-year-old software engineer living in Toronto, Canada.
```

**Output:**

```json
"```json\n{\n  \"person_name\": \"Sarah Chen\",\n  \"age\": 34,\n  \"occupation\": \"software engineer\",\n  \"location\": \"Toronto, Canada\"\n}\n```"
```

**Issue**: The model wraps the JSON in markdown code blocks (` ```json ... ``` `), violating the "valid JSON only" requirement.

---

##### Approach 2: Few-Shot (With Examples in System Prompt)

**Enhanced System Prompt:**

```markdown
You are a technical assistant specializing in information extraction.

## Core Capabilities
- Extract person_name, age, occupation and location from a given text.
- Return the information in JSON format.
- If a given text does not contain the required information, leave the value empty.

## Response Format
Return only valid JSON.

## Examples
Example 1:
Text: "John Smith, a 28-year-old teacher from London, lives in the United Kingdom."
Output:
{
  "person_name": "John Smith",
  "age": 28,
  "occupation": "teacher",
  "location": "London, United Kingdom"
}

Example 2:
Text: "My doctor is Maria Rodriguez. She works from Barcelona, Spain."
Output:
{
  "person_name": "Maria Rodriguez",
  "age": "",
  "occupation": "doctor",
  "location": "Barcelona, Spain"
}
```

**Input:**

```text
Sarah Chen is a 34-year-old software engineer living in Toronto, Canada.
```

**Output:**

```json
"{\n  \"person_name\": \"Sarah Chen\",\n  \"age\": 34,\n  \"occupation\": \"software engineer\",\n  \"location\": \"Toronto, Canada\"\n}"
```

**Result**: Clean JSON output without markdown formatting. The examples taught the model the exact output format expected.

---

##### Approach 3: Few-Shot (With Message History)

For chat completion endpoints, you can alternatively provide examples as actual conversation history:

**System Prompt:**

```markdown
You are a technical assistant specializing in information extraction.

## Core Capabilities
- Extract person_name, age, occupation and location from a given text.
- Return the information in JSON format.
- If a given text does not contain the required information, leave the value empty.

## Response Format
Return only valid JSON.
```

**Message History:**

```json
[
  {
    "role": "system",
    "content": "[System prompt above]"
  },
  {
    "role": "user",
    "content": "Extract information from: 'John Smith, a 28-year-old teacher from London, lives in the United Kingdom.'"
  },
  {
    "role": "assistant",
    "content": "{\n  \"person_name\": \"John Smith\",\n  \"age\": 28,\n  \"occupation\": \"teacher\",\n  \"location\": \"London, United Kingdom\"\n}"
  },
  {
    "role": "user",
    "content": "Extract information from: 'My doctor is Maria Rodriguez. She works from Barcelona, Spain.'"
  },
  {
    "role": "assistant",
    "content": "{\n  \"person_name\": \"Maria Rodriguez\",\n  \"age\": \"\",\n  \"occupation\": \"doctor\",\n  \"location\": \"Barcelona, Spain\"\n}"
  }
]
```

**Result**: Same clean JSON output. This approach uses actual message history instead of embedding examples in the system prompt.

---

##### Choosing Between Approaches

**Both few-shot approaches work effectively.** Choose based on your context:

**System Prompt Examples (Approach 2) are better when:**

- Using base models or instruction-tuned (non-chat) models
- Examples are simple input→output patterns
- You want to preserve conversation history for actual user context
- Token efficiency is critical (system prompts are often cached by providers)

**Message History Examples (Approach 3) are better when:**

- Using chat-tuned models specifically trained on conversational formats
- Examples involve multi-turn interactions or dialogue
- You need the model to maintain conversational flow and context

**Key Takeaway**: Few-shot examples are essential for structured output tasks. They demonstrate the exact format, handling of missing data, and prevent unwanted formatting and behaviors. Choose the approach that best fits your use case.

---

#### Break Down Complex Instructions

When system prompts become too complex, decompose them into smaller, focused sections. For smaller models, this improves readability, reduces ambiguity, and makes it easier to follow instructions consistently.

A complex prompt often mixes role, task, style, output format, and edge cases into one overloaded block. This makes it harder for smaller models to identify what matters most.

#### Example: Before and After Decomposition

**Before:**

```markdown
You are a helpful, detail-oriented, thoughtful, careful, security-aware, maintainability-focused, precision-driven technical assistant for Python code review tasks.

Please carefully and thoroughly analyze the Python code that a user provides, while keeping in mind : 
- bugs
- possible vulnerabilities
- maintainability
- readability
- style consistency
- best practices
- edge cases
- possible refactoring opportunities
- documentation quality
- naming conventions
- architectural concerns
- future extensibility concerns
- general software craftsmanship principles

Be concise, yet detailed oriented. 
* Do not be too detailed unless the detail is relevant
* Do not missing anything important

Put the most important things first, except when background context is helpful, in which case it should be present.
Use examples if useful, and keep the response organized in a summary + details + recommendations. 
Readability counts, and some developers prefer short variable names. 
If there are no issues, make that clear in a confident but balanced way. 
Also consider performance, but do not over-focus on performance unless it matters. 
Try to be practical, insightful, balanced, and user-friendly ✨📌🔍
```

**After:**

```markdown
You are a technical assistant specializing in Python code review.

## Capabilities
- Identify issues
- Explain impact
- Recommend fixes

## Guidelines
- Prioritize the most serious issues first
- Use concise, direct language
- Focus on relevant findings only
- Include examples only when they improve clarity

## Format Requirements
1. Summary
2. Detailed findings
3. Recommendations
```

### Budget Context Deliberately

For SLMs, prompt decomposition is also a context-window management strategy. Small models typically have limited context windows, so each section should justify the tokens it consumes.

A suggested budgeting approach, aligned to the structure above, is:

- **Role Definition + Capabilities** (40%): Identity, task, and core abilities
- **Guidelines** (25%): Behavioral instructions and critical constraints
- **Format Requirements** (15%): Output structure and style expectations
- **Examples** (20%): Few-shot demonstrations when they materially improve accuracy

When space is tight:

- Remove examples before removing core instructions
- Shorten capability and guideline bullets
- Use abbreviated parameter descriptions
- Reference external documentation instead of inline explanations

This budgeting approach helps smaller models:

- Preserve limited context for the active task
- Retain the highest-priority instructions
- Avoid losing critical behavior or format constraints

Note to reader : Link to context management blog post coming soon.

## Tool Definitions: Best Practices

Tool definitions are crucial for enabling LLMs to interact with external systems effectively. Well-designed tool definitions guide the model to use tools correctly and efficiently.

### Tool Schema Structure

The current standard for tool definitions is structured as follows:

```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "Clear description of what the tool does",
    "parameters": {
      "type": "object",
      "properties": {
        "parameter_name": {
          "type": "string",
          "description": "Parameter description"
        }
      },
      "required": ["parameter_name"],
      "additionalProperties": false
    }
  }
}
```

### Naming Conventions

Use **action-oriented verbs** with consistent casing for function names. Choose the convention that matches your ecosystem (snake_case for Python, camelCase for JavaScript, PascalCase for C#):

**Good Examples**

| Examples | Notes |
| -------- | ----- |
| `read_file` | Clear action verb, snake_case format |
| `execute_command` | Specific and descriptive |
| `search_database` | Action-oriented naming |
| `generate_report` | Concise and clear |

**Poor Examples**

| Examples | Notes |
| -------- | ----- |
| `file_reader_tool` | Redundant "tool" suffix |
| `ReadFile` | Not snake_case |
| `file` | Not action-oriented |
| `do_file_stuff` | Vague, not specific |

**Guidelines:**

- Start with a verb that clearly indicates the action
- Use specific, descriptive names
- Keep names concise (2-3 words maximum)
- Avoid redundant suffixes like "_tool" or "_function"

### Description Best Practices

The `description` field is critical for model understanding. Structure it to include:

1. **Clear Purpose Statement**: What the tool does in one concise sentence
2. **Key Capabilities**: What the tool can accomplish (as bullet points if multiple)
3. **Critical Constraints**: Important limitations in **bold**
4. **Usage Guidance**: When to use this tool

**Example:**

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read and return the contents of one or more files from the filesystem. Supports reading text files, extracting text from PDF and DOCX files, and reading specific line ranges for large files. **Maximum 5 files per request.** **Cannot read binary files except PDF/DOCX.** Use when examining source code, reviewing documentation, or gathering context from multiple related files.",
    "parameters": { }
  }
}
```

### Parameter Schema Documentation

Parameters follow JSON Schema specifications. Each parameter should specify:

1. **Type**: Use JSON Schema types (`string`, `number`, `integer`, `boolean`, `array`, `object`)
2. **Description**: Clear explanation of purpose, format, and constraints
3. **Enum**: List allowed values when applicable
4. **Required Status**: Include in `required` array if mandatory
5. **Additional Constraints**: Use `pattern`, `default`, `minimum`, `maximum`, `items`, etc.

**Complete Example:**

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read and return the contents of one or more files from the filesystem. Supports text files, PDFs, and DOCX files. **Maximum 5 files per request.** Use when examining source code or gathering context from multiple files.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "File path relative to workspace directory. Format: 'relative/path/to/file.ext'. Must exist and be readable."
        },
        "line_range": {
          "type": "string",
          "description": "Optional line range to read in format 'start-end' (e.g., '1-100'). Both start and end are inclusive with 1-based indexing. If not provided, reads entire file.",
          "pattern": "^\\d+-\\d+$"
        },
        "encoding": {
          "type": "string",
          "description": "Character encoding for file reading. Standard encoding name such as 'utf-8' or 'ascii'.",
          "enum": ["utf-8", "ascii", "iso-8859-1", "utf-16"],
          "default": "utf-8"
        }
      },
      "required": ["path"],
      "additionalProperties": false
    }
  }
}
```

## Multi-Turn Conversations with SLMs

SLMs can experience context drift in long conversations. Design for this limitation:

### State Management

- Store critical information (IDs, names, parameters) in structured state
- Critical information should be reiterated when context drift is observed
- Explicitly pass previous results to status check tools

### Context Compression

- Summarize long conversations periodically
- Remove redundant turns from history
- Keep only the last N turns plus the initial request

**Example:**

- Weak: "What's the status?" (requires remembering previous context)
- Better: "What's the status of job JOB12345 on system S2?" (self-contained)

## Integration Strategies

### Tool reference integration

When tools are defined using JSON schemas and passed to the model, explicitly referencing those same tools within the system prompt helps the model understand when and how to use them. By naming the tools directly in the instructions and describing their purpose in context, the prompt provides additional guidance that complements the tool definitions.

Structure agent instructions to reference tools explicitly:

```markdown
You are a code analysis assistant with access to file reading and search tools.

## Workflow
1. Use `read_file` to examine source code
2. Use `search_files` to find patterns across the codebase
3. Analyze findings and provide recommendations
4. Use `write_file` to document your analysis

## Guidelines
- Read files before making suggestions
- Search for similar patterns when identifying issues
- Provide specific line references in your analysis
- Document findings in a structured format
```

### Iterative Refinement

Use a systematic approach to improve prompts based on measurable outcomes:

**1. Establish Baseline**

- Run 50-100 test cases with current prompt
- Measure key metrics: accuracy, tool selection rate, argument correctness
- Document failure patterns by analyzing:
  - **Error patterns**: Where does the model consistently struggle?
  - **Misuse cases**: Which tools are used incorrectly?
  - **Response quality**: Are outputs meeting expectations?
  - **Efficiency**: Is the model using tools optimally?

**2. Hypothesis-Driven Changes**

- Change ONE thing at a time
- Document what you changed and why
- Predict expected improvement

**3. Measure Impact**

- Re-run same test cases
- Compare metrics to baseline
- Keep change only if improvement > 5%

**4. Common Refinement Patterns**

- **Tool selection wrong** → Clarify tool descriptions
- **Arguments wrong** → Clarify argument description, add argument examples
- **Format wrong** → Add in few-shot examples, showcasing the desired response format template
- **Context lost** → Add state management or reiterate instructions
- **General confusion** → Add clarifying examples for problematic areas

**Anti-Pattern:** Changing multiple things at once without knowing what worked

#### Task Decomposition Based on Testing

During testing, if you find the model cannot handle various instructions in different forms or struggles with complex multi-step tasks, consider **task decomposition**:

- **Break complex tasks into smaller subtasks**: Instead of one large instruction, create a sequence of simpler, focused instructions
- **Use explicit step-by-step workflows**: For frequently reused and well defined workflows, document those explicitly in the system prompt or through few-shot examples, to guide the model through each phase of a complex operation
- **Separate concerns**: Divide tasks by function (e.g., data gathering, analysis, output generation)
- **Test individual components**: Validate each subtask independently before combining them

**Example of Task Decomposition:**

Instead of:

```markdown
Analyze the codebase, identify security vulnerabilities, suggest fixes, and generate a report.
```

Decompose into:

```markdown
1. Use `search_files` to scan for common security patterns
2. Use `read_file` to examine flagged files in detail
3. Document findings with specific line references
4. Generate recommendations for each vulnerability
5. Use `write_file` to create the final report
```

This approach helps smaller models maintain focus and accuracy throughout complex workflows.

For comprehensive guidance on testing methodologies and iterative improvement strategies, see [`testing_agents.md`](testing_agents.md).

## Conclusion

Effective prompting of small large language models requires precision, clarity, and thoughtful design. By following these principles—concise system prompts with positive framing, well-structured tool definitions with clear constraints, and comprehensive examples—you can build reliable agent systems that leverage smaller models effectively.

Remember:

- **Conciseness is key**: Every token matters with smaller models
- **Positive framing works better**: Tell the model what to do, not what to avoid
- **Structure reduces complexity**: Break down complex instructions into manageable parts
- **Minimal formatting**: Avoid emojis and decorative characters; use markdown only when it adds clarity
- **Context window management**: Budget tokens carefully across task definition, tools, and examples
- **Multi-turn awareness**: Design for context loss; make requests self-contained when possible
- **Clear tool definitions prevent errors**: Invest time in comprehensive documentation
- **Systematic refinement**: Test changes one at a time with measurable metrics
- **Examples guide behavior**: Show the model what success looks like

By applying these best practices, you'll create agent systems that deliver accurate, reliable responses even with resource-constrained models.