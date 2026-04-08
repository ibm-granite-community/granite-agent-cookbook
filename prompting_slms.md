# Prompting with SLM: Best Practices for Accurate Agent Responses

## Introduction

When working with "small" large language models (SLMs) in agent systems and applications, effective prompting becomes critical. Unlike their larger counterparts, smaller models have limited capacity and require more precise, well-structured prompts to deliver accurate and reliable responses. This guide explores best practices for crafting system prompts and tool definitions that maximize the performance of smaller LLMs within agent frameworks.

## System Prompt Design Principles

### Keep It Concise and Clear

Smaller models have limited context windows and processing capacity. Every token in your system prompt matters. Focus on:

- **Direct instructions**: State what the model should accomplish
- **Essential context only**: Include only information directly relevant to the task
- **Simple language**: Use straightforward vocabulary and sentence structures
- **Logical organization**: Present information in a clear, sequential manner
- **Plain text only**: Keep prompts free of emojis and special characters that add no semantic value

### Prioritize Task Oriented Guidelines

Frame instructions using positive action words rather than negative prohibitions. 
This approach is more effective because:

- Positive framing tells the model **what to do** rather than what to avoid
- It reduces ambiguity and cognitive load on the model
- It provides clear direction for desired behavior

Just like telling someone "Don't look behind you" is the surest way to have them look behind them, asking a model to "not do something" might end up introducing the unwanted behavior.

**Example of Negative vs. Positive Framing:**

| Avoid (Negative Framing) | Prefer (Positive Framing) |
|------------------------------|------------------------------|
| "Do not provide verbose explanations. Do not include unnecessary details." | "Provide concise explanations. Focus on essential details." |
| "Don't forget to validate input parameters." | "Validate all input parameters before processing." |
| "Never use deprecated functions in your code." | "Use current, supported functions in your code." |
| "Don't make assumptions about user requirements." | "Clarify user requirements before proceeding." |
| "Avoid writing code without proper error handling." | "Include comprehensive error handling in all code." |


### Break Down Complex Instructions

When system prompts become too complex, decompose them into smaller, focused sections:

```markdown
## Role Definition
You are a technical assistant specializing in code analysis.

## Core Capabilities
- Analyze code structure and patterns
- Identify potential issues
- Suggest improvements

## Response Format
Provide responses in three parts:
1. Summary of findings
2. Detailed analysis
3. Actionable recommendations
```

This modular approach helps smaller models:
- Process instructions more effectively
- Maintain focus on specific aspects
- Reduce confusion from information overload

### Structure for Success

Organize your system prompt with clear sections:

1. **Role Definition**: Establish the model's identity and purpose
2. **Capabilities**: List what the model can do
3. **Guidelines**: Provide behavioral instructions using positive framing
4. **Format Requirements**: Specify output structure expectations
5. **Examples**: Include 1-2 concrete examples when helpful

---

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

Use **action-oriented verbs in snake_case** for function names:

**Good Examples**
| Examples | Notes |
|----------|-------|
| `read_file` | Clear action verb, snake_case format |
| `execute_command` | Specific and descriptive |
| `search_database` | Action-oriented naming |
| `generate_report` | Concise and clear |

**Poor Examples**
| Examples | Notes |
|----------|-------|
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

**Description Guidelines:**
- Keep the first sentence direct and action-focused
- Use bold (**text**) for critical constraints that prevent errors
- Include when-to-use guidance to help the model choose appropriate tools
- Aim for 2-4 sentences maximum for clarity

### Parameter Schema Documentation

Parameters follow JSON Schema specifications. Each parameter should specify:

1. **Type**: Use JSON Schema types (`string`, `number`, `integer`, `boolean`, `array`, `object`)
2. **Description**: Clear explanation of purpose, format, and constraints
3. **Enum**: List allowed values when applicable
4. **Required Status**: Include in `required` array if mandatory
5. **Additional Constraints**: Use `pattern`, `minimum`, `maximum`, `items`, etc.

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

## Integration Strategies

### Combining System Prompts with Tool Definitions

Structure your agent's instructions to reference tools naturally:

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

Monitor your model's performance and refine prompts based on:

1. **Error patterns**: Where does the model consistently struggle?
2. **Misuse cases**: Which tools are used incorrectly?
3. **Response quality**: Are outputs meeting expectations?
4. **Efficiency**: Is the model using tools optimally?

Adjust your prompts by:
- Adding clarifying examples for problematic areas
- Emphasizing frequently missed constraints
- Simplifying overly complex instructions
- Adding positive guidance for common mistakes

## Conclusion

Effective prompting of small large language models requires precision, clarity, and thoughtful design. By following these principles—concise system prompts with positive framing, well-structured tool definitions with clear constraints, and comprehensive examples—you can build reliable agent systems that leverage smaller models effectively.

Remember:
- **Conciseness is key**: Every token matters with smaller models
- **Positive framing works better**: Tell the model what to do, not what to avoid
- **Structure reduces complexity**: Break down complex instructions into manageable parts
- **Plain text only**: Keep prompts free of emojis and unnecessary special characters
- **Clear tool definitions prevent errors**: Invest time in comprehensive documentation
- **Examples guide behavior**: Show the model what success looks like

By applying these best practices, you'll create agent systems that deliver accurate, reliable responses even with resource-constrained models.