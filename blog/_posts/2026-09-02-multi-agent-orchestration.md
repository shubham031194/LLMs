---
layout: post
title: "Multi-Agent Orchestration: Designing Collaborative AI Workflows"
date: 2026-09-02
tags: [agents, multi-agent, orchestration, llm, architecture]
read_time: 9
---

Single-agent systems hit a ceiling. One LLM trying to research, write code, review it, and handle edge cases simultaneously leads to context dilution and role confusion. The solution is the same one that works for human teams — specialize and coordinate.

Multi-agent orchestration splits complex tasks across specialized agents that collaborate through a structured protocol.

## The Core Pattern

```
                    ┌─────────────┐
                    │ Orchestrator │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Researcher│ │  Coder   │ │ Reviewer  │
        └──────────┘ └──────────┘ └──────────┘
```

Each agent has a focused system prompt, specific tools, and a narrow responsibility. The orchestrator routes tasks and manages the workflow.

## Agent Definitions

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    tools: list[dict] = field(default_factory=list)
    model: str = "llama3-8b"
    temperature: float = 0.3

researcher = Agent(
    name="researcher",
    role="Information gathering and analysis",
    system_prompt="""You are a research specialist. Your job is to:
- Search for relevant information using available tools
- Summarize findings clearly and concisely
- Flag conflicting information or gaps
- Never make claims without supporting evidence

Return your findings as structured JSON with 'findings', 'sources', 
and 'confidence' fields.""",
    tools=[search_tool, web_fetch_tool]
)

coder = Agent(
    name="coder",
    role="Code generation and implementation",
    system_prompt="""You are a senior software engineer. Your job is to:
- Write clean, well-documented code based on specifications
- Follow best practices for the target language
- Include error handling and edge cases
- Write unit tests for critical functions

Return code in clearly labeled code blocks with the filename.""",
    tools=[file_write_tool, execute_tool]
)

reviewer = Agent(
    name="reviewer",
    role="Code review and quality assurance",
    system_prompt="""You are a code reviewer. Your job is to:
- Check for bugs, security issues, and performance problems
- Verify the code meets the original requirements
- Suggest concrete improvements with examples
- Rate confidence in the code's correctness (1-10)

Return a structured review with 'issues', 'suggestions', 
and 'approval' fields.""",
    tools=[file_read_tool, execute_tool]
)
```

## The Orchestrator

The orchestrator is itself an LLM that plans the workflow:

```python
class Orchestrator:
    def __init__(self, agents: dict[str, Agent], llm):
        self.agents = agents
        self.llm = llm
        self.conversation_log = []
    
    async def execute(self, task: str) -> dict:
        # Step 1: Plan the workflow
        plan = await self.create_plan(task)
        
        results = {}
        
        for step in plan["steps"]:
            agent = self.agents[step["agent"]]
            
            # Build context from previous steps
            context = self.build_context(step, results)
            
            # Execute the step
            result = await self.run_agent(
                agent=agent,
                instruction=step["instruction"],
                context=context
            )
            
            results[step["id"]] = result
            self.conversation_log.append({
                "step": step["id"],
                "agent": agent.name,
                "instruction": step["instruction"],
                "result": result
            })
            
            # Check if we need to revise the plan
            if result.get("needs_revision"):
                plan = await self.revise_plan(plan, step, result)
        
        return self.compile_output(results)
    
    async def create_plan(self, task: str) -> dict:
        plan_prompt = f"""Break this task into steps. For each step, specify:
- id: unique step identifier
- agent: which agent handles it ({', '.join(self.agents.keys())})
- instruction: what the agent should do
- depends_on: list of step IDs this depends on

Task: {task}

Return as JSON with a 'steps' array."""

        response = await self.llm.generate(plan_prompt)
        return json.loads(response)
    
    async def run_agent(self, agent: Agent, instruction: str, context: str) -> dict:
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{instruction}"}
        ]
        
        response = await self.llm.generate(
            messages=messages,
            tools=agent.tools,
            model=agent.model,
            temperature=agent.temperature
        )
        
        return parse_agent_response(response)
```

## Handling Agent Communication

Agents don't talk to each other directly — everything flows through the orchestrator. This is intentional. Direct agent-to-agent communication creates feedback loops that are hard to debug.

```python
def build_context(self, current_step: dict, results: dict) -> str:
    context_parts = []
    
    for dep_id in current_step.get("depends_on", []):
        if dep_id in results:
            dep_result = results[dep_id]
            context_parts.append(
                f"=== Output from {dep_id} ===\n{json.dumps(dep_result, indent=2)}"
            )
    
    return "\n\n".join(context_parts)
```

## The Review Loop

The most valuable pattern is the coder-reviewer loop:

```python
async def code_with_review(self, spec: str, max_revisions: int = 3) -> dict:
    code = await self.run_agent(self.agents["coder"], spec, "")
    
    for revision in range(max_revisions):
        review = await self.run_agent(
            self.agents["reviewer"],
            "Review this code against the spec",
            f"Spec:\n{spec}\n\nCode:\n{code}"
        )
        
        if review["approval"]:
            return {"code": code, "review": review, "revisions": revision}
        
        # Send review feedback back to coder
        code = await self.run_agent(
            self.agents["coder"],
            "Revise the code based on this review",
            f"Original spec:\n{spec}\n\nYour code:\n{code}\n\nReview:\n{json.dumps(review)}"
        )
    
    return {"code": code, "review": review, "revisions": max_revisions, "warning": "max revisions reached"}
```

## Practical Guardrails

Multi-agent systems can spiral without controls:

**Token budgets per agent.** Each agent gets a maximum token allocation. If the researcher is burning through tokens without converging, it gets cut off and forced to summarize what it has.

**Step limits.** The orchestrator has a maximum of 10-15 steps. Plans that require more get simplified.

**Deadlock detection.** If two agents keep sending work back and forth (coder and reviewer disagreeing), the orchestrator intervenes after 3 cycles.

```python
def detect_deadlock(self, log: list[dict]) -> bool:
    if len(log) < 4:
        return False
    
    recent = log[-4:]
    agents = [entry["agent"] for entry in recent]
    
    # A-B-A-B pattern = potential deadlock
    if agents[0] == agents[2] and agents[1] == agents[3] and agents[0] != agents[1]:
        return True
    return False
```

## When Multi-Agent Makes Sense

Not every task needs this complexity. Single-agent is fine for straightforward Q&A, simple code generation, or single-step tasks. Multi-agent earns its overhead when:

- The task has distinct phases (research → implement → review)
- Different phases need different tools or model configurations
- Quality requires adversarial checking (coder vs reviewer)
- The task scope exceeds what one context window can handle

Start with single-agent, measure where it fails, and introduce specialization only where the failure patterns demand it. The best multi-agent system is the simplest one that achieves your quality bar.
