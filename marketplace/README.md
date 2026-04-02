# TurboAgent Marketplace

Pre-built, TurboQuant-optimized agent templates for common use cases.

## Templates

| Template | Description | Agents | Use Case |
|----------|-------------|--------|----------|
| `research_swarm` | 3-agent research pipeline | researcher + critic + writer | Academic papers, literature review |
| `code_analyst` | Codebase analysis agent | single agent with RAG | Code review, documentation, refactoring |
| `document_qa` | Document Q&A with RAG | single agent + vector store | Enterprise knowledge base, support |

## Usage

```python
from turboagent.marketplace import load_template

# Load a pre-built template
agent = load_template("research_swarm", model="meta-llama/Llama-3.1-70B-Instruct")
result = agent.run("Analyze recent advances in KV cache compression.")
```

## Contributing Templates

1. Create a JSON config in `marketplace/templates/`
2. Follow the schema in existing templates
3. Submit a PR

## Premium Templates

Additional templates available with TurboAgent Enterprise:
- **Compliance Analyst** — SOC-2/GDPR audit with structured output
- **Data Pipeline Agent** — ETL orchestration with tool calling
- **Customer Support Swarm** — Escalation routing with sentiment analysis

Contact: enterprise@turboagent.to
