"""GraphRAG core, vendored from github.com/sairambommagani/GraphRAG (commit b65e84c).

Only the pipeline stages the proctoring knowledge graph needs are included:
extraction interface, graph construction + entity resolution, Louvain
community detection with a 2-level hierarchy, extractive community
summaries, and local/global search with IDF reranking. Imports were changed
from `app.` to `graphrag_core.`, and community/detect.py falls back to
NetworkX's built-in Louvain when python-louvain isn't installed (one less
dependency to build). The code is otherwise unchanged, so fixes can be synced
both ways. The spaCy and Groq-specific modules are not needed here:
proctor/knowledge.py supplies a domain extractor for exam rules and incidents.
"""
