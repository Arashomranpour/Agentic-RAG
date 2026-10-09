"""Prompts. The leading [TASK:...] tag is ignored by real models and lets the
deterministic fake model used in tests recognise which step it is playing."""

CONDENSE = """[TASK:CONDENSE]
Rewrite the user's latest question as a standalone question that can be searched on its own.
Only replace vague references ("it", "that project", "her", "the team") with the subject named in
the earlier user questions. Do NOT add any new details, and keep proper names exactly.
If the latest question is already standalone, return it unchanged. Return ONLY the question."""

ROUTE = """[TASK:ROUTE]
Decide how to answer the question. Reply with exactly one word:
- retrieve: it could plausibly be answered by the user's uploaded documents
- web: it clearly needs current / real-world information (news, prices, recent events)
- direct: ONLY greetings, thanks, or general knowledge unrelated to the documents

The user has uploaded these documents: {documents}
Names, projects, people or topics that might appear in them mean retrieve.
When in doubt, choose retrieve."""

GRADE = """[TASK:GRADE]
Do the passages contain information that helps answer the question?
Reply with exactly one word: yes or no."""

REWRITE = """[TASK:REWRITE]
The search found nothing useful. Rewrite the question as a better search query
(different keywords, more specific). Return ONLY the query."""

ANSWER = """[TASK:ANSWER]
You are a helpful assistant that answers questions about the user's documents.
{context_instructions}"""

WITH_CONTEXT = """Answer ONLY from the context below. Cite sources with their number, e.g. [1].
If the context does not contain the answer, say you could not find it in the sources.

Context:
{context}"""

NO_CONTEXT = """No documents or web results were available for this question.
Answer from general knowledge if you reliably can, and say when you are unsure.
If the question is about the user's documents, tell them nothing relevant was found."""
