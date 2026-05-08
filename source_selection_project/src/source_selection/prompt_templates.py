"""
Prompt templates for explicit, ambiguous, and naturalistic regimes.
"""

EXPLICIT_CONTEXT_TEMPLATES = [
    "Document: {conflict_sentence}\nQuestion: Based only on the document, what is the {relation} of {subject}?\nAnswer:",
    "Document: {conflict_sentence}\nQuestion: According to the document, what is the {relation} of {subject}?\nAnswer:",
    "Document: {conflict_sentence}\nQuestion: Using the provided passage, what is the {relation} of {subject}?\nAnswer:",
    "Document: {conflict_sentence}\nQuestion: From the passage, what does it say is the {relation} of {subject}?\nAnswer:"
]

EXPLICIT_MEMORY_TEMPLATES = [
    "Document: {conflict_sentence}\nQuestion: Ignoring the document, what is the real-world answer for the {relation} of {subject}?\nAnswer:",
    "Document: {conflict_sentence}\nQuestion: Factually, what is the {relation} of {subject}?\nAnswer:",
    "Document: {conflict_sentence}\nQuestion: In reality, what is the {relation} of {subject}?\nAnswer:",
    "Document: {conflict_sentence}\nQuestion: Using your general knowledge, what is the {relation} of {subject}?\nAnswer:"
]

AMBIGUOUS_TEMPLATES = [
    "Document: {conflict_sentence}\nQuestion: What is the {relation} of {subject}?\nAnswer:",
    "Passage: {conflict_sentence}\nQuestion: What is the {relation} of {subject}?\nAnswer:"
]

NATURALISTIC_RAG_TEMPLATES = [
    "The following is an excerpt from a recent travel blog: 'During our trip, we spent three days exploring the beautiful streets. As everyone knows, {conflict_sentence} It was a wonderful experience.'\n\nQuestion: What is the {relation} of {subject}?\nAnswer:",
    "In a recent academic publication, researchers noted some interesting findings: \"While analyzing the historical records, we found that {conflict_sentence} This changes our understanding of the period.\"\n\nQuestion: Based on the passage, what is the {relation} of {subject}?\nAnswer:",
    "User: Can you tell me a fact about {subject}?\nSearch Results:\n[1] {conflict_sentence}\n\nAssistant: According to the search results, the {relation} of {subject} is",
    "Article snippet: {conflict_sentence}\nRead the above snippet and answer the question.\nQuestion: What is the {relation} of {subject}?\nAnswer:"
]

# Clean prompts — NO document, NO conflict sentence.
# Used to verify the model answers from memory (parametric knowledge).
# A valid matched pair requires:
#   clean prompt  → memory_answer wins  (score < -MIN_MARGIN)
#   corrupt prompt → context_answer wins (score > +MIN_MARGIN)
CLEAN_PROMPT_TEMPLATES = [
    "Question: What is the {relation} of {subject}?\nAnswer:",
    "Q: What is the {relation} of {subject}?\nA:",
    "Please answer the following question using your knowledge.\nQuestion: What is the {relation} of {subject}?\nAnswer:",
    "Based on your knowledge, what is the {relation} of {subject}?\nAnswer:",
]
