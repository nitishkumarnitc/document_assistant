from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template.
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an intent classifier for a document processing assistant.

Given the user input and conversation history, classify the user's intent into one of these categories:
- qa: Questions about documents or records that do not require calculations.
- summarization: Requests to summarize or extract key points from documents that do not require calculations.
- calculation: Mathematical operations or numerical computations. Or questions about documents that may require calculations
- unknown: Cannot determine the intent clearly

User Input: {user_input}

Recent Conversation History:
{conversation_history}

Analyze the user's request and classify their intent with a confidence score and brief reasoning.
"""
    )


# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone

"""

# Summarization System Prompt
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information
"""

CALCULATION_SYSTEM_PROMPT = """
You are the Calculation Agent. Your job is to interpret the user's query, 
retrieve any required documents, identify the correct mathematical expression, 
and perform the calculation using the available tools.

Your required steps:

1. **Determine which document is needed**  
   - If the user references a document ID (e.g., INV-001, REC-202), you MUST call the `document_reader` tool to retrieve it.
   - Use the document contents to extract numbers or values needed for the calculation.

2. **Identify the mathematical expression**  
   - Based on the user input and any retrieved document data, formulate a clear arithmetic expression.
   - Examples: “5200 - 300”, “(subtotal + tax) * 0.10”, “642 / 3”, etc.

3. **Perform ALL calculations using the `calculator` tool**  
   - You MUST never compute math yourself.
   - Even simple operations like “2 + 2” must be delegated to the calculator tool.
   - You should output a `calculator` tool call after you form the expression.

4. **Use REACT style reasoning**  
   - Think step-by-step about what document is required, what needs to be calculated, and which tool to call.
   - After thinking, call the appropriate tool(s) in correct order.

5. **Final Output**  
   - Your final response MUST conform to the `CalculationResponse` structured schema.

Never perform arithmetic directly.  
Never guess document contents.  
Always use tools for retrieval and calculation.
"""



# TODO: Finish the function to return the correct prompt based on intent type
# Refer to README.md Task 3.1 for details
def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on the intent type.
    """
    if intent_type == "qa":
        system_prompt = QA_SYSTEM_PROMPT
    elif intent_type == "summarization":
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT
    elif intent_type == "calculation":
        system_prompt = CALCULATION_SYSTEM_PROMPT
    else:
        # default fallback
        system_prompt = QA_SYSTEM_PROMPT

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])



# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = """Summarize the following conversation history into a concise summary:

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions
"""
