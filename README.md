# Document Assistant — Design, Implementation and Examples

## Project summary

This assistant processes user requests using a small multi-agent graph (LangGraph + LangChain). It classifies intent, routes the request to the appropriate agent (QA / Summarization / Calculation), uses tools (Document Retriever & Calculator), enforces **structured outputs** using Pydantic schemas, and updates persistent state/memory between turns.

Key goals:

* Reliable intent routing
* Typed, machine-parseable responses (structured outputs)
* Clear state & memory so conversation can be resumed
* Auditability (tools used, sources, timestamps)

---

## Design overview (agent graph)

```
User Input
   ↓
classify_intent  (LLM -> returns UserIntent)
   ↓ next_step -> "qa_agent" | "summarization_agent" | "calculation_agent"
qa_agent / summarization_agent / calculation_agent
   ↓
update_memory
   ↓
END
```

* `classify_intent` is always the entry point. It returns a `UserIntent` (see schema below) and sets `next_step`.
* Each agent (qa/summarization/calculation) uses structured-output schemas for its response and may call tools:

  * Document Retriever (reads source docs; returns doc ids & snippets)
  * Calculator (performs numerical computations; used by calculation agent)
* After an agent completes, `update_memory` summarizes the turn and updates `AgentState`.

---

## Implementation decisions & rationale

1. **Structured outputs** — enforce LLM responses using Pydantic schemas via `llm.with_structured_output(Schema)`.

   * Reason: avoids brittle string parsing; outputs are validated and typed.
2. **Small, focused agents** — each agent is responsible for one class of tasks (QA, summarization, calculation).

   * Reason: easier prompt tuning, tools mapping and testing.
3. **Tools as first-class functions** — `@tool` decorated functions (LangChain style) with safe validation.

   * Calculator validates expressions to avoid code injection and only allows basic arithmetic.
4. **State and persistence** — use an `AgentState` model and `InMemorySaver` checkpointer (or replaceable persistence layer).

   * Reason: Allows resuming conversations and testing state transitions.
5. **Auditable actions** — state keeps `tools_used`, `actions_taken`, `active_documents`, timestamps for traceability.

---

## Key schemas

### `UserIntent` (intent classification)

```py
class UserIntent(BaseModel):
    intent_type: Literal["qa", "summarization", "calculation", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

### `AnswerResponse` (Q&A response)

```py
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
```

(You will also have schemas for `SummarizationResponse` and `CalculationResponse` — follow the same pattern: typed fields, `sources`, `confidence`, `timestamp`.)

### `AgentState` (important fields)

```py
class AgentState(BaseModel):
    user_input: str
    messages: List[MessageAnnotation]
    intent: Optional[UserIntent] = None
    next_step: Optional[str] = None
    conversation_summary: Optional[str] = None
    active_documents: List[str] = []
    current_response: Optional[Dict] = None   # structured result placed here
    tools_used: List[str] = []
    actions_taken: List[str] = []              # reducer: operator.add
    session_id: str
    user_id: Optional[str]
```

* `actions_taken` uses an `operator.add` reducer in the workflow to accumulate nodes executed each turn (helps auditing).
* `current_response` stores the validated structured output of the last executed agent.

---

## How structured outputs are enforced (practical notes)

* Use the LLM helper to **wrap** responses with the Pydantic schema:

  ```py
  llm_structured = llm.with_structured_output(UserIntent)
  out = llm_structured.generate(prompt)  # library-specific API
  user_intent: UserIntent = out.value
  ```
* If the LLM returns invalid data, the `.with_structured_output()` wrapper will raise/return validation errors; handle them by:

  * re-prompting (fallback)
  * use default: `intent_type="unknown"`
* For agent nodes, do the same with `AnswerResponse` / `SummarizationResponse` / `CalculationResponse`.

---

## Prompts & how they map to agents

* `get_intent_classification_prompt()` — used by `classify_intent`. Inputs: `user_input`, `conversation_history`. Output schema: `UserIntent`.
* `QA_SYSTEM_PROMPT`, `SUMMARIZATION_SYSTEM_PROMPT`, `CALCULATION_SYSTEM_PROMPT` — used in `get_chat_prompt_template(intent_type, ...)` which selects the system context for the agent node.

**Important**: `CALCULATION_SYSTEM_PROMPT` must instruct the model to *always* use the `calculator` tool for arithmetic and to indicate which document(s) it needs.

---

## Tools

### Document Retriever

* API: `retrieve_documents(query) -> List[Document]`
* Returns document id, title, and local snippet.
* Populates `active_documents` in state.

### Calculator tool (safe)

* Decorated with `@tool`.
* Validates input expression using a whitelist regex (digits, spaces, `+ - * / ( ) .`).
* Uses `eval()` only after validation, inside a restricted namespace.
* Logs tool use (ToolLogger) and returns a formatted string.

Example (pseudocode):

```py
@tool("calculator")
def calculator_tool(expression: str) -> str:
    if not SAFE_RE.match(expression):
        return "Invalid expression"
    result = eval(expression, {"__builtins__": {}})
    ToolLogger.log("calculator", expression)
    return f"Result: {result}"
```

---

## State & memory flow (detailed)

1. **Start**: `process_message` creates or loads `AgentState` for `session_id`.
2. **classify_intent**:

   * Compose prompt with `user_input` + `messages`.
   * Call `llm.with_structured_output(UserIntent)` and parse response.
   * Populate `state.intent`, choose `next_step`:

     * `qa` → `"qa_agent"`
     * `summarization` → `"summarization_agent"`
     * `calculation` → `"calculation_agent"`
     * default → `"qa_agent"`
   * `actions_taken += ["classify_intent"]`
3. **Agent Node (qa/summarization/calculation)**:

   * Construct prompt using `get_chat_prompt_template(intent_type, ...)`.
   * If needed, call `retrieve_documents()` to get sources; add to `state.active_documents` and `tools_used`.
   * If `calculation_agent`:

     * the LLM must decide the expression and call `calculator` tool; we enforce calculator usage in the system prompt.
     * record `tools_used += ["calculator"]`.
   * Receive structured response and set `state.current_response`.
   * `actions_taken += ["qa_agent"]` (or the relevant agent name).
4. **update_memory**:

   * Summarize the turn (LLM or local summarizer).
   * Update `conversation_summary`, `messages`, `active_documents`.
   * Persist `AgentState` via the chosen checkpointer.
5. **Persist**: the workflow compiled with `InMemorySaver()` or other saver persists the `AgentState` across invocations.

---

## Session persistence & logging

* **Persistence**:

  * Default: `InMemorySaver()` — useful for tests and demos.
  * For production, swap the saver for a file/db-backed saver (Redis, DynamoDB, or custom).
  * Sessions saved under `sessions/` (or configured storage) by `session_id`. Example file: `sessions/<session_id>.json` (if using file saver).
* **Logging**:

  * Tools log entries via `ToolLogger` with fields: `timestamp`, `tool_name`, `input`, `output`.
  * Application logs (info/debug) stored in `logs/assistant.log`.
  * Example log contents:

    ```
    2025-12-05T10:12:23Z INFO classify_intent session=abc123 prompt=... 
    2025-12-05T10:12:24Z INFO qa_agent session=abc123 retrieved_docs=['doc_42','doc_9']
    2025-12-05T10:12:25Z INFO tool:calculator session=abc123 input='10 / 2' output='5.0'
    ```
* **How to inspect previous sessions**:

  * Read `sessions/<session_id>.json` or call `assistant.get_session(session_id)` (helper).
  * Logs are plain text rotated monthly; look in `logs/`.

---

## Example conversations

> NOTE: These examples include the structured outputs the agents produce. Timestamps and session ids are illustrative.

### 1) QA flow (intent: `qa`)

**User**: "In the March financial statement, what is the total operating expense for product line A?"

**Flow**:

1. `classify_intent` → returns:

   ```json
   {
     "intent_type":"qa",
     "confidence":0.92,
     "reasoning":"User asks a direct fact question about a document (financial statement)."
   }
   ```
2. `qa_agent`:

   * Document retriever called with query "March financial statement product line A operating expense".
   * Tools used: `doc_retriever` → found `["doc_fin_2025_march"]`
   * Prompt enforces `AnswerResponse` structured output.
   * LLM returns `AnswerResponse`:

     ```json
     {
       "question":"In the March financial statement, what is the total operating expense for product line A?",
       "answer":"The total operating expense for product line A in March is $1,238,450.",
       "sources":["doc_fin_2025_march"],
       "confidence":0.86,
       "timestamp":"2025-12-05T10:20:32Z"
     }
     ```
3. `update_memory` summarizes the turn and records `active_documents=["doc_fin_2025_march"]`.

### 2) Summarization flow (intent: `summarization`)

**User**: "Summarize the main risks discussed in the uploaded compliance report."

**Flow**:

* `classify_intent` → `{"intent_type":"summarization", ...}`
* `summarization_agent`:

  * retriever finds `doc_compliance_v2`.
  * LLM enforces `SummarizationResponse` schema, returns:

    ```json
    {
      "summary":"The report highlights three major risks: 1) Third-party vendor compliance gaps; 2) Data retention policy inconsistencies; 3) Insufficient access controls for exporting data.",
      "key_points":[
        "Vendor compliance checks missing in 12% of cases",
        "Retention policy not unified across subsidiaries",
        "Export permissions enabled for legacy accounts"
      ],
      "sources":["doc_compliance_v2"],
      "confidence":0.81,
      "timestamp":"2025-12-05T10:25:01Z"
    }
    ```
* `update_memory` appends the summary to `conversation_summary`.

### 3) Calculation flow (intent: `calculation`)

**User**: "From the spreadsheet, what's the net margin for product B if revenue is 4,200 and costs equal fixed 1200 plus 12% of revenue?"

**Flow**:

* `classify_intent` → `{"intent_type":"calculation", ...}`
* `calculation_agent`:

  * Document retriever optionally fetches numeric parameters from `doc_spreadsheet_productB`.
  * System prompt forces: "use the calculator tool for all math".
  * LLM decides expression: `net_margin = revenue - (fixed_cost + 0.12 * revenue)` → expression `4200 - (1200 + 0.12*4200)`.
  * Calls `calculator` tool with expression `4200 - (1200 + 0.12*4200)`.
  * Calculator returns `Result: 4200 - (1200 + 504) = 2496`.
  * `CalculationResponse`:

    ```json
    {
      "formula":"net_margin = revenue - (fixed_cost + 0.12 * revenue)",
      "expression":"4200 - (1200 + 0.12*4200)",
      "result":2496,
      "sources":["doc_spreadsheet_productB"],
      "confidence":0.9,
      "timestamp":"2025-12-05T10:30:10Z"
    }
    ```
* `update_memory` stores the numeric result and the fact that `calculator` was used.

---

## Example code snippets

### classify_intent (illustrative)

```py
def classify_intent(state: AgentState, config) -> AgentState:
    llm = config.llm
    # wrap with schema enforcement
    llm_struct = llm.with_structured_output(UserIntent)

    prompt = get_intent_classification_prompt().format(
        user_input=state.user_input,
        conversation_history="\n".join([m.text for m in state.messages])
    )

    response = llm_struct.generate(prompt)
    intent: UserIntent = response.value

    state.intent = intent
    state.actions_taken = state.actions_taken + ["classify_intent"]
    state.next_step = {
        "qa":"qa_agent",
        "summarization":"summarization_agent",
        "calculation":"calculation_agent"
    }.get(intent.intent_type, "qa_agent")
    return state
```

### Using the calculator tool (illustrative)

```py
@tool("calculator")
def calculator_tool(expression: str) -> str:
    # allow only digits, whitespace, parentheses, ., and + - * /
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expression):
        raise ValueError("Unsafe expression")
    result = eval(expression, {"__builtins__": {}})
    ToolLogger.log(tool="calculator", input=expression, output=str(result))
    return f"Result: {result}"
```

---

## Testing & example scripts

* **Integration**:

  * Run `python main.py` and try the three examples above. Inspect `sessions/<session_id>.json` to verify `actions_taken`, `current_response`, `conversation_summary`, and `active_documents`.
* **Manual checks**:

  * Inspect `logs/assistant.log` and `logs/tools.log` for full traceability.

---

## Deployment & config notes

* `.env` must contain `OPENAI_API_KEY` (or the appropriate LLM provider credentials).
* Replace `InMemorySaver` with a persistent saver for production (Redis or DB).
* Consider rate-limiting or batching prompts for cost control.
* For production, add stricter input sanitation (especially for any user input passed into `eval()`-like constructs).

