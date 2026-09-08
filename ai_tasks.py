"""
ai_tasks.py
-----------
The three AI tasks. This is where LangChain and the prompts live.

Every task is built the same way:

    prompt | llm.with_structured_output(SomePydanticModel)

`with_structured_output` is the important part. It tells OpenAI:
"answer by filling in these exact fields", and LangChain then validates the
answer against the Pydantic model. So we never have to parse free text.
"""

import logging
import time
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from models import CaseSummary, ComplaintData, CustomerEmail

log = logging.getLogger(__name__)

# If an API call fails (network glitch, rate limit), try again this many times.
MAX_ATTEMPTS = 3

# This rule is repeated in all three prompts. It is the single most useful
# instruction for stopping the model from making things up.
NO_MAKING_THINGS_UP = (
    "Use ONLY information that is present in the text you are given. "
    "If something is not mentioned, leave it empty or false. "
    "Never invent names, phone numbers, dates, amounts or promises."
)


@lru_cache(maxsize=1)
def build_llm() -> ChatOpenAI:
    """Create the OpenAI chat model using the settings in config.py.

    lru_cache means the model object is built once and reused for every call.
    """
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "No OpenAI API key found.\n"
            "Create a file called .env next to app.py containing:\n"
            "    OPENAI_API_KEY=sk-your-key-here"
        )
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        temperature=config.TEMPERATURE,
    )


def call_with_retry(chain, variables: dict, task_name: str):
    """Run an LLM chain, retrying a few times if the API call fails.

    Calls over the internet fail occasionally - a dropped connection, a rate
    limit, a momentary server error. Retrying with a growing pause fixes almost
    all of those. If every attempt fails we raise a clear error, and app.py
    catches it so the rest of the batch still runs.
    """
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return chain.invoke(variables)
        except Exception as error:
            last_error = error
            log.warning(
                "%s failed (attempt %d of %d): %s", task_name, attempt, MAX_ATTEMPTS, error
            )
            if attempt < MAX_ATTEMPTS:
                pause = 2**attempt  # wait 2s, then 4s - this is "exponential backoff"
                log.warning("Waiting %d seconds before trying again...", pause)
                time.sleep(pause)

    raise RuntimeError(f"{task_name} failed after {MAX_ATTEMPTS} attempts: {last_error}")


# ---------------------------------------------------------------------------
# TASK 1 - read the document and pull out structured information
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer service analyst. You read one complaint document "
            "and fill in a structured case record.\n\n"
            + NO_MAKING_THINGS_UP
            + "\n\nNotes:\n"
            "- 'is_complaint' is false for a simple enquiry or a compliment.\n"
            "- 'escalation_required' is true if the document mentions escalation, a "
            "manager, legal action, a regulator, or an unresolved repeated problem.\n"
            "- 'supporting_document_available' is true only if evidence such as an "
            "invoice, receipt, photo or screenshot is mentioned.",
        ),
        ("human", "Here is the complaint document:\n\n{document_text}"),
    ]
)


def extract_complaint_data(document_text: str) -> ComplaintData:
    """TASK 1: document text -> structured ComplaintData."""
    chain = EXTRACTION_PROMPT | build_llm().with_structured_output(ComplaintData)
    return call_with_retry(chain, {"document_text": document_text}, "Extraction")


# ---------------------------------------------------------------------------
# TASK 2 - write the reply to the customer
# ---------------------------------------------------------------------------
EMAIL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write replies to customers on behalf of a customer care team.\n\n"
            + NO_MAKING_THINGS_UP
            + "\n\nStyle rules:\n"
            "- Professional, warm and clear. No marketing language, no emojis.\n"
            "- If no resolution is recorded, say the case is under review and give "
            "the next step. Do not promise a refund or a date.\n"
            "- Never leave placeholders like [insert name] in the text.",
        ),
        (
            "human",
            "Write the reply email for this case:\n\n{case_data}",
        ),
    ]
)


def write_customer_email(data: ComplaintData) -> CustomerEmail:
    """TASK 2: structured data -> a customer-facing email."""
    chain = EMAIL_PROMPT | build_llm().with_structured_output(CustomerEmail)
    return call_with_retry(
        chain, {"case_data": data.model_dump_json(indent=2)}, "Email generation"
    )


# ---------------------------------------------------------------------------
# TASK 3 - write the internal summary for the manager
# ---------------------------------------------------------------------------
SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write short internal case notes for a service manager.\n\n"
            + NO_MAKING_THINGS_UP
            + "\n\nStyle rules:\n"
            "- Write for colleagues, not for the customer. Be factual and brief.\n"
            "- 'recommended_next_action' must be one concrete instruction, for "
            "example 'Billing team to confirm the refund reached the card'.",
        ),
        (
            "human",
            "Write the internal case summary for this case:\n\n{case_data}",
        ),
    ]
)


def write_case_summary(data: ComplaintData) -> CaseSummary:
    """TASK 3: structured data -> an internal management summary."""
    chain = SUMMARY_PROMPT | build_llm().with_structured_output(CaseSummary)
    return call_with_retry(
        chain, {"case_data": data.model_dump_json(indent=2)}, "Summary generation"
    )
