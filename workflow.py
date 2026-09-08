"""
workflow.py
-----------
This connects the three AI tasks together using LangGraph.

The picture:

        START
          |
          v
      [ extract ]                 Task 1 - must run first
          |
      +---+---+                   the next two do NOT depend on each other,
      |       |                   so LangGraph runs them AT THE SAME TIME
      v       v
  [ email ] [ summary ]           Task 2 and Task 3
      |       |
      +---+---+
          |
          v
         END

Why bother with a graph instead of calling three functions in a row?

  1. The email and the summary both only need the extracted data, so they can
     run in parallel. That makes each document roughly 40% faster.
  2. The steps are separate, named and easy to change. You can add a fourth
     task later without touching the other three.
"""

from functools import lru_cache
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import ai_tasks
from models import CaseSummary, ComplaintData, CustomerEmail


class State(TypedDict):
    """The shared box of data that gets passed from step to step.

    Each step reads what it needs and returns only the key it produced.
    """

    document_text: str
    complaint_data: Optional[ComplaintData]
    customer_email: Optional[CustomerEmail]
    case_summary: Optional[CaseSummary]


# ---------------------------------------------------------------------------
# The three steps ("nodes" in LangGraph language)
# ---------------------------------------------------------------------------
def extract_step(state: State) -> dict:
    """Step 1: read the document and pull out the structured information."""
    data = ai_tasks.extract_complaint_data(state["document_text"])
    return {"complaint_data": data}


def email_step(state: State) -> dict:
    """Step 2: write the customer reply (runs in parallel with step 3)."""
    email = ai_tasks.write_customer_email(state["complaint_data"])
    return {"customer_email": email}


def summary_step(state: State) -> dict:
    """Step 3: write the internal summary (runs in parallel with step 2)."""
    summary = ai_tasks.write_case_summary(state["complaint_data"])
    return {"case_summary": summary}


# ---------------------------------------------------------------------------
# Wiring the steps together
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def build_workflow():
    """Build and compile the LangGraph workflow shown in the diagram above.

    lru_cache means the graph is built once and reused for every document.
    """
    graph = StateGraph(State)

    # Register the three steps.
    graph.add_node("extract", extract_step)
    graph.add_node("email", email_step)
    graph.add_node("summary", summary_step)

    # Connect them.
    graph.add_edge(START, "extract")

    # Two arrows out of "extract" = these two steps run at the same time.
    graph.add_edge("extract", "email")
    graph.add_edge("extract", "summary")

    # Both must finish before the workflow ends.
    graph.add_edge("email", END)
    graph.add_edge("summary", END)

    return graph.compile()


def process_document(document_text: str) -> State:
    """Run the whole workflow for one document and return the finished state."""
    workflow = build_workflow()
    starting_state: State = {
        "document_text": document_text,
        "complaint_data": None,
        "customer_email": None,
        "case_summary": None,
    }
    return workflow.invoke(starting_state)
