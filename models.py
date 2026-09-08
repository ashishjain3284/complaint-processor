"""
models.py
---------
The "shapes" of the data we ask the AI to produce.

This is the most important idea in the project:

    We do NOT ask the LLM for free text and then try to parse it.
    We give it a Pydantic model and it MUST fill in exactly these fields.

Each `description=` below is sent to the model as part of the instruction,
so these classes are half data-structure and half prompt.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# TASK 1 output: the structured information pulled out of the document
# ---------------------------------------------------------------------------
class ComplaintData(BaseModel):
    """All the facts we extract from one complaint document."""

    customer_name: Optional[str] = Field(
        default=None,
        description="Full name of the customer. Null if the document does not say.",
    )
    email: Optional[str] = Field(
        default=None,
        description="Customer email address. Null if the document does not say.",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Customer phone number. Null if the document does not say.",
    )
    product_or_service: Optional[str] = Field(
        default=None,
        description="The product or service the complaint is about.",
    )

    complaint_category: Literal[
        "Billing",
        "Delivery",
        "Product Defect",
        "Service Quality",
        "Technical Support",
        "Account Access",
        "Refund",
        "Warranty",
        "Other",
    ] = Field(
        default="Other",
        description="Pick the single best category for the main issue.",
    )

    issue_description: str = Field(
        description="2-4 sentences describing the problem, using only what the document says.",
    )
    resolution_provided: Optional[str] = Field(
        default=None,
        description="What the company actually did. Null if the document records no resolution.",
    )

    # The three Yes/No flags the assignment asks for.
    is_complaint: bool = Field(
        description="True if the document reports a problem or dissatisfaction.",
    )
    escalation_required: bool = Field(
        description=(
            "True if the document says the case was escalated, or if it is unresolved, "
            "repeated, or involves legal / regulatory / safety risk."
        ),
    )
    supporting_document_available: bool = Field(
        description=(
            "True only if the document mentions evidence such as an invoice, receipt, "
            "photo, screenshot or attachment."
        ),
    )

    case_status: Literal[
        "Open", "In Progress", "Escalated", "Resolved", "Closed", "Unknown"
    ] = Field(
        default="Unknown",
        description="Overall status of the case according to the document.",
    )

    priority: Literal["Low", "Medium", "High"] = Field(
        default="Medium",
        description="How urgent this case is for the business.",
    )


# ---------------------------------------------------------------------------
# TASK 2 output: the email we send back to the customer
# ---------------------------------------------------------------------------
class CustomerEmail(BaseModel):
    """A professional reply to the customer."""

    subject: str = Field(description="A short, specific subject line.")
    greeting: str = Field(
        description="Greeting line. Use 'Dear Customer,' if there is no name."
    )
    body: str = Field(
        description=(
            "3-4 short paragraphs: apologise, restate the issue, give the resolution "
            "or current status, and invite the customer to reply. 150-250 words. "
            "Do not invent refunds, dates or promises that are not in the case data."
        ),
    )

    def as_text(self) -> str:
        """Turn the email into plain text we can save to a .txt file."""
        return (
            f"Subject: {self.subject}\n\n"
            f"{self.greeting}\n\n"
            f"{self.body}\n\n"
            f"Kind regards,\n"
            f"Customer Care Team\n"
        )


# ---------------------------------------------------------------------------
# TASK 3 output: the internal note for the manager
# ---------------------------------------------------------------------------
class CaseSummary(BaseModel):
    """A short internal summary for the management team."""

    case_overview: str = Field(description="2-3 sentences describing the case.")
    key_issue: str = Field(description="The single most important problem, in one sentence.")
    action_taken: str = Field(
        description="What has been done so far. Use 'No action recorded' if nothing."
    )
    current_status: str = Field(description="Where the case stands right now.")
    recommended_next_action: str = Field(
        description="The one next step somebody should take, written as an instruction."
    )

    def as_text(self) -> str:
        """Turn the summary into readable text we can save to a .txt file."""
        return (
            "INTERNAL CASE SUMMARY\n"
            "=====================\n\n"
            f"Case overview          : {self.case_overview}\n\n"
            f"Key issue              : {self.key_issue}\n\n"
            f"Action taken           : {self.action_taken}\n\n"
            f"Current status         : {self.current_status}\n\n"
            f"Recommended next action: {self.recommended_next_action}\n"
        )
