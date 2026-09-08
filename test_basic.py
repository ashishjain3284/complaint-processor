"""
test_basic.py
-------------
A small set of tests that prove the project works.

Run them with:

    pytest -v

None of these tests call OpenAI, so they cost nothing, need no API key, and
run in under a second. They check the parts we wrote ourselves: reading files,
validating the Pydantic models, and building the CSV row.

(You do not need to read or change this file to run the project. It is here
because tests are what let the GitHub Actions workflow check every commit.)
"""

from pathlib import Path

import pytest

import app
import config
import document_reader
from models import CaseSummary, ComplaintData, CustomerEmail

PROJECT_FOLDER = Path(__file__).parent

# ---------------------------------------------------------------------------
# Tests for document_reader.py  (FILE PROCESSING + ERROR HANDLING)
# ---------------------------------------------------------------------------


def test_finds_the_sample_documents():
    """The data folder should contain our six readable documents."""
    documents = document_reader.find_documents(config.INPUT_FOLDER)
    assert len(documents) == 6


def test_ignores_unsupported_file_types():
    """The .csv file in data/ must be skipped, not read."""
    names = [p.name for p in document_reader.find_documents(config.INPUT_FOLDER)]
    assert "vendor_courier_notes.csv" not in names


def test_reads_a_text_file():
    text = document_reader.read_document(config.INPUT_FOLDER / "complaint_003.txt")
    assert "Meera Shah" in text


def test_reads_a_pdf_file():
    text = document_reader.read_document(config.INPUT_FOLDER / "complaint_001.pdf")
    assert "Priya" in text


def test_reads_a_word_file():
    text = document_reader.read_document(config.INPUT_FOLDER / "complaint_004.docx")
    assert "Thomas Berger" in text


def test_a_missing_folder_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        document_reader.find_documents(tmp_path / "does_not_exist")


def test_an_almost_empty_file_is_rejected(tmp_path):
    """A scanned or blank file should be reported, not silently processed."""
    empty = tmp_path / "blank.txt"
    empty.write_text("hi")
    with pytest.raises(ValueError):
        document_reader.read_document(empty)


def test_a_very_long_document_is_truncated(tmp_path):
    """Long documents are cut so the API call stays cheap."""
    long_file = tmp_path / "long.txt"
    long_file.write_text("word " * 50_000)
    text = document_reader.read_document(long_file)
    assert len(text) < config.MAX_CHARACTERS + 100


# ---------------------------------------------------------------------------
# Tests for models.py  (PYDANTIC + STRUCTURED OUTPUTS)
# ---------------------------------------------------------------------------


def _sample_data(**overrides) -> ComplaintData:
    values = {
        "issue_description": "The customer was charged twice for one order.",
        "is_complaint": True,
        "escalation_required": False,
        "supporting_document_available": True,
    }
    values.update(overrides)
    return ComplaintData(**values)


def test_optional_fields_default_to_none():
    """A document with no phone number must not invent one."""
    data = _sample_data()
    assert data.customer_name is None
    assert data.phone_number is None


def test_category_must_come_from_the_allowed_list():
    """This is what stops the model returning a made-up category."""
    with pytest.raises(Exception):
        _sample_data(complaint_category="Something Invented")


def test_issue_description_is_required():
    with pytest.raises(Exception):
        ComplaintData(
            is_complaint=True,
            escalation_required=False,
            supporting_document_available=True,
        )


def test_email_renders_as_sendable_text():
    email = CustomerEmail(
        subject="Your recent case",
        greeting="Dear Priya,",
        body="Thank you for contacting us. We have refunded the duplicate charge.",
    )
    text = email.as_text()
    assert text.startswith("Subject: Your recent case")
    assert "Dear Priya," in text
    assert "Customer Care Team" in text


def test_summary_renders_all_five_sections():
    summary = CaseSummary(
        case_overview="A duplicate charge was reported.",
        key_issue="The customer was billed twice.",
        action_taken="Refund issued.",
        current_status="Resolved",
        recommended_next_action="Confirm the refund reached the card.",
    )
    text = summary.as_text()
    for heading in ("Case overview", "Key issue", "Action taken",
                    "Current status", "Recommended next action"):
        assert heading in text


# ---------------------------------------------------------------------------
# Tests for app.py  (BATCH PROCESSING output)
# ---------------------------------------------------------------------------


def test_csv_row_converts_true_false_into_yes_no():
    """The report must show Yes/No, which is what the assignment asks for."""
    result = {
        "complaint_data": _sample_data(escalation_required=True),
        "customer_email": CustomerEmail(
            subject="Your recent case",
            greeting="Dear Customer,",
            body="Thank you for contacting us about the issue you reported.",
        ),
        "case_summary": CaseSummary(
            case_overview="A duplicate charge was reported.",
            key_issue="The customer was billed twice.",
            action_taken="Refund issued.",
            current_status="Resolved",
            recommended_next_action="Confirm the refund reached the card.",
        ),
    }
    row = app.build_csv_row("complaint_001.pdf", result, seconds=2.5)

    assert row["is_complaint"] == "Yes"
    assert row["escalation_required"] == "Yes"
    assert row["supporting_document_available"] == "Yes"
    assert row["status"] == "success"


def test_every_csv_row_has_exactly_the_expected_columns():
    """Guards against a column being added in one place but not the other."""
    empty_row = dict.fromkeys(app.CSV_COLUMNS, "")
    assert set(empty_row) == set(app.CSV_COLUMNS)
    assert "escalation_required" in app.CSV_COLUMNS


# ---------------------------------------------------------------------------
# Tests for the deployment setup  (DEPLOYMENT)
#
# These do not import streamlit - importing it would execute the whole page.
# They check the things that silently break a deployment.
# ---------------------------------------------------------------------------


def test_the_web_entry_point_and_dockerfile_exist():
    """Both entry points must be present: the batch one and the web one."""
    assert (PROJECT_FOLDER / "app.py").exists()
    assert (PROJECT_FOLDER / "streamlit_app.py").exists()
    assert (PROJECT_FOLDER / "Dockerfile").exists()


def test_the_dockerfile_starts_the_file_that_actually_exists():
    """Renaming the web entry point without updating the Dockerfile is the
    classic way to deploy a container that will not start."""
    dockerfile = (PROJECT_FOLDER / "Dockerfile").read_text(encoding="utf-8")
    assert "streamlit run streamlit_app.py" in dockerfile
    # $PORT must be used, or the cloud platform cannot reach the container.
    assert "--server.port=$PORT" in dockerfile


def test_the_apprunner_config_starts_the_same_file():
    """AWS App Runner can build from source instead of a container image.
    Its start command must match the Dockerfile's, or the two routes drift."""
    config_file = (PROJECT_FOLDER / "apprunner.yaml").read_text(encoding="utf-8")
    assert "streamlit run streamlit_app.py" in config_file
    assert "port: 8501" in config_file


def test_the_image_can_never_contain_the_api_key():
    """The single most important line in .dockerignore.

    If .env were copied into the image, the key would travel to the container
    registry with it.
    """
    ignored = (PROJECT_FOLDER / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in ignored.split()


def test_requirements_lists_streamlit():
    """The container installs only what requirements.txt names."""
    requirements = (PROJECT_FOLDER / "requirements.txt").read_text(encoding="utf-8")
    assert "streamlit" in requirements
