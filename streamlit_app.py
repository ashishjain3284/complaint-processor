"""
streamlit_app.py
================
THE WEB VERSION OF THIS PROJECT.

    streamlit run streamlit_app.py

This file adds a browser interface so the project can be deployed and shared
as a URL. It is the ONLY new application code needed for that - nothing in the
six original modules was changed.

    app.py               the batch version:  python app.py
    streamlit_app.py     the web version:    streamlit run streamlit_app.py
                                              |
                                              +--> document_reader.read_document()
                                              +--> workflow.process_document()
                                              +--> app.build_csv_row()

Both entry points call exactly the same pipeline. The UI only decides what to
show; every piece of logic still lives in the modules underneath.

One deliberate difference: the batch version writes its results into the
output/ folder. This version keeps them in memory and offers them as download
buttons instead, because a deployed container has a temporary disk - anything
written there disappears when the platform restarts the app.
"""

import csv
import io
import json
import tempfile
import time
from pathlib import Path

import streamlit as st

# The existing project modules. None of these were modified for the web UI.
import app as batch          # reused for CSV_COLUMNS and build_csv_row
import config
import document_reader
import workflow

# ---------------------------------------------------------------------------
# Page setup - this must be the first Streamlit call in the file.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Complaint & Case Processing System",
    page_icon="📋",
    layout="wide",
)


# ===========================================================================
# HELPERS
#
# These four functions are the entire adaptation layer between the pipeline
# and the browser. They are plain functions with no Streamlit calls inside
# process_one() or build_csv_text(), so both are covered by the tests.
# ===========================================================================
def process_one(path: Path) -> dict:
    """Run the full pipeline on one file and return everything the UI needs.

    This is the web equivalent of the loop body in app.py: read the file, run
    the workflow, time it. It never raises - a failure comes back as part of
    the result, so one bad document cannot abandon the rest of the batch.
    """
    started = time.time()
    try:
        text = document_reader.read_document(path)
        result = workflow.process_document(text)
        seconds = time.time() - started
        return {
            "file_name": path.name,
            "ok": True,
            "seconds": seconds,
            "result": result,
            "row": batch.build_csv_row(path.name, result, seconds),
            "error": "",
        }
    except Exception as error:
        # Same behaviour as the batch version: record the failure, carry on.
        failed_row = dict.fromkeys(batch.CSV_COLUMNS, "")
        failed_row["file_name"] = path.name
        failed_row["status"] = "failed"
        failed_row["error"] = str(error)
        return {
            "file_name": path.name,
            "ok": False,
            "seconds": time.time() - started,
            "result": None,
            "row": failed_row,
            "error": str(error),
        }


def build_csv_text(rows: list[dict]) -> str:
    """Build final_report.csv in memory so it can be offered as a download."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=batch.CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def save_uploads(uploaded_files) -> list[Path]:
    """Write the browser uploads to a temporary folder and return their paths.

    document_reader decides how to read a file from its extension, so the
    uploads have to become real files on disk under their original names.
    """
    folder = Path(tempfile.mkdtemp(prefix="uploads_"))
    paths = []
    for uploaded in uploaded_files:
        path = folder / Path(uploaded.name).name    # strip any directory part
        path.write_bytes(uploaded.getbuffer())
        paths.append(path)
    return sorted(paths)


def run_batch(paths: list[Path]) -> None:
    """Process a list of documents, showing progress, and store the results."""
    progress = st.progress(0.0, text="Starting…")
    results = []

    for number, path in enumerate(paths, start=1):
        progress.progress(
            (number - 1) / len(paths),
            text=f"Processing {path.name}  ({number} of {len(paths)})",
        )
        results.append(process_one(path))

    progress.progress(1.0, text=f"Finished {len(paths)} document(s)")
    st.session_state["results"] = results


# ===========================================================================
# SIDEBAR - status and a short explanation of what the app does
# ===========================================================================
with st.sidebar:
    st.header("System status")

    if config.OPENAI_API_KEY:
        st.success("OpenAI key loaded")
    else:
        st.error("No OpenAI key")

    st.caption(f"**Model** · {config.MODEL_NAME}")
    st.caption(f"**Accepted files** · {', '.join(config.SUPPORTED_FILE_TYPES)}")
    st.caption(f"**Size limit** · {config.MAX_CHARACTERS:,} characters per document")

    st.divider()
    st.subheader("What happens to a document")
    st.markdown(
        """
1. **Read** — text is pulled out of the PDF, Word or text file
2. **Extract** — an LLM fills in a Pydantic model of 13 fields
3. **Reply** — a customer email is drafted *(runs in parallel)*
4. **Summarise** — an internal case note is drafted *(runs in parallel)*

Steps 2–4 are a LangGraph workflow. Steps 3 and 4 do not depend on each
other, so the graph runs them at the same time.
        """
    )

    st.divider()
    st.caption(
        "The same pipeline runs from the command line with `python app.py`, "
        "which processes the `data/` folder as a batch and writes `output/`."
    )


# ===========================================================================
# LANDING PAGE
# ===========================================================================
st.title("AI Customer Complaint & Case Processing System")
st.markdown(
    "Give it a complaint document and it extracts the case details, drafts a "
    "reply to the customer, and writes an internal summary for the management "
    "team — from a single pass over the document."
)

# ---- Stop here with clear instructions if the key is missing --------------
if not config.OPENAI_API_KEY:
    st.error("**The application is not configured yet.**")
    st.markdown(
        """
No `OPENAI_API_KEY` was found, so no document can be processed.

**Running locally** — create a file called `.env` next to `app.py`:

```
OPENAI_API_KEY=sk-your-key-here
```

**Running in the cloud** — set `OPENAI_API_KEY` in the hosting platform's
settings (Azure: *Configuration → Application settings*; AWS App Runner:
*Configuration → Environment variables*), then restart the app.
        """
    )
    st.stop()

st.divider()

# ---- Choose the input ------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Choose what to process")
    source = st.radio(
        "Source",
        ["Use the sample documents", "Upload my own documents"],
        label_visibility="collapsed",
        horizontal=True,
    )

    paths: list[Path] = []

    if source == "Upload my own documents":
        uploaded = st.file_uploader(
            "Complaint documents",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            help="PDF, Word or plain text. You can select several at once.",
        )
        if uploaded:
            paths = save_uploads(uploaded)
            st.caption(f"{len(paths)} file(s) ready.")
    else:
        try:
            paths = document_reader.find_documents(config.INPUT_FOLDER)
        except FileNotFoundError:
            paths = []
        if paths:
            st.caption(
                f"{len(paths)} sample document(s) in `data/` — "
                + ", ".join(p.name for p in paths)
            )
        else:
            st.warning("No sample documents were found in the `data/` folder.")

    start = st.button(
        "Process documents",
        type="primary",
        disabled=not paths,
        use_container_width=True,
    )

with right:
    st.subheader("What you get back")
    st.markdown(
        """
For every document:

- **Structured case data** — 13 validated fields, downloadable as JSON
- **Customer email** — a ready-to-send reply
- **Internal case summary** — for the management team

And for the batch as a whole, a **`final_report.csv`** with one row per
document — the consolidated report the assignment asks for.
        """
    )

# ---- Run -------------------------------------------------------------------
if start and paths:
    run_batch(paths)

results = st.session_state.get("results")

# ===========================================================================
# RESULTS
# ===========================================================================
if results:
    st.divider()
    st.header("Results")

    succeeded = sum(1 for item in results if item["ok"])
    failed = len(results) - succeeded
    escalations = sum(
        1 for item in results
        if item["ok"] and item["result"]["complaint_data"].escalation_required
    )
    total_seconds = sum(item["seconds"] for item in results)

    one, two, three, four = st.columns(4)
    one.metric("Processed", len(results))
    two.metric("Succeeded", succeeded)
    three.metric("Failed", failed)
    four.metric("Need escalation", escalations)
    st.caption(
        f"Total time {total_seconds:.1f}s · "
        f"{total_seconds / len(results):.1f}s per document"
    )

    # ---- The consolidated report ------------------------------------------
    st.subheader("Consolidated report")
    rows = [item["row"] for item in results]
    st.dataframe(
        [
            {
                "Document": row["file_name"],
                "Status": row["status"],
                "Category": row["complaint_category"],
                "Complaint": row["is_complaint"],
                "Escalate": row["escalation_required"],
                "Case status": row["case_status"],
                "Priority": row["priority"],
            }
            for row in rows
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download final_report.csv",
        # utf-8-sig so that Excel opens the file with the right characters.
        data=build_csv_text(rows).encode("utf-8-sig"),
        file_name="final_report.csv",
        mime="text/csv",
        type="primary",
    )

    # ---- One expander per document ----------------------------------------
    st.subheader("Each document in detail")

    for item in results:
        if not item["ok"]:
            with st.expander(f"❌  {item['file_name']} — failed"):
                st.error(item["error"])
                st.caption(
                    "The batch carried on with the other documents. This is the "
                    "same behaviour as the command-line version."
                )
            continue

        data = item["result"]["complaint_data"]
        email = item["result"]["customer_email"]
        summary = item["result"]["case_summary"]

        flag = "🔴" if data.escalation_required else "🟢"
        heading = (
            f"{flag}  {item['file_name']} — {data.complaint_category} · "
            f"{data.priority} priority · {data.case_status}"
        )

        with st.expander(heading):
            tab_data, tab_email, tab_summary = st.tabs(
                ["Structured data", "Customer email", "Internal case summary"]
            )

            # --- 1. the extracted fields -------------------------------
            with tab_data:
                facts, flags = st.columns(2)
                with facts:
                    st.markdown(f"**Customer** · {data.customer_name or '—'}")
                    st.markdown(f"**Email** · {data.email or '—'}")
                    st.markdown(f"**Phone** · {data.phone_number or '—'}")
                    st.markdown(f"**Product / service** · {data.product_or_service or '—'}")
                with flags:
                    st.markdown(f"**Category** · {data.complaint_category}")
                    st.markdown(f"**Is a complaint** · {'Yes' if data.is_complaint else 'No'}")
                    st.markdown(
                        f"**Escalation required** · "
                        f"{'Yes' if data.escalation_required else 'No'}"
                    )
                    st.markdown(
                        f"**Supporting document** · "
                        f"{'Yes' if data.supporting_document_available else 'No'}"
                    )

                st.markdown("**Issue**")
                st.write(data.issue_description)
                st.markdown("**Resolution provided**")
                st.write(data.resolution_provided or "— none recorded —")

                json_text = json.dumps(data.model_dump(), indent=2)
                st.download_button(
                    "Download JSON",
                    data=json_text,
                    file_name=f"{Path(item['file_name']).stem}.json",
                    mime="application/json",
                    key=f"json_{item['file_name']}",
                )

            # --- 2. the customer email ---------------------------------
            with tab_email:
                st.markdown(f"**Subject:** {email.subject}")
                st.text(email.as_text())
                st.download_button(
                    "Download email",
                    data=email.as_text(),
                    file_name=f"{Path(item['file_name']).stem}_email.txt",
                    mime="text/plain",
                    key=f"email_{item['file_name']}",
                )

            # --- 3. the internal summary -------------------------------
            with tab_summary:
                st.markdown(f"**Case overview** · {summary.case_overview}")
                st.markdown(f"**Key issue** · {summary.key_issue}")
                st.markdown(f"**Action taken** · {summary.action_taken}")
                st.markdown(f"**Current status** · {summary.current_status}")
                st.info(f"**Recommended next action** · {summary.recommended_next_action}")
                st.download_button(
                    "Download summary",
                    data=summary.as_text(),
                    file_name=f"{Path(item['file_name']).stem}_summary.txt",
                    mime="text/plain",
                    key=f"summary_{item['file_name']}",
                )

    if st.button("Clear results"):
        del st.session_state["results"]
        st.rerun()
