"""
app.py
======
THIS IS THE FILE YOU RUN.

    python app.py

What it does, in order:

    1. Finds every complaint document in the data/ folder
    2. For each document:
         a. reads the text out of it              (document_reader.py)
         b. runs the 3 AI tasks                   (workflow.py + ai_tasks.py)
         c. saves the 3 results into output/
    3. Writes one summary file: output/final_report.csv
    4. Prints a table of what happened

If one document fails, the program says so and carries on with the rest.
"""

import csv
import json
import logging
import sys
import time

import config
import document_reader
import workflow

# ---------------------------------------------------------------------------
# Logging: one line per important event, with a timestamp.
# Everything is printed to the screen, and also saved to output/run.log so you
# have a record of the run afterwards.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("complaint-processor")


def start_log_file() -> None:
    """Also write every log message to output/run.log."""
    config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.OUTPUT_FOLDER / "run.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

# The columns of the final report, in order.
CSV_COLUMNS = [
    "file_name",
    "status",
    "customer_name",
    "email",
    "phone_number",
    "product_or_service",
    "complaint_category",
    "issue_description",
    "resolution_provided",
    "is_complaint",
    "escalation_required",
    "supporting_document_available",
    "case_status",
    "priority",
    "recommended_next_action",
    "email_subject",
    "seconds",
    "error",
]


def create_output_folders() -> None:
    """Make sure output/ and its three sub-folders exist."""
    for folder in ("structured_data", "customer_emails", "case_summaries"):
        (config.OUTPUT_FOLDER / folder).mkdir(parents=True, exist_ok=True)


def save_results(file_name: str, result: dict) -> None:
    """Write the three artefacts for one document into the output folders.

    `file_name` is the document name without its extension, so
    complaint_001.pdf produces complaint_001.json / .txt / .txt
    """
    data = result["complaint_data"]
    email = result["customer_email"]
    summary = result["case_summary"]

    # 1. The structured data, as JSON.
    (config.OUTPUT_FOLDER / "structured_data" / f"{file_name}.json").write_text(
        json.dumps(data.model_dump(), indent=2), encoding="utf-8"
    )

    # 2. The customer email, as plain text.
    (config.OUTPUT_FOLDER / "customer_emails" / f"{file_name}.txt").write_text(
        email.as_text(), encoding="utf-8"
    )

    # 3. The internal case summary, as plain text.
    (config.OUTPUT_FOLDER / "case_summaries" / f"{file_name}.txt").write_text(
        summary.as_text(), encoding="utf-8"
    )


def build_csv_row(path_name: str, result: dict, seconds: float) -> dict:
    """Turn one successful result into a row for the final report."""
    data = result["complaint_data"]
    return {
        "file_name": path_name,
        "status": "success",
        "customer_name": data.customer_name or "",
        "email": data.email or "",
        "phone_number": data.phone_number or "",
        "product_or_service": data.product_or_service or "",
        "complaint_category": data.complaint_category,
        "issue_description": data.issue_description,
        "resolution_provided": data.resolution_provided or "",
        # The assignment asks for Yes/No, so convert the True/False here.
        "is_complaint": "Yes" if data.is_complaint else "No",
        "escalation_required": "Yes" if data.escalation_required else "No",
        "supporting_document_available": (
            "Yes" if data.supporting_document_available else "No"
        ),
        "case_status": data.case_status,
        "priority": data.priority,
        "recommended_next_action": result["case_summary"].recommended_next_action,
        "email_subject": result["customer_email"].subject,
        "seconds": round(seconds, 1),
        "error": "",
    }


def write_final_report(rows: list[dict]) -> None:
    """Write output/final_report.csv - one row per document."""
    report_path = config.OUTPUT_FOLDER / "final_report.csv"
    # utf-8-sig so that Excel opens the file with the right characters.
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Final report written to %s", report_path)


def print_summary_table(rows: list[dict]) -> None:
    """Print a readable table of the results at the end of the run."""
    print("\n" + "=" * 92)
    print(f"{'DOCUMENT':<22}{'STATUS':<10}{'CATEGORY':<18}{'ESCALATE':<10}"
          f"{'CASE STATUS':<14}{'PRIORITY'}")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['file_name']:<22}{row['status']:<10}{row['complaint_category']:<18}"
            f"{row['escalation_required']:<10}{row['case_status']:<14}{row['priority']}"
        )
    print("=" * 92)


def main() -> int:
    """Run the whole pipeline. Returns 0 if everything worked."""
    print("\n" + "=" * 92)
    print("  AI CUSTOMER COMPLAINT & CASE PROCESSING SYSTEM")
    print("=" * 92)

    # --- Step 0: check the API key before doing anything else ------------
    if not config.OPENAI_API_KEY:
        log.error("No OpenAI API key found.")
        log.error("Create a file named  .env  in this folder containing one line:")
        log.error("    OPENAI_API_KEY=sk-your-key-here")
        log.error("(There is an example file called .env.example you can copy.)")
        return 1

    # --- Step 1: find the documents -------------------------------------
    try:
        documents = document_reader.find_documents(config.INPUT_FOLDER)
    except FileNotFoundError as error:
        log.error("%s", error)
        return 1

    if not documents:
        log.error("No documents found in %s", config.INPUT_FOLDER)
        log.error("Supported types are: %s", ", ".join(config.SUPPORTED_FILE_TYPES))
        return 1

    log.info("Found %d document(s) in %s", len(documents), config.INPUT_FOLDER)
    create_output_folders()
    start_log_file()

    # --- Step 2: process each document ----------------------------------
    rows: list[dict] = []
    failures = 0

    for number, path in enumerate(documents, start=1):
        log.info("[%d/%d] Processing %s", number, len(documents), path.name)
        started = time.time()

        try:
            # a. file -> text
            text = document_reader.read_document(path)

            # b. text -> 3 AI results (this is where the LLM is called)
            result = workflow.process_document(text)

            # c. save the 3 results
            save_results(path.stem, result)

            rows.append(build_csv_row(path.name, result, time.time() - started))
            log.info("      done in %.1fs", time.time() - started)

        except Exception as error:
            # One bad document must not stop the whole batch.
            failures += 1
            log.error("      FAILED: %s", error)
            failed_row = dict.fromkeys(CSV_COLUMNS, "")
            failed_row["file_name"] = path.name
            failed_row["status"] = "failed"
            failed_row["error"] = str(error)
            rows.append(failed_row)

    # --- Step 3: the consolidated report --------------------------------
    write_final_report(rows)
    print_summary_table(rows)

    succeeded = len(rows) - failures
    print(f"\n  {succeeded} succeeded, {failures} failed")
    print(f"  All results are in: {config.OUTPUT_FOLDER}\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
