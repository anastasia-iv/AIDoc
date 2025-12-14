#!/usr/bin/env python3
from pathlib import Path
import csv
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse MedMentions PubTator file into CSV for text classification"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to MedMentions PubTator .txt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)

    input_file: Path = args.input
    output_file: Path = args.output

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Input file: %s", input_file)
    logging.info("Output file: %s", output_file)

    docs = {}
    entities = []

    with input_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # title or abstract
            if "|t|" in line or "|a|" in line:
                pmid, kind, text = line.split("|", 2)
                docs.setdefault(pmid, {"t": "", "a": ""})
                docs[pmid][kind] = text

            # entity annotation
            elif "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 6:
                    pmid, start, end, mention, sem_type, umls = parts
                    entities.append({
                        "pmid": pmid,
                        "mention": mention,
                        "sem_type": sem_type,
                    })

    logging.info("Parsed %d documents", len(docs))
    logging.info("Parsed %d entity annotations", len(entities))

    rows = []
    skipped = 0

    for e in entities:
        doc = docs.get(e["pmid"])
        if not doc:
            skipped += 1
            continue

        context = (doc["t"] + " " + doc["a"]).strip()

        rows.append({
            "text": e["mention"],
            "context": context,
            "label": e["sem_type"],
        })

    logging.info("Prepared %d samples (%d skipped)", len(rows), skipped)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "context", "label"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Saved dataset to %s", output_file)


if __name__ == "__main__":
    main()
