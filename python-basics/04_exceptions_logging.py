"""
Exceptions + logging: how an inference script should handle bad input.

Why this matters: on an edge device, input is unreliable -- a corrupted image
file, a sensor glitch, a missing file. A script that crashes on the first bad
frame is useless in the field. It must log the problem and keep going.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference_demo")


def load_and_validate(record):
    """Simulates loading one 'sensor reading' and checking it's usable.

    Raises specific exceptions so the caller can decide what to do about each.
    """
    if "value" not in record:
        raise KeyError(f"record {record.get('id', '?')} is missing 'value'")

    value = record["value"]
    if not isinstance(value, (int, float)):
        raise TypeError(f"record {record['id']} has non-numeric value: {value!r}")

    if value < 0 or value > 1:
        # Values here represent a normalized sensor/model output, must be in [0, 1].
        raise ValueError(f"record {record['id']} value out of range [0,1]: {value}")

    return value


def process_batch(records):
    """Process every record, skipping bad ones instead of crashing the batch."""
    good_values = []
    for record in records:
        try:
            value = load_and_validate(record)
        except KeyError as e:
            logger.error("Skipping record, missing field: %s", e)
            continue
        except TypeError as e:
            logger.error("Skipping record, wrong type: %s", e)
            continue
        except ValueError as e:
            logger.warning("Skipping record, value out of range: %s", e)
            continue
        else:
            # else-branch on try only runs if no exception was raised -- keeps
            # the "success path" visually separate from error handling.
            good_values.append(value)
            logger.info("record %s OK: value=%.3f", record["id"], value)
        finally:
            # finally always runs, used here just to make a heartbeat log per record.
            logger.debug("finished handling record %s", record.get("id", "?"))

    return good_values


if __name__ == "__main__":
    # A realistic mixed batch: some good, some malformed, some out of range.
    records = [
        {"id": 1, "value": 0.87},
        {"id": 2, "value": "not_a_number"},   # bad type
        {"id": 3},                             # missing field
        {"id": 4, "value": 1.5},               # out of range
        {"id": 5, "value": 0.12},
    ]

    results = process_batch(records)
    print(f"\nProcessed batch. {len(results)}/{len(records)} records were valid.")
    print("Valid values:", results)
