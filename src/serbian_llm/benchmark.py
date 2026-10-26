import hashlib
from collections import defaultdict

from .inference import Generator
from .utils import file_sha256, read_jsonl, write_json


GENERATION = {"max_new_tokens": 96, "temperature": 0.8, "top_k": 40, "seed": 42}
SCORE_FIELDS = ("language", "coherence", "task")


def load_cases(path):
    cases = list(read_jsonl(path))
    ids = [c["id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("Benchmark needs nonempty cases with unique IDs.")
    for case in cases:
        if (
            case["type"] not in {"question", "continuation"}
            or not 1 <= case["level"] <= 5
            or not case["prompt"].strip()
            or not case["criterion_ru"].strip()
        ):
            raise ValueError("Invalid benchmark case: " + str(case["id"]))
    return cases


def completion_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def benchmark_report(generator, examples, run_name, options=None):
    return {
        "run": run_name,
        "benchmark_sha256": file_sha256(examples),
        "checkpoint_step": generator.checkpoint["state"]["step"],
        "model_config": generator.checkpoint["model_config"],
        "tokenizer_sha256": generator.checkpoint["identity"]["tokenizer_sha256"],
        "precision": generator.precision,
        "options": dict(options or GENERATION),
        "protocol": "Exact raw Serbian prompts; no chat template. Questions: greedy; continuations: seeded sampling. EOS can end early.",
        "examples": [],
    }


def generate_case(generator, case, options=None, question_format="raw"):
    options = dict(options or GENERATION)
    if case["type"] == "question":
        options["temperature"] = 0.0
    if question_format not in {"raw", "sft"}:
        raise ValueError("question_format must be raw or sft")
    if question_format == "sft" and case["type"] == "question":
        result = generator.answer(case["prompt"], **options)
        result["model_prompt"] = result.pop("prompt")
        result["prompt"] = case["prompt"]
    else:
        result = generator.generate(case["prompt"], **options)
    if result["prompt_was_truncated"]:
        raise ValueError("Benchmark prompt exceeds model context: " + str(case["id"]))
    return {
        **case,
        **result,
        "generation": options,
        "completion_sha256": completion_hash(result["completion"]),
    }


def find_review(record, reviews):
    review = reviews.get(str(record["id"]))
    if (
        review is None
        or review.get("completion_sha256") != completion_hash(record["completion"])
        or review.get("prompt_sha256") != completion_hash(record["prompt"])
    ):
        return None
    scores = review["scores"]
    if any(type(scores.get(k)) is not int or not 0 <= scores[k] <= 2 for k in SCORE_FIELDS):
        raise ValueError("Review scores must be integers in 0..2.")
    return review


def summarize(report, reviews):
    reviewed = [(record, find_review(record, reviews)) for record in report["examples"]]
    reviewed = [(record, review) for record, review in reviewed if review is not None]
    groups = defaultdict(list)
    for record, review in reviewed:
        groups["all"].append(review)
        groups["level_" + str(record["level"])].append(review)
        groups[record["type"]].append(review)
    return {
        "generated": len(report["examples"]),
        "reviewed": len(reviewed),
        "checkpoint_step": report["checkpoint_step"],
        "scores": {
            group: {
                "examples": len(items),
                **{
                    field: sum(r["scores"][field] for r in items) / len(items)
                    for field in SCORE_FIELDS
                },
            }
            for group, items in groups.items()
        },
        "note": "Manual assistant review, not an independent expert benchmark. Scores cover reviewed cases only.",
    }


def run_benchmark(
    checkpoint,
    examples,
    output,
    device="auto",
    precision="auto",
    limit=0,
    options=None,
    question_format="raw",
):
    if limit < 0:
        raise ValueError("limit must be nonnegative; 0 means all cases.")
    cases = load_cases(examples)
    generator = Generator(checkpoint, device, precision)
    report = benchmark_report(generator, examples, str(checkpoint), options)
    report["question_format"] = question_format
    if question_format == "sft":
        report["protocol"] = (
            "Questions use Питање/Одговор, greedy. Continuations use raw prompts, seeded sampling. EOS can end early."
        )
    for case in cases[: limit or None]:
        report["examples"].append(
            generate_case(generator, case, report["options"], question_format)
        )
    write_json(output, report)
    return {
        "output": str(output),
        "examples": len(report["examples"]),
        "checkpoint_step": report["checkpoint_step"],
        "benchmark_sha256": report["benchmark_sha256"],
    }
