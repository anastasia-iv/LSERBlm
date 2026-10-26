import argparse
import json

from .config import load_config


def config_options(parser, default):
    parser.add_argument("--config", default=default)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")


def device_options(parser):
    parser.add_argument("--device", default="auto")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serbian decoder-only language models.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hardware")

    p = sub.add_parser("prepare")
    config_options(p, "configs/data.json")
    p.add_argument("--output", required=True)
    p.add_argument("--input-jsonl")

    p = sub.add_parser("expand-corpus")
    config_options(p, "configs/data_expanded.json")
    p.add_argument("--base-corpus", required=True)
    p.add_argument("--input-jsonl", nargs="*", default=[])
    p.add_argument("--output", required=True)

    p = sub.add_parser("audit-data")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output")

    p = sub.add_parser("tokenizer", help="Train ByteLevel BPE on train documents.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocab-size", type=int, default=8192)

    p = sub.add_parser("encode")
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output")
    p.add_argument("--reuse-tokenizer", action="store_true")
    p.add_argument("--include-test", action="store_true")

    p = sub.add_parser("model-info")
    config_options(p, "configs/main_rope.json")

    p = sub.add_parser("speed")
    config_options(p, "configs/main_rope.json")
    p.add_argument("--output", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    device_options(p)

    for command, default in [
        ("train", "configs/main_rope.json"),
        ("continue-train", "configs/continued_deep.json"),
        ("sft", "configs/sft.json"),
    ]:
        p = sub.add_parser(command)
        config_options(p, default)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--resume")
        p.add_argument("--stop-after", type=int)
        if command in {"continue-train", "sft"}:
            p.add_argument("--init-from")
        if command == "sft":
            p.add_argument("--train-jsonl", required=True)
            p.add_argument("--validation-jsonl", required=True)
        device_options(p)

    p = sub.add_parser("generate")
    p.add_argument("--checkpoint", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--question")
    p.add_argument("--context")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="auto")
    device_options(p)

    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--precision", default="auto")
    device_options(p)

    p = sub.add_parser("evaluate-sft")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--precision", default="auto")
    device_options(p)

    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--examples", default="benchmarks/serbian_25.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--precision", default="auto")
    p.add_argument("--question-format", choices=["raw", "sft"], default="raw")
    device_options(p)

    args = parser.parse_args(argv)
    config = load_config(args.config, args.set) if hasattr(args, "config") else None
    if args.command == "hardware":
        from .measurement import hardware

        result = hardware()
    elif args.command == "prepare":
        from .data import prepare

        result = prepare(config, args.output, args.input_jsonl)
    elif args.command == "expand-corpus":
        from .expansion import expand_corpus

        result = expand_corpus(args.base_corpus, args.input_jsonl, args.output, config)
    elif args.command == "audit-data":
        from .data import audit_splits

        result = audit_splits(args.corpus, output=args.output)
    elif args.command == "tokenizer":
        from .tokenization import train_tokenizer

        result = train_tokenizer(args.corpus, args.output, args.vocab_size)
    elif args.command == "encode":
        from .tokenization import encode_corpus

        result = encode_corpus(
            args.corpus, args.tokenizer, args.include_test, args.output, args.reuse_tokenizer
        )
    elif args.command == "model-info":
        from .measurement import describe

        result = describe(config)
    elif args.command == "speed":
        from .measurement import speed

        result = speed(config, args.output, args.device, args.warmup, args.steps)
    elif args.command in {"train", "continue-train", "sft"}:
        from .training import train

        expected_mode = {"train": "pretrain", "continue-train": "continued_pretrain", "sft": "sft"}[
            args.command
        ]
        if config["train"]["mode"] != expected_mode:
            raise ValueError("The CLI command and config train.mode disagree.")
        result = train(
            config,
            args.tokenizer,
            args.output,
            resume=args.resume,
            init_from=getattr(args, "init_from", None),
            sft_train=getattr(args, "train_jsonl", None),
            sft_validation=getattr(args, "validation_jsonl", None),
            stop_after=args.stop_after,
            device_name=args.device,
        )
    elif args.command == "generate":
        from .inference import Generator

        generator = Generator(args.checkpoint, args.device, args.precision)
        options = dict(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        result = (
            generator.answer(args.question, args.context, **options)
            if args.question
            else generator.generate(args.prompt, **options)
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_lm

        result = evaluate_lm(
            args.checkpoint,
            args.data,
            args.output,
            args.device,
            args.precision,
            args.limit,
            args.stride,
        )
    elif args.command == "evaluate-sft":
        from .evaluation import evaluate_sft

        result = evaluate_sft(
            args.checkpoint, args.data, args.output, args.device, args.precision, args.batch_size
        )
    elif args.command == "report":
        from .reporting import summarize_runs

        result = summarize_runs(args.runs, args.output)
    elif args.command == "benchmark":
        from .benchmark import GENERATION, run_benchmark

        if args.max_new_tokens < 1:
            parser.error("--max-new-tokens must be positive")
        result = run_benchmark(
            args.checkpoint,
            args.examples,
            args.output,
            args.device,
            args.precision,
            args.limit,
            {**GENERATION, "max_new_tokens": args.max_new_tokens},
            question_format=args.question_format,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
