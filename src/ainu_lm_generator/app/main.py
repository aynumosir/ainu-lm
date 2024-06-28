from ..services import generate_back_translation, generate_rule_based_errors
from .args import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.task == "rule-based":
        generate_rule_based_errors(
            dataset_name=args.dataset_name,
            repeat=args.repeat,
            push_to_hub=args.push_to_hub,
        )
    elif args.task == "back-translation":
        generate_back_translation(
            dataset_name=args.dataset_name,
            inference_endpoint_url=args.inference_endpoint_url,
            semaphore_count=args.semaphore_count,
            batch_size=args.batch_size,
            push_to_hub=args.push_to_hub,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
