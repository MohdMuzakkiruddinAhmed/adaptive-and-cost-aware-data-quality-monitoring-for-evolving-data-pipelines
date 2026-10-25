"""Check that AWS Bedrock is reachable from this machine (needed for Experiment 3).

    python scripts/check_bedrock.py                 # credentials type, region, available models
    python scripts/check_bedrock.py --probe         # try every us.* Claude / OpenAI / Meta / Mistral / DeepSeek profile
                                                    # (tiny calls, < $0.01 total)
    python scripts/check_bedrock.py --model <id>    # send one short test message to that model

Credentials are read from the environment (never from this repository):
    AWS_BEARER_TOKEN_BEDROCK                         Bedrock API key
    or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY     IAM access key (or `aws configure`)
    AWS_REGION                                       e.g. us-east-1
"""
import argparse
import os
import sys

try:
    import boto3
except ImportError:
    sys.exit("boto3 is not installed: pip install --upgrade boto3")


FAMILIES = ("anthropic", "openai", "meta", "mistral", "deepseek")   # meta / mistral / deepseek: candidate
                                                                     # independent authors (protocol.yaml manual_author)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model ID or inference profile ID to test")
    ap.add_argument("--probe", action="store_true",
                    help="test which us.* Claude / OpenAI inference profiles this account can call")
    args = ap.parse_args()

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        kind = "Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)"
    elif os.environ.get("AWS_ACCESS_KEY_ID"):
        kind = "IAM access key (environment variables)"
    else:
        kind = "none in environment; trying ~/.aws/credentials (aws configure)"
    print(f"boto3 version : {boto3.__version__}")
    print(f"credentials   : {kind}")
    print(f"region        : {region or 'NOT SET  -> set AWS_REGION'}")
    if not region:
        sys.exit(1)

    bedrock = boto3.client("bedrock", region_name=region)
    runtime = boto3.client("bedrock-runtime", region_name=region)
    profiles = []
    try:
        profiles = [p["inferenceProfileId"] for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]]
        models = [p for p in profiles if any(f in p for f in FAMILIES)]
        print("\nInference profiles in this region (rule generators and candidate independent authors):")
        for p in sorted(models):
            print(f"  {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not list models: {type(exc).__name__}: {exc}")

    if args.probe:
        ids = sorted(p for p in profiles if p.startswith(tuple(f"us.{f}." for f in FAMILIES)))
        print(f"\nProbing {len(ids)} us.* profiles with a tiny request each:")
        ok = []
        for mid in ids:
            try:
                runtime.converse(modelId=mid, messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
                                 inferenceConfig={"maxTokens": 16})
                ok.append(mid)
                print(f"  CALLABLE       {mid}")
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
                msg = getattr(exc, "response", {}).get("Error", {}).get("Message", str(exc))[:90]
                print(f"  not available  {mid}  ({code}: {msg})")
        print(f"\n{len(ok)} callable model(s): {', '.join(ok) if ok else 'none'}")

    if args.model:
        resp = runtime.converse(
            modelId=args.model,
            messages=[{"role": "user", "content": [{"text": "Reply with exactly: Bedrock OK"}]}],
            inferenceConfig={"maxTokens": 2000},
        )
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        print(f"\nTest call to {args.model}: {text.strip()!r}")
        print(f"tokens: {resp['usage']}")


if __name__ == "__main__":
    main()
