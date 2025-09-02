import os
import argparse
import subprocess
import sys
from pathlib import Path
from openai import OpenAI

def test_blocking(client: OpenAI, text: str, output_path: Path):
    print(f"\n--- Running Blocking Test ---")
    print(f"Text: '{text}'")
    print(f"Saving to: {output_path}")

    try:
        response = client.audio.speech.create(
            model="csm-1b",
            voice="alloy",
            input=text,
            extra_body=dict(
                speaker_id=0,
                temperature=0.75,
            )
        )
        response.stream_to_file(output_path)
        print(f"✅ Success! Audio saved to {output_path}")
    except Exception as e:
        print(f"❌ Error during blocking test: {e}")
        sys.exit(1)


def test_streaming(client: OpenAI, text: str, output_path: Path):
    print(f"\n--- Running Streaming Test ---")
    print(f"Text: '{text}'")
    print(f"Saving to: {output_path}")

    try:
        with client.audio.speech.with_streaming_response.create(
            model="csm-1b",
            voice="alloy",
            input=text,
            response_format="wav",
            extra_body=dict(
                speaker_id=1,
            )
        ) as response:
            chunk_count = 0
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024):
                    f.write(chunk)
                    chunk_count += 1

            print(f"✅ Success! Received and saved {chunk_count} audio chunks to {output_path}")

    except Exception as e:
        print(f"❌ Error during streaming test: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test script for the CSM OpenAI-compatible API.")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host of the server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port of the server."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("CSM_API_KEY"),
        help="API key for authentication. Can also be set via CSM_API_KEY env var."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello there",
        help="Text to synthesize."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default="test_output.wav",
        help="Output file for the blocking test."
    )
    parser.add_argument(
        "--streaming-output-file",
        type=Path,
        default="test_streaming_output.wav",
        help="Output file for the streaming test."
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key=args.api_key if args.api_key else "dummy-key"
    )

    test_blocking(client, args.text, args.output_file)
    test_streaming(client, args.text, args.streaming_output_file)

if __name__ == "__main__":
    main()