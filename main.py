"""
Keboola Component: Job Title Mapping using Claude LLM

Maps raw job titles to standardized titles using Claude API.
"""

import json
import logging
import time
from pathlib import Path

import anthropic
import pandas as pd
from keboola.component import CommonInterface
from pydantic import BaseModel

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models that support structured outputs (beta)
# See: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
STRUCTURED_OUTPUT_MODELS = {
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1",
    "claude-opus-4-1-20250805",
}


class MappedTitle(BaseModel):
    """Schema for a single mapped job title."""
    id: str
    job_title: str
    standardized_job_title: str | None
    department: str | None
    seniority: float


class MappedTitlesResponse(BaseModel):
    """Schema for the batch response."""
    mappings: list[MappedTitle]


def create_taxonomy_reference(df_taxonomy: pd.DataFrame) -> str:
    """Create a formatted taxonomy reference for the LLM prompt."""
    lines = []
    for _, row in df_taxonomy.iterrows():
        title = row["standardized_job_title"]
        seniority = row["seniority"]
        department = row["department"]
        lines.append(f"- {title} | seniority: {seniority} | department: {department}")
    return "\n".join(lines)


def get_json_schema() -> dict:
    """Return JSON schema for structured output."""
    return {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "job_title": {"type": "string"},
                        "standardized_job_title": {"type": ["string", "null"]},
                        "department": {"type": ["string", "null"]},
                        "seniority": {"type": "number"}
                    },
                    "required": ["id", "job_title", "standardized_job_title", "department", "seniority"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["mappings"],
        "additionalProperties": False
    }


def build_system_prompt(taxonomy_ref: str) -> str:
    """Build the system prompt with taxonomy (cacheable)."""
    return f"""You are a job title mapping assistant. Your task is to map raw job titles to standardized titles from a taxonomy.

## Taxonomy (standardized titles with seniority and department):
{taxonomy_ref}

## Instructions:
1. For each job title, find the BEST matching standardized title from the taxonomy above.
2. If no good match exists (similarity < 50%), use null for standardized_job_title and -1 for seniority.
3. Use the seniority and department values from the matched taxonomy entry.
4. Return the results in the specified JSON format.

Always return a JSON object with "mappings" array containing all mapped titles."""


def build_user_prompt(job_titles: list[dict]) -> str:
    """Build the user prompt with job titles (dynamic, not cached)."""
    titles_list = "\n".join([
        f"{i+1}. ID: {t['id']} | Title: {t['job_title']}"
        for i, t in enumerate(job_titles)
    ])

    return f"""Map these job titles to the taxonomy:

{titles_list}"""


def map_titles_with_structured_output(
    client: anthropic.Anthropic,
    job_titles: list[dict],
    taxonomy_ref: str,
    model: str
) -> tuple[list[dict], dict]:
    """Map titles using structured outputs (beta feature) with prompt caching."""
    system_prompt = build_system_prompt(taxonomy_ref)
    user_prompt = build_user_prompt(job_titles)

    response = client.beta.messages.create(
        model=model,
        max_tokens=16384,
        betas=["structured-outputs-2025-11-13", "prompt-caching-2024-07-31"],
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        output_format={
            "type": "json_schema",
            "schema": get_json_schema()
        }
    )

    # Extract usage stats
    usage_stats = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }

    result = json.loads(response.content[0].text)
    return result.get("mappings", []), usage_stats


def map_titles_without_structured_output(
    client: anthropic.Anthropic,
    job_titles: list[dict],
    taxonomy_ref: str,
    model: str
) -> tuple[list[dict], dict]:
    """Map titles using regular JSON output (fallback) with prompt caching."""
    system_prompt = build_system_prompt(taxonomy_ref) + """

## Output format (JSON):
{
  "mappings": [
    {"id": "...", "job_title": "original", "standardized_job_title": "matched or null", "department": "...", "seniority": number}
  ]
}

Return ONLY valid JSON, no markdown formatting or explanation."""

    user_prompt = build_user_prompt(job_titles)

    response = client.messages.create(
        model=model,
        max_tokens=16384,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_prompt}]
    )

    # Extract usage stats
    usage_stats = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }

    response_text = response.content[0].text.strip()

    # Handle potential markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
        return result.get("mappings", []), usage_stats
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Response was: {response_text[:500]}")
        return [], usage_stats


def map_titles_batch(
    client: anthropic.Anthropic,
    job_titles: list[dict],
    taxonomy_ref: str,
    model: str,
    use_structured_output: bool
) -> tuple[list[dict], dict]:
    """Map a batch of job titles to standardized titles."""
    if use_structured_output and model in STRUCTURED_OUTPUT_MODELS:
        return map_titles_with_structured_output(client, job_titles, taxonomy_ref, model)
    else:
        if use_structured_output and model not in STRUCTURED_OUTPUT_MODELS:
            logger.warning(f"Model {model} doesn't support structured outputs, using fallback")
        return map_titles_without_structured_output(client, job_titles, taxonomy_ref, model)


# ============================================================================
# Batch API Functions
# ============================================================================

def create_batch_requests(
    titles_to_map: list[dict],
    taxonomy_ref: str,
    model: str,
    batch_size: int
) -> list[dict]:
    """Create batch requests for the Message Batches API."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    system_prompt = build_system_prompt(taxonomy_ref)
    requests = []

    for i in range(0, len(titles_to_map), batch_size):
        batch = titles_to_map[i:i + batch_size]
        batch_id = f"batch_{i // batch_size}"
        user_prompt = build_user_prompt(batch)

        requests.append(
            Request(
                custom_id=batch_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=16384,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}]
                )
            )
        )

    return requests


def create_single_batch_job(
    client: anthropic.Anthropic,
    requests: list,
    job_index: int
) -> tuple[str, int]:
    """Create a single batch job and return its ID."""
    message_batch = client.messages.batches.create(requests=requests)
    logger.info(f"  Batch job {job_index + 1} created: {message_batch.id} ({len(requests)} requests)")
    return message_batch.id, len(requests)


def poll_batch_jobs(
    client: anthropic.Anthropic,
    batch_ids: list[str],
    poll_interval: int = 10
) -> None:
    """Poll multiple batch jobs until all complete."""
    pending_batches = set(batch_ids)

    while pending_batches:
        completed_this_round = []

        for batch_id in list(pending_batches):
            message_batch = client.messages.batches.retrieve(batch_id)
            status = message_batch.processing_status
            counts = message_batch.request_counts

            short_id = batch_id[-8:]  # Last 8 chars for readability
            logger.info(
                f"  ...{short_id}: {status} | "
                f"proc: {counts.processing}, ok: {counts.succeeded}, "
                f"err: {counts.errored}"
            )

            if status == "ended":
                completed_this_round.append(batch_id)

        for batch_id in completed_this_round:
            pending_batches.remove(batch_id)

        if pending_batches:
            logger.info(f"Waiting {poll_interval}s... ({len(pending_batches)} jobs remaining)")
            time.sleep(poll_interval)


def collect_batch_results(
    client: anthropic.Anthropic,
    batch_ids: list[str]
) -> tuple[list[dict], dict]:
    """Collect results from all completed batch jobs."""
    all_results = []
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    for batch_id in batch_ids:
        short_id = batch_id[-8:]

        for result in client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                message = result.result.message
                response_text = message.content[0].text.strip()

                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    response_text = "\n".join(lines[1:-1])

                try:
                    parsed = json.loads(response_text)
                    mappings = parsed.get("mappings", [])
                    all_results.extend(mappings)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error for {result.custom_id} in ...{short_id}: {e}")

                # Accumulate usage
                total_usage["input_tokens"] += message.usage.input_tokens
                total_usage["output_tokens"] += message.usage.output_tokens
                if hasattr(message.usage, "cache_creation_input_tokens"):
                    total_usage["cache_creation_input_tokens"] += message.usage.cache_creation_input_tokens or 0
                if hasattr(message.usage, "cache_read_input_tokens"):
                    total_usage["cache_read_input_tokens"] += message.usage.cache_read_input_tokens or 0

            elif result.result.type == "errored":
                logger.error(f"Request {result.custom_id} in ...{short_id} failed: {result.result.error}")
            elif result.result.type == "expired":
                logger.error(f"Request {result.custom_id} in ...{short_id} expired")

    return all_results, total_usage


def process_with_batch_api(
    client: anthropic.Anthropic,
    titles_to_map: list[dict],
    taxonomy_ref: str,
    model: str,
    batch_size: int,
    max_requests_per_batch_job: int = 1000
) -> tuple[list[dict], dict]:
    """Process job titles using the Message Batches API with parallel batch jobs.

    Args:
        client: Anthropic client
        titles_to_map: List of job titles to map
        taxonomy_ref: Taxonomy reference string
        model: Model name
        batch_size: Number of titles per API request
        max_requests_per_batch_job: Max requests per batch job (for parallelization)
    """
    # Create all individual requests (each request = batch_size titles)
    all_requests = create_batch_requests(titles_to_map, taxonomy_ref, model, batch_size)
    total_requests = len(all_requests)

    # Split requests into multiple batch jobs for parallel processing
    batch_jobs_requests = []
    for i in range(0, total_requests, max_requests_per_batch_job):
        batch_jobs_requests.append(all_requests[i:i + max_requests_per_batch_job])

    num_jobs = len(batch_jobs_requests)
    logger.info(f"Creating {num_jobs} parallel batch jobs ({total_requests} total requests)...")

    # Submit all batch jobs
    batch_ids = []
    for i, requests in enumerate(batch_jobs_requests):
        batch_id, count = create_single_batch_job(client, requests, i)
        batch_ids.append(batch_id)

    logger.info(f"All {num_jobs} batch jobs submitted, polling for completion...")

    # Poll all batch jobs until complete
    poll_batch_jobs(client, batch_ids)

    logger.info("All batch jobs complete, retrieving results...")

    # Collect results from all batch jobs
    all_results, total_usage = collect_batch_results(client, batch_ids)

    logger.info(f"Batch processing complete: {len(all_results)} titles mapped")
    return all_results, total_usage


def process_contacts(
    df_contacts: pd.DataFrame,
    df_taxonomy: pd.DataFrame,
    api_key: str,
    model: str,
    batch_size: int,
    use_structured_output: bool,
    use_batch_api: bool = False,
    max_requests_per_batch_job: int = 1000
) -> tuple[pd.DataFrame, dict]:
    """Process all contacts and map their job titles."""
    process_start = time.time()
    client = anthropic.Anthropic(api_key=api_key)
    taxonomy_ref = create_taxonomy_reference(df_taxonomy)

    # Extract and deduplicate job titles
    df_with_titles = df_contacts[df_contacts["job_title"].notna()].copy()
    df_with_titles["job_title_clean"] = df_with_titles["job_title"].astype(str).str.strip()
    df_with_titles = df_with_titles[df_with_titles["job_title_clean"] != ""]

    unique_titles = df_with_titles["job_title_clean"].unique().tolist()
    total_with_titles = len(df_with_titles)

    logger.info(f"Contacts with job titles: {total_with_titles:,}")
    logger.info(f"Unique job titles: {len(unique_titles):,} ({(1 - len(unique_titles)/total_with_titles)*100:.1f}% deduplication)")

    # Prepare unique titles for processing (use title as ID for dedup mapping)
    titles_to_map = [{"id": title, "job_title": title} for title in unique_titles]

    total_batches = (len(titles_to_map) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(titles_to_map):,} unique titles in {total_batches} batches of {batch_size}...")
    logger.info(f"Model: {model}")
    logger.info(f"Mode: {'Batch API (async, 50% cheaper)' if use_batch_api else 'Real-time API'}")
    if not use_batch_api:
        logger.info(f"Structured outputs: {'enabled' if use_structured_output and model in STRUCTURED_OUTPUT_MODELS else 'disabled'}")

    # Process based on mode
    if use_batch_api:
        # Use Message Batches API (async, 50% cheaper, parallel jobs)
        all_results, total_usage = process_with_batch_api(
            client, titles_to_map, taxonomy_ref, model, batch_size,
            max_requests_per_batch_job=max_requests_per_batch_job
        )
    else:
        # Use real-time API with progress tracking
        all_results = []
        total_batches = (len(titles_to_map) + batch_size - 1) // batch_size

        # Track usage stats
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        start_time = time.time()
        log_interval = max(1, total_batches // 20)  # Log ~20 times during processing

        for i in range(0, len(titles_to_map), batch_size):
            batch = titles_to_map[i:i + batch_size]
            batch_num = i // batch_size + 1

            results, usage_stats = map_titles_batch(client, batch, taxonomy_ref, model, use_structured_output)
            all_results.extend(results)

            # Accumulate usage stats
            for key in total_usage:
                total_usage[key] += usage_stats.get(key, 0)

            # Progress logging
            if batch_num == 1 or batch_num % log_interval == 0 or batch_num == total_batches:
                elapsed = time.time() - start_time
                titles_done = min(batch_num * batch_size, len(titles_to_map))
                titles_per_sec = titles_done / elapsed if elapsed > 0 else 0
                remaining_titles = len(titles_to_map) - titles_done
                eta_seconds = remaining_titles / titles_per_sec if titles_per_sec > 0 else 0
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)

                cache_status = "CACHE HIT" if usage_stats.get("cache_read_input_tokens", 0) > 0 else "CACHE MISS"
                pct = (batch_num / total_batches) * 100

                logger.info(
                    f"Batch {batch_num}/{total_batches} ({pct:.0f}%) | "
                    f"{titles_done:,}/{len(titles_to_map):,} titles | "
                    f"{titles_per_sec:.1f} titles/sec | "
                    f"ETA: {eta_min}m {eta_sec}s | {cache_status}"
                )

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)

    if df_results.empty:
        logger.warning("No results returned from API")
        return pd.DataFrame(), total_usage

    # Create mapping from job_title to standardized fields
    # The 'id' field contains the original job_title (used for dedup)
    title_mapping = df_results.set_index("id")[["standardized_job_title", "department", "seniority"]].to_dict("index")

    # Build output dataframe
    df_output = df_contacts[["contact_id", "job_title", "email", "contact_type"]].copy()

    # Map each contact's job_title to standardized fields
    def map_title(job_title):
        if pd.isna(job_title):
            return pd.Series({"standardized_job_title": None, "department": None, "seniority": -1})
        clean_title = str(job_title).strip()
        if clean_title in title_mapping:
            return pd.Series(title_mapping[clean_title])
        return pd.Series({"standardized_job_title": None, "department": None, "seniority": -1})

    mapped = df_output["job_title"].apply(map_title)
    df_output = pd.concat([df_output, mapped], axis=1)

    # Ensure seniority is numeric
    df_output["seniority"] = pd.to_numeric(df_output["seniority"], errors="coerce").fillna(-1)

    # Add timing to usage stats
    total_usage["elapsed_seconds"] = time.time() - process_start

    return df_output, total_usage


def main():
    """Main entry point for Keboola component."""
    ci = CommonInterface()
    params = ci.configuration.parameters

    # Get parameters with defaults
    api_key = params.get("#ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing required parameter: #ANTHROPIC_API_KEY")

    model = params.get("model", "claude-sonnet-4-5")
    batch_size = int(params.get("batch_size", 200))
    use_structured_output = params.get("use_structured_output", True)
    use_batch_api = params.get("use_batch_api", False)  # Use Message Batches API (async, 50% cheaper)
    max_requests_per_batch_job = int(params.get("max_requests_per_batch_job", 3))  # Parallel batch jobs
    limit = params.get("limit", 1000)  # Limit records for testing, 0 = no limit

    logger.info("=== Job Title Mapping Component ===")
    logger.info(f"Model: {model}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Batch API: {'enabled (async, 50% cheaper)' if use_batch_api else 'disabled (real-time)'}")
    if use_batch_api:
        logger.info(f"Max requests per batch job: {max_requests_per_batch_job}")
    logger.info(f"Structured outputs: {use_structured_output}")
    logger.info(f"Record limit: {limit if limit else 'unlimited'}")

    # Fixed input paths (Table Input Mapping)
    contacts_path = Path(ci.data_folder_path) / "in/tables/contact.csv"
    taxonomy_path = Path(ci.data_folder_path) / "in/tables/standardized_job_titles.csv"

    if not contacts_path.exists():
        raise FileNotFoundError(f"Contacts file not found: {contacts_path}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    # Load data
    df_contacts = pd.read_csv(contacts_path, low_memory=False)
    df_taxonomy = pd.read_csv(taxonomy_path)

    total_contacts = len(df_contacts)
    logger.info(f"Loaded {total_contacts:,} contacts and {len(df_taxonomy)} taxonomy entries")
    logger.info(f"Contact columns: {list(df_contacts.columns)}")
    logger.info(f"Taxonomy columns: {list(df_taxonomy.columns)}")

    # Validate required columns
    required_contact_cols = ["contact_id", "job_title", "email", "contact_type"]
    missing_cols = [col for col in required_contact_cols if col not in df_contacts.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in contact.csv: {missing_cols}. Available: {list(df_contacts.columns)}")

    # Apply limit if set
    if limit and limit > 0:
        df_contacts = df_contacts.head(limit)
        logger.info(f"Limited to {len(df_contacts):,} contacts (limit={limit})")

    # Process contacts
    df_output, usage_stats = process_contacts(
        df_contacts,
        df_taxonomy,
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        use_structured_output=use_structured_output,
        use_batch_api=use_batch_api,
        max_requests_per_batch_job=max_requests_per_batch_job
    )

    if df_output.empty:
        logger.warning("No results to save")
        return

    # Write output to fixed path
    output_path = Path(ci.data_folder_path) / "out/tables/contact_taxonomy_seniority.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    logger.info(f"Results saved: {len(df_output)} contacts mapped")
    logger.info(f"Output: {output_path}")

    # Log summary
    elapsed = usage_stats.get('elapsed_seconds', 0)
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)

    logger.info("=== Processing Summary ===")
    logger.info(f"Total time: {elapsed_min}m {elapsed_sec}s")
    logger.info(f"Contacts processed: {len(df_output):,}")
    logger.info(f"Throughput: {len(df_output) / elapsed:.1f} contacts/sec" if elapsed > 0 else "N/A")

    logger.info("=== Token Usage ===")
    logger.info(f"Input tokens: {usage_stats['input_tokens']:,}")
    logger.info(f"Output tokens: {usage_stats['output_tokens']:,}")
    total_tokens = usage_stats['input_tokens'] + usage_stats['output_tokens']
    logger.info(f"Total tokens: {total_tokens:,}")

    # Cache stats
    cache_read = usage_stats['cache_read_input_tokens']
    cache_creation = usage_stats['cache_creation_input_tokens']
    logger.info(f"Cache creation tokens: {cache_creation:,}")
    logger.info(f"Cache read tokens: {cache_read:,}")
    if cache_read > 0:
        cache_hit_rate = (cache_read / (cache_read + cache_creation)) * 100
        logger.info(f"Cache hit rate: {cache_hit_rate:.1f}%")


if __name__ == "__main__":
    main()
