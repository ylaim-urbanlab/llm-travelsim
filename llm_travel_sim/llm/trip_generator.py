"""
Call OpenAI API to generate trips, parse and validate output.
"""
from __future__ import annotations
import json
import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI, RateLimitError, APIStatusError

from llm_travel_sim.llm.cache import LLMCache


@dataclass
class Trip:
    origin: str
    destination: str
    mode: str
    purpose: str
    departure_time: str
    journal: str
    weight: float = 1.0
    hex_context_time: str = ""


VALID_MODES = {"driving", "transit", "cycling", "walking", "rideshare"}
VALID_PURPOSES = {"work", "shopping", "school", "recreation", "dining",
                  "medical", "social", "return_home", "other"}


def _validate_trip(raw: dict, valid_hex_ids: set) -> Optional[dict]:
    """Validate and coerce a raw trip dict."""
    if not isinstance(raw, dict):
        return None
    mode = raw.get("mode", "driving").lower().strip()
    purpose = raw.get("purpose", "other").lower()
    origin = raw.get("origin", "")
    destination = raw.get("destination", "")

    # Normalize common LLM variants to our canonical mode
    if mode in ("bike", "bicycle", "biking"):
        mode = "cycling"
    if mode not in VALID_MODES:
        mode = "driving"
    if purpose not in VALID_PURPOSES:
        purpose = "other"

    if destination not in valid_hex_ids and origin in valid_hex_ids:
        destination = origin  # stay in place if destination unknown

    return {
        "origin": str(origin),
        "destination": str(destination) if destination else str(origin),
        "mode": mode,
        "purpose": purpose,
        "departure_time": str(raw.get("departure_time", "00:00")),
        "journal": str(raw.get("journal", ""))[:500],
    }


def _extract_json_array(text: str) -> Optional[List[dict]]:
    """Robustly extract the first JSON array from LLM output."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    objects = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
    if objects:
        parsed = []
        for obj_str in objects:
            try:
                parsed.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if parsed:
            return parsed

    return None


class TripGenerator:
    def __init__(self, model: str, max_tokens: int, temperature: float,
                 max_retries: int = 3, cache: Optional[LLMCache] = None):
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache = cache
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(
        self,
        prompt: str,
        hex_id: str,
        timestep_str: str,
        valid_hex_ids: set,
        importance_weights: List[float],
        cluster_memberships: List[float],
        context: Optional[dict] = None,
    ) -> List[Trip]:
        """Call LLM, parse trips, apply importance weights."""
        # --- Cache lookup ---
        if self.cache is not None and context is not None:
            cached = self.cache.get(prompt, context, cluster_memberships)
            if cached is not None:
                # Re-stamp origin/destination to this hex and timestep
                trips = []
                dominant_cluster_idx = int(max(range(len(cluster_memberships)),
                                               key=lambda i: cluster_memberships[i]))
                base_weight = float(importance_weights[dominant_cluster_idx]) if importance_weights else 1.0
                for raw in cached:
                    raw = dict(raw)
                    raw["origin"] = hex_id
                    if raw.get("destination") not in valid_hex_ids:
                        raw["destination"] = hex_id
                    trips.append(Trip(**raw, weight=base_weight, hex_context_time=timestep_str))
                return trips

        raw_trips = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                content = response.choices[0].message.content
                raw_trips = _extract_json_array(content)
                if raw_trips is not None:
                    break
                print(f"[trip_gen] JSON parse failed for hex {hex_id}, attempt {attempt+1}")
            except RateLimitError:
                wait = 2 ** attempt * 5
                print(f"[trip_gen] Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                print(f"[trip_gen] API error: {e}, attempt {attempt+1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[trip_gen] Unexpected error: {e}, attempt {attempt+1}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        if raw_trips is None:
            print(f"[trip_gen] All retries failed for hex {hex_id} at {timestep_str}, using empty trip list")
            return []

        trips = []
        dominant_cluster_idx = int(max(range(len(cluster_memberships)),
                                       key=lambda i: cluster_memberships[i]))
        base_weight = float(importance_weights[dominant_cluster_idx]) if importance_weights else 1.0

        for raw in raw_trips:
            validated = _validate_trip(raw, valid_hex_ids)
            if validated:
                trips.append(Trip(
                    **validated,
                    weight=base_weight,
                    hex_context_time=timestep_str,
                ))

        # --- Cache store ---
        if self.cache is not None and context is not None and trips:
            # Store without weight/hex_context_time so cached entries are reusable
            storable = [
                {k: v for k, v in vars(t).items() if k not in ("weight", "hex_context_time")}
                for t in trips
            ]
            self.cache.put(prompt, context, cluster_memberships, storable)

        return trips
