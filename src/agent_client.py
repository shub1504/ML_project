"""
MaternalRisk AI - Azure AI Foundry Agent Client
================================================
Client module to communicate with the hosted Azure AI Foundry Agent via the
Microsoft Agent Framework Responses API protocol (POST /responses).

Safety and Clinical Governance:
- SVM model executions must occur exclusively via the agent's registered
  'assess_maternal_risk' tool.
- Output from the SVM is an experimental research classification, not a clinical diagnosis.
- All secrets and credentials remain server-side; client communicates via local or
  configured host URL using environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Environment & Configuration Resolution
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Attempt to load .env from project root or agent directory without hard dependency
def _load_env_fallback():
    """Load key-value pairs from .env files into os.environ if not already set."""
    env_candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "agent-framework-agent-with-local-tools-responses" / "src" / "agent-framework-agent-with-local-tools-responses" / ".env",
        PROJECT_ROOT / "agent-framework-agent-with-local-tools-responses" / ".azure" / "agent-framework-agent-with-local-tools-responses-dev" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.debug("Could not read %s: %s", env_path, e)

_load_env_fallback()

DEFAULT_HOST_URL = "http://localhost:8088"
DEFAULT_TIMEOUT = 60  # seconds


class FoundryAgentClient:
    """Client for interacting with the Azure AI Foundry Responses Host Server."""

    def __init__(self, host_url: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the agent client.

        Parameters:
            host_url (str | None): Base URL of the agent server. If omitted,
                                   reads AGENT_HOST_URL from environment,
                                   defaulting to http://localhost:8088.
            timeout (int): Default HTTP request timeout in seconds.
        """
        raw_url = host_url or os.environ.get("AGENT_HOST_URL") or DEFAULT_HOST_URL
        # Normalize host URL: strip trailing slash and '/responses' if provided
        clean_url = raw_url.rstrip("/")
        if clean_url.endswith("/responses"):
            clean_url = clean_url[:-10]
        self.host_url = clean_url
        self.endpoint = f"{self.host_url}/responses"
        self.timeout = timeout

    def check_health(self, timeout: int = 3) -> tuple[bool, str]:
        """
        Check whether the local or remote agent server is reachable.

        Returns:
            tuple[bool, str]: (is_healthy, status_message)
        """
        try:
            # ResponsesHostServer handles POST /responses; a GET to / or /responses
            # returns 404 or 405 when active, which confirms the server process is alive.
            response = requests.get(self.endpoint, timeout=timeout)
            if response.status_code in (200, 400, 404, 405):
                return True, f"Agent server reachable at {self.host_url}"
            return True, f"Agent server responded with HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Could not connect to agent server at {self.host_url}. Server is offline."
        except requests.exceptions.Timeout:
            return False, f"Connection to agent server at {self.host_url} timed out."
        except Exception as exc:
            return False, f"Agent health check error: {exc}"

    def send_chat_message(
        self,
        messages: list[dict[str, str]],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Send conversation history to the agent and receive assistant reply.

        Parameters:
            messages: List of conversation dicts with 'role' ('user'|'assistant')
                      and 'content' (str).
            timeout: Optional override for request timeout in seconds.

        Returns:
            dict containing:
                - success (bool): True if communication succeeded.
                - text (str): Assistant message response text.
                - tool_calls (list[dict]): Details of any tools executed (e.g. assess_maternal_risk).
                - raw (dict | None): Raw server response payload.
                - error (str | None): Error description if success is False.
                - error_type (str | None): Category of error for UX display.
        """
        if not messages:
            return {
                "success": False,
                "text": "",
                "tool_calls": [],
                "raw": None,
                "error": "No messages provided to send.",
                "error_type": "validation",
            }

        # Format messages into the Responses protocol input list
        formatted_input: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Canonicalize role
            if role not in ("user", "assistant", "system"):
                role = "user"
            formatted_input.append({
                "type": "message",
                "role": role,
                "content": content,
            })

        payload = {
            "input": formatted_input,
            "stream": False,
        }

        req_timeout = timeout or self.timeout

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=req_timeout,
            )

            if response.status_code != 200:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text or f"HTTP {response.status_code}"
                return {
                    "success": False,
                    "text": "",
                    "tool_calls": [],
                    "raw": None,
                    "error": f"Agent server returned HTTP {response.status_code}: {err_msg}",
                    "error_type": "server_error",
                }

            data = response.json()
            return self._parse_agent_response(data)

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "text": "",
                "tool_calls": [],
                "raw": None,
                "error": (
                    f"Unable to connect to the Azure AI Foundry agent at {self.host_url}.\n\n"
                    "Please ensure the local agent server is running. You can start it with:\n"
                    "```bash\n"
                    "cd agent-framework-agent-with-local-tools-responses/src/agent-framework-agent-with-local-tools-responses\n"
                    "uv run --no-sync python main.py\n"
                    "```"
                ),
                "error_type": "connection",
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "text": "",
                "tool_calls": [],
                "raw": None,
                "error": f"Agent request timed out after {req_timeout} seconds. The model or tool execution took too long.",
                "error_type": "timeout",
            }

        except Exception as exc:
            return {
                "success": False,
                "text": "",
                "tool_calls": [],
                "raw": None,
                "error": f"Unexpected communication error: {exc}",
                "error_type": "unexpected",
            }

    @staticmethod
    def _parse_agent_response(data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the JSON response payload from the ResponsesHostServer.

        Extracts:
        - Assistant output text
        - Executed tool calls and their outputs (matching function_call with function_call_output)
        """
        output_items = data.get("output", [])
        text_chunks: list[str] = []
        function_calls_map: dict[str, dict[str, Any]] = {}

        for item in output_items:
            item_type = item.get("type")

            # 1. Match function_call
            if item_type == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name", "")
                raw_args = item.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = raw_args
                else:
                    args = raw_args

                function_calls_map[call_id] = {
                    "call_id": call_id,
                    "name": name,
                    "arguments": args,
                    "output": None,
                }

            # 2. Match function_call_output
            elif item_type == "function_call_output":
                call_id = item.get("call_id")
                output_val = item.get("output", "")
                if call_id and call_id in function_calls_map:
                    function_calls_map[call_id]["output"] = output_val
                else:
                    # Stash anyway if function_call wasn't mapped
                    function_calls_map[call_id or "unknown"] = {
                        "call_id": call_id,
                        "name": "unknown_tool",
                        "arguments": {},
                        "output": output_val,
                    }

            # 3. Extract assistant text from message item
            elif item_type == "message" and item.get("role") == "assistant":
                contents = item.get("content", [])
                for content in contents:
                    if isinstance(content, dict):
                        c_type = content.get("type")
                        if c_type in ("output_text", "text"):
                            text_chunks.append(content.get("text", ""))
                    elif isinstance(content, str):
                        text_chunks.append(content)

        assistant_text = "".join(text_chunks).strip()
        tool_calls_list = list(function_calls_map.values())

        return {
            "success": True,
            "text": assistant_text,
            "tool_calls": tool_calls_list,
            "raw": data,
            "error": None,
            "error_type": None,
        }
