# RevitMCP: This script runs in a standard CPython 3.7+ environment. Modern Python syntax is expected.
"""
External Flask server for RevitMCP.
This server will handle requests from the Revit UI (via a listener)
and can also host a web UI for direct interaction.
"""

import os
import sys # Ensure sys is imported for stdout/stderr redirection if used
import logging
import traceback # For detailed exception logging
import json
import uuid
import re
import difflib
import threading
from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS

# LLM Libraries
import openai
import anthropic
import google.generativeai as genai
from google.generativeai import types as google_types

# MCP SDK Import
from mcp.server.fastmcp import FastMCP

# --- Centralized Logging Configuration ---
USER_DOCUMENTS = os.path.expanduser("~/Documents")
LOG_BASE_DIR = os.path.join(USER_DOCUMENTS, 'RevitMCP', 'server_logs')
if not os.path.exists(LOG_BASE_DIR):
    os.makedirs(LOG_BASE_DIR)

STARTUP_LOG_FILE = os.path.join(LOG_BASE_DIR, 'server_startup_error.log')
APP_LOG_FILE = os.path.join(LOG_BASE_DIR, 'server_app.log')

startup_logger = logging.getLogger('RevitMCPServerStartup')
startup_logger.setLevel(logging.DEBUG)
if os.path.exists(STARTUP_LOG_FILE):
    try:
        os.remove(STARTUP_LOG_FILE)
    except Exception:
        pass
startup_file_handler = logging.FileHandler(STARTUP_LOG_FILE, encoding='utf-8')
startup_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
startup_logger.addHandler(startup_file_handler)
startup_logger.info("--- Server script attempting to start ---")

def configure_flask_logger(app_instance, debug_mode):
    file_handler = logging.FileHandler(APP_LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    for handler in list(app_instance.logger.handlers):
        app_instance.logger.removeHandler(handler)
    app_instance.logger.addHandler(file_handler)
    if debug_mode:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        app_instance.logger.addHandler(console_handler)
        app_instance.logger.setLevel(logging.DEBUG)
        app_instance.logger.info("Flask app logger: Configured for DEBUG mode (file and console).")
    else:
        app_instance.logger.setLevel(logging.INFO)
        app_instance.logger.info("Flask app logger: Configured for INFO mode (file only).")
# --- End Centralized Logging Configuration ---

try:
    startup_logger.info("--- RevitMCP External Server script starting (inside main try block) ---")
    print("--- RevitMCP External Server script starting (Python print) ---")

    app = Flask(__name__, template_folder='templates', static_folder='static')
    # Default to local-only and explicit localhost origins for safer out-of-the-box behavior.
    default_cors_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
    cors_origins_raw = os.environ.get('FLASK_CORS_ORIGINS', '')
    if cors_origins_raw.strip():
        cors_origins = [origin.strip() for origin in cors_origins_raw.split(',') if origin.strip()]
    else:
        cors_origins = default_cors_origins
    CORS(app, resources={r"/*": {"origins": cors_origins}})

    DEBUG_MODE = os.environ.get('FLASK_DEBUG_MODE', 'False').lower() == 'true'
    PORT = int(os.environ.get('FLASK_PORT', 8000))
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')

    def resolve_runtime_surface(argv_values):
        """Resolve runtime surface from CLI args or environment."""
        valid_surfaces = ('web', 'mcp')
        cli_surface = None

        for i, arg in enumerate(argv_values):
            if arg.startswith('--surface='):
                cli_surface = arg.split('=', 1)[1].strip().lower()
                break
            if arg == '--surface' and i + 1 < len(argv_values):
                cli_surface = argv_values[i + 1].strip().lower()
                break

        env_surface = os.environ.get('REVITMCP_SURFACE', '').strip().lower()
        requested_surface = cli_surface or env_surface or 'web'

        if requested_surface not in valid_surfaces:
            startup_logger.warning(
                "Invalid runtime surface '%s'. Falling back to 'web'. Valid values: %s",
                requested_surface, ', '.join(valid_surfaces)
            )
            return 'web'

        return requested_surface
    
    configure_flask_logger(app, DEBUG_MODE)
    app.logger.info(
        "Flask app initialized. Debug mode: %s. Host: %s. Port: %s. CORS origins: %s.",
        DEBUG_MODE, HOST, PORT, cors_origins
    )
    print(f"--- Flask DEBUG_MODE is set to: {DEBUG_MODE} (from print) ---")

    # --- MCP Server Instance ---
    mcp_server = FastMCP("RevitMCPServer")
    app.logger.info("FastMCP server instance created: %s", mcp_server.name)

    # --- Session Storage for Element IDs ---
    # Keep full result sets server-side and return compact summaries to MCP clients.
    element_storage = {}        # legacy/category lookup: {"windows": {...}}
    result_handle_storage = {}  # primary lookup: {"res_xxxxx": {...}}

    MAX_ELEMENTS_FOR_SELECTION = 250
    MAX_ELEMENTS_FOR_PROPERTY_READ = int(os.environ.get("REVITMCP_MAX_ELEMENTS_FOR_PROPERTY_READ", "300"))
    DEFAULT_SERVER_FILTER_BATCH_SIZE = int(os.environ.get("REVITMCP_SERVER_FILTER_BATCH_SIZE", "600"))
    MAX_ELEMENTS_IN_RESPONSE = int(os.environ.get("REVITMCP_MAX_ELEMENTS_IN_RESPONSE", "40"))
    MAX_RECORDS_IN_RESPONSE = int(os.environ.get("REVITMCP_MAX_RECORDS_IN_RESPONSE", "20"))
    MIN_CONFIDENCE_FOR_PARAMETER_REMAP = float(
        os.environ.get("REVITMCP_MIN_CONFIDENCE_FOR_PARAMETER_REMAP", "0.82")
    )

    def _now_timestamp():
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _normalize_storage_key(name: str) -> str:
        return str(name or "").lower().replace("ost_", "").replace(" ", "_")

    def _new_result_handle() -> str:
        return "res_{}".format(uuid.uuid4().hex[:12])

    def store_elements(category_name: str, element_ids: list, count: int) -> tuple[str, str]:
        """Store element IDs and return (storage_key, result_handle)."""
        timestamp = _now_timestamp()
        storage_key = _normalize_storage_key(category_name)
        normalized_ids = [str(eid) for eid in (element_ids or [])]
        result_handle = _new_result_handle()

        record = {
            "element_ids": normalized_ids,
            "count": int(count if count is not None else len(normalized_ids)),
            "category": category_name,
            "timestamp": timestamp,
            "storage_key": storage_key,
            "result_handle": result_handle
        }

        element_storage[storage_key] = record
        result_handle_storage[result_handle] = record
        app.logger.info(
            "Stored %s element IDs for category '%s' with key '%s' and handle '%s'",
            record["count"], category_name, storage_key, result_handle
        )
        return storage_key, result_handle

    def get_result_by_handle(result_handle: str) -> dict:
        if not result_handle:
            return None
        return result_handle_storage.get(str(result_handle).strip())

    def get_stored_elements(storage_key: str) -> dict:
        """Retrieve stored elements by category key or by result_handle."""
        key = str(storage_key or "").strip()
        if not key:
            return None
        if key.startswith("res_"):
            return get_result_by_handle(key)
        return element_storage.get(_normalize_storage_key(key))

    def list_stored_categories() -> dict:
        """List all currently stored categories (compact metadata only)."""
        return {
            key: {
                "category": data["category"],
                "count": data["count"],
                "timestamp": data["timestamp"],
                "result_handle": data.get("result_handle")
            }
            for key, data in element_storage.items()
        }

    def resolve_element_ids(element_ids=None, result_handle: str = None, category_name: str = None):
        """Resolve IDs from direct list, handle, or stored category key."""
        if element_ids:
            return [str(eid) for eid in element_ids], None, None

        if result_handle:
            record = get_result_by_handle(result_handle)
            if not record:
                return None, None, {"status": "error", "message": "Unknown result_handle '{}'".format(result_handle)}
            return list(record.get("element_ids", [])), record, None

        if category_name:
            record = get_stored_elements(category_name)
            if not record:
                return None, None, {"status": "error", "message": "No stored elements found for '{}'".format(category_name)}
            return list(record.get("element_ids", [])), record, None

        return None, None, {"status": "error", "message": "No elements were provided. Use element_ids, result_handle, or category_name."}

    def compact_result_payload(result: dict, preserve_keys: list = None) -> dict:
        """Avoid sending huge arrays back through MCP responses."""
        if not isinstance(result, dict):
            return result

        preserve_keys = preserve_keys or []
        compact = dict(result)

        if isinstance(compact.get("element_ids"), list):
            ids = compact.get("element_ids", [])
            if len(ids) > MAX_ELEMENTS_IN_RESPONSE:
                compact["element_ids_sample"] = ids[:MAX_ELEMENTS_IN_RESPONSE]
                compact["element_ids_truncated"] = True
                compact["element_ids_total"] = len(ids)
                compact.pop("element_ids", None)

        if isinstance(compact.get("elements"), list):
            records = compact.get("elements", [])
            if len(records) > MAX_RECORDS_IN_RESPONSE:
                compact["elements_sample"] = records[:MAX_RECORDS_IN_RESPONSE]
                compact["elements_truncated"] = True
                compact["elements_total"] = len(records)
                compact.pop("elements", None)

        if isinstance(compact.get("data"), dict):
            data_dict = dict(compact["data"])
            if isinstance(data_dict.get("selected_ids_processed"), list):
                selected_ids = data_dict.get("selected_ids_processed", [])
                if len(selected_ids) > MAX_ELEMENTS_IN_RESPONSE:
                    data_dict["selected_ids_sample"] = selected_ids[:MAX_ELEMENTS_IN_RESPONSE]
                    data_dict["selected_ids_truncated"] = True
                    data_dict["selected_ids_total"] = len(selected_ids)
                    data_dict.pop("selected_ids_processed", None)
            compact["data"] = data_dict

        if isinstance(compact.get("message"), str) and len(compact["message"]) > 1200:
            compact["message"] = compact["message"][:1200] + "... [truncated]"
            compact["message_truncated"] = True

        for key in preserve_keys:
            if key in result:
                compact[key] = result[key]

        return compact

    # --- Revit MCP API Communication ---
    # Auto-detect which port the Revit MCP API is running on
    REVIT_MCP_API_BASE_URL = None
    POSSIBLE_PORTS = [48885, 48884, 48886]  # Common ports used by pyRevit

    def detect_revit_mcp_port():
        """Detect which port the Revit MCP API is running on by trying common ports."""
        global REVIT_MCP_API_BASE_URL
        
        for port in POSSIBLE_PORTS:
            test_url = f"http://localhost:{port}/revit-mcp-v1"
            try:
                # Try a simple connection test - just check if the port responds
                response = requests.get(f"{test_url}/project_info", timeout=2)
                if response.status_code in [200, 404, 405]:  # Any response means server is running
                    REVIT_MCP_API_BASE_URL = test_url
                    startup_logger.info(f"Detected Revit MCP API running on port {port}")
                    print(f"--- Detected Revit MCP API on port {port} ---")
                    return True
            except requests.exceptions.RequestException:
                # Port not responding, try next one
                continue
        
        startup_logger.warning("Could not detect Revit MCP API on any common ports. Defaulting to 48884.")
        print("--- Warning: Could not detect Revit MCP API port, defaulting to 48884 ---")
        REVIT_MCP_API_BASE_URL = "http://localhost:48884/revit-mcp-v1"
        return False

    # Detect the correct port at startup
    detect_revit_mcp_port()

    # --- Tool Name Constants (used for REVIT_TOOLS_SPEC and dispatch) ---
    REVIT_INFO_TOOL_NAME = "get_revit_project_info"
    GET_ELEMENTS_BY_CATEGORY_TOOL_NAME = "get_elements_by_category"
    SELECT_ELEMENTS_TOOL_NAME = "select_elements_by_id"
    SELECT_STORED_ELEMENTS_TOOL_NAME = "select_stored_elements"
    LIST_STORED_ELEMENTS_TOOL_NAME = "list_stored_elements"
    FILTER_ELEMENTS_TOOL_NAME = "filter_elements"
    FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME = "filter_stored_elements_by_parameter"
    GET_ELEMENT_PROPERTIES_TOOL_NAME = "get_element_properties"
    UPDATE_ELEMENT_PARAMETERS_TOOL_NAME = "update_element_parameters"
    GET_SCHEMA_CONTEXT_TOOL_NAME = "get_revit_schema_context"
    RESOLVE_TARGETS_TOOL_NAME = "resolve_revit_targets"
    
    # Sheet and view management tools
    PLACE_VIEW_ON_SHEET_TOOL_NAME = "place_view_on_sheet"
    LIST_VIEWS_TOOL_NAME = "list_views"

    # Replace batch workflow with generic planner
    PLANNER_TOOL_NAME = "plan_and_execute_workflow"

    schema_context_cache = {
        "doc_fingerprint": None,
        "context": None
    }

    # --- Helper function to call Revit Listener (remains mostly the same) ---
    # This function is now central for all Revit interactions triggered by MCP tools.
    def call_revit_listener(command_path: str, method: str = 'POST', payload_data: dict = None):
        global REVIT_MCP_API_BASE_URL
        logger_instance = app.logger # Correctly uses app.logger for this function scope

        # Try to discover the API URL if not already set (Now REVIT_MCP_API_BASE_URL is set directly)
        if REVIT_MCP_API_BASE_URL is None: # This condition should ideally not be met anymore
            logger_instance.error("Revit MCP API base URL is not set. Attempting auto-detection...")
            if not detect_revit_mcp_port():
                logger_instance.error("Failed to detect Revit MCP API port. Listener might not be running or accessible.")
                return {"status": "error", "message": "Could not connect to Revit Listener: API URL not configured."}
            
        logger_instance.info(f"Using pre-configured Revit MCP API base URL: {REVIT_MCP_API_BASE_URL}")

        def attempt_api_call():
            """Attempt the actual API call with current URL."""
            full_url = REVIT_MCP_API_BASE_URL.rstrip('/') + "/" + command_path.lstrip('/')
            logger_instance.debug(f"Calling Revit MCP API: {method} {full_url} with payload: {payload_data}")

            if method.upper() == 'POST':
                listener_response = requests.post(
                    full_url, 
                    json=payload_data, 
                    headers={'Content-Type': 'application/json'},
                    timeout=60 
                )
            elif method.upper() == 'GET':
                listener_response = requests.get(
                    full_url, 
                    params=payload_data, # GET requests use params for payload
                    timeout=60
                )
            else:
                logger_instance.error(f"Unsupported HTTP method: {method} for call_revit_listener")
                raise ValueError(f"Unsupported HTTP method: {method}")

            listener_response.raise_for_status()
            return listener_response.json()

        # First attempt
        try:
            response_json = attempt_api_call()
            logger_instance.info(f"Revit MCP API success for {command_path}: {response_json}")
            return response_json
        except requests.exceptions.ConnectionError as conn_err:
            logger_instance.warning(f"Connection failed to {REVIT_MCP_API_BASE_URL}. Attempting to re-detect port...")
            
            # Try to re-detect the port
            old_url = REVIT_MCP_API_BASE_URL
            if detect_revit_mcp_port() and REVIT_MCP_API_BASE_URL != old_url:
                logger_instance.info(f"Port re-detected. Retrying with new URL: {REVIT_MCP_API_BASE_URL}")
                try:
                    response_json = attempt_api_call()
                    logger_instance.info(f"Revit MCP API success after retry for {command_path}: {response_json}")
                    return response_json
                except Exception as retry_err:
                    logger_instance.error(f"Retry failed: {retry_err}")
            
            # If re-detection didn't help or failed
            msg = f"Could not connect to the Revit MCP API for command {command_path}. Tried {old_url} and {REVIT_MCP_API_BASE_URL}"
            logger_instance.error(msg)
            return {"status": "error", "message": msg}
        except requests.exceptions.Timeout:
            msg = f"Request to Revit MCP API at {REVIT_MCP_API_BASE_URL} for command {command_path} timed out."
            logger_instance.error(msg)
            return {"status": "error", "message": msg}
        except requests.exceptions.RequestException as e_req:
            msg_prefix = f"Error communicating with Revit MCP API at {REVIT_MCP_API_BASE_URL} for {command_path}"
            if hasattr(e_req, 'response') and e_req.response is not None:
                status_code = e_req.response.status_code
                try:
                    listener_err_data = e_req.response.json()
                    listener_message = str(listener_err_data.get('message', listener_err_data.get('error', 'Unknown API error')))
                    route_exception_message = ""
                    if isinstance(listener_err_data.get('exception'), dict):
                        route_exception_message = str(listener_err_data['exception'].get('message', ''))
                    is_missing_route = "RouteHandlerNotDefinedException" in listener_message or "RouteHandlerNotDefinedException" in route_exception_message
                    full_msg = f"{msg_prefix}: HTTP {status_code}. API Response: {listener_err_data.get('message', listener_err_data.get('error', 'Unknown API error'))}"
                    logger_instance.error(full_msg, exc_info=False) # No need for exc_info if we have API message
                    result = {"status": "error", "message": full_msg, "details": listener_err_data}
                    if is_missing_route:
                        result["error_type"] = "route_not_defined"
                        result["missing_route"] = command_path
                    return result
                except ValueError:
                    full_msg = f"{msg_prefix}: HTTP {status_code}. Response: {e_req.response.text[:200]}"
                    logger_instance.error(full_msg, exc_info=True)
                    return {"status": "error", "message": full_msg}
            else:
                logger_instance.error(f"{msg_prefix}: {e_req}", exc_info=True)
                return {"status": "error", "message": f"{msg_prefix}: {e_req}"}
        except Exception as e_gen:
            logger_instance.error(f"Unexpected error in call_revit_listener for {command_path} at {REVIT_MCP_API_BASE_URL}: {e_gen}", exc_info=True)
            return {"status": "error", "message": f"Unexpected error processing API response for {command_path}."}

    def _is_route_not_defined(result: dict, route_hint: str = None) -> bool:
        """Return True when a tool result represents a missing pyRevit route."""
        if not isinstance(result, dict):
            return False
        if result.get("error_type") == "route_not_defined":
            return True
        text_parts = [str(result.get("message", "")), str(result.get("details", ""))]
        if route_hint:
            text_parts.append(str(route_hint))
        text = " ".join(text_parts)
        return "RouteHandlerNotDefinedException" in text

    def _normalize_label(value: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', str(value or "").lower())

    def _best_match(term: str, candidates: list[str], fuzzy_cutoff: float = 0.5):
        if not term or not candidates:
            return None, [], 0.0
        term = str(term).strip()
        normalized_term = _normalize_label(term)
        normalized_map = {}
        for c in candidates:
            normalized_map.setdefault(_normalize_label(c), []).append(c)

        # Exact normalized match first
        if normalized_term in normalized_map:
            choice = normalized_map[normalized_term][0]
            return choice, [], 1.0

        # Substring heuristic
        contains_matches = [c for c in candidates if normalized_term and normalized_term in _normalize_label(c)]
        if contains_matches:
            primary = contains_matches[0]
            alternatives = contains_matches[1:6]
            return primary, alternatives, 0.85

        # Fuzzy fallback
        fuzzy = difflib.get_close_matches(term, candidates, n=6, cutoff=fuzzy_cutoff)
        if fuzzy:
            primary = fuzzy[0]
            alternatives = fuzzy[1:6]
            score = difflib.SequenceMatcher(None, term.lower(), primary.lower()).ratio()
            return primary, alternatives, score

        return None, [], 0.0

    def get_revit_schema_context_mcp_tool(force_refresh: bool = False) -> dict:
        """Fetches canonical Revit schema context (categories, levels, families, types, parameters)."""
        app.logger.info("MCP Tool executed: %s (force_refresh=%s)", GET_SCHEMA_CONTEXT_TOOL_NAME, force_refresh)

        project_info = call_revit_listener(command_path='/project_info', method='GET')
        if project_info.get("status") == "error":
            return project_info

        doc_fingerprint = "{}|{}|{}".format(
            project_info.get("file_path", ""),
            project_info.get("project_name", ""),
            project_info.get("project_number", "")
        )

        if not force_refresh and schema_context_cache["context"] and schema_context_cache["doc_fingerprint"] == doc_fingerprint:
            cached = dict(schema_context_cache["context"])
            cached["cache"] = {"status": "hit", "doc_fingerprint": doc_fingerprint}
            return cached

        context_result = call_revit_listener(command_path='/schema/context', method='GET')
        if context_result.get("status") == "error":
            if _is_route_not_defined(context_result, "/schema/context"):
                context_result["message"] = (
                    "Route '/schema/context' is not available. Reload Revit to register new schema routes, "
                    "or update extension files."
                )
            return context_result

        schema_context_cache["doc_fingerprint"] = doc_fingerprint
        schema_context_cache["context"] = context_result

        result = dict(context_result)
        result["cache"] = {"status": "refreshed", "doc_fingerprint": doc_fingerprint}
        return compact_result_payload(result)

    def _resolve_revit_targets_internal(query_terms: dict = None) -> dict:
        context_result = get_revit_schema_context_mcp_tool(force_refresh=False)
        if context_result.get("status") == "error":
            return context_result

        schema = context_result.get("schema", {})
        built_in_categories = schema.get("built_in_categories", []) or []
        document_categories = schema.get("document_categories", []) or []
        levels = schema.get("levels", []) or []
        family_names = schema.get("family_names", []) or []
        type_names = schema.get("type_names", []) or []
        parameter_names = schema.get("parameter_names", []) or []

        query_terms = query_terms or {}
        category_term = query_terms.get("category_name")
        level_term = query_terms.get("level_name")
        family_term = query_terms.get("family_name")
        type_term = query_terms.get("type_name")
        parameter_terms = query_terms.get("parameter_names", []) or []
        if isinstance(parameter_terms, str):
            parameter_terms = [parameter_terms]
        elif not isinstance(parameter_terms, list):
            parameter_terms = []

        # Backward compatibility: accept singular parameter keys too.
        for key in ("parameter_name", "parameter"):
            legacy_term = query_terms.get(key)
            if isinstance(legacy_term, str) and legacy_term.strip():
                parameter_terms.append(legacy_term.strip())

        # Preserve order, remove duplicates/empties.
        seen_terms = set()
        normalized_parameter_terms = []
        for pterm in parameter_terms:
            pterm_str = str(pterm).strip()
            if not pterm_str:
                continue
            pkey = pterm_str.lower()
            if pkey in seen_terms:
                continue
            seen_terms.add(pkey)
            normalized_parameter_terms.append(pterm_str)
        parameter_terms = normalized_parameter_terms

        resolution = {"status": "success", "resolved": {}, "alternatives": {}, "confidence": {}, "context_doc": context_result.get("doc", {})}

        if category_term:
            category_candidates = list(set(document_categories + built_in_categories))
            resolved_category, alternatives, confidence = _best_match(category_term, category_candidates)
            if resolved_category:
                # Favor OST_ names when they exist, but keep human names if that's what matched.
                resolution["resolved"]["category_name"] = resolved_category
                resolution["confidence"]["category_name"] = round(confidence, 3)
                if alternatives:
                    resolution["alternatives"]["category_name"] = alternatives
            else:
                resolution["status"] = "partial"
                resolution["alternatives"]["category_name"] = category_candidates[:25]

        if level_term:
            resolved_level, alternatives, confidence = _best_match(level_term, levels)
            if resolved_level:
                resolution["resolved"]["level_name"] = resolved_level
                resolution["confidence"]["level_name"] = round(confidence, 3)
                if alternatives:
                    resolution["alternatives"]["level_name"] = alternatives
            else:
                resolution["status"] = "partial"
                resolution["alternatives"]["level_name"] = levels[:25]

        if family_term:
            resolved_family, alternatives, confidence = _best_match(family_term, family_names)
            if resolved_family:
                resolution["resolved"]["family_name"] = resolved_family
                resolution["confidence"]["family_name"] = round(confidence, 3)
                if alternatives:
                    resolution["alternatives"]["family_name"] = alternatives
            else:
                resolution["status"] = "partial"
                resolution["alternatives"]["family_name"] = family_names[:25]

        if type_term:
            resolved_type, alternatives, confidence = _best_match(type_term, type_names)
            if resolved_type:
                resolution["resolved"]["type_name"] = resolved_type
                resolution["confidence"]["type_name"] = round(confidence, 3)
                if alternatives:
                    resolution["alternatives"]["type_name"] = alternatives
            else:
                resolution["status"] = "partial"
                resolution["alternatives"]["type_name"] = type_names[:25]

        if parameter_terms:
            resolved_params = {}
            unresolved_params = {}
            for pname in parameter_terms:
                resolved_param, alternatives, confidence = _best_match(
                    pname,
                    parameter_names,
                    fuzzy_cutoff=max(0.5, MIN_CONFIDENCE_FOR_PARAMETER_REMAP)
                )
                if resolved_param and confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                    resolved_params[pname] = {"resolved_name": resolved_param, "confidence": round(confidence, 3)}
                    if alternatives:
                        unresolved_params[pname] = alternatives
                else:
                    resolution["status"] = "partial"
                    unresolved_params[pname] = parameter_names[:25]
            resolution["resolved"]["parameter_names"] = resolved_params
            if unresolved_params:
                resolution["alternatives"]["parameter_names"] = unresolved_params

        return compact_result_payload(resolution)

    def resolve_revit_targets_mcp_tool(query_terms: dict) -> dict:
        """Resolves user-supplied names to exact Revit targets with confidence and alternatives."""
        app.logger.info("MCP Tool executed: %s with query_terms=%s", RESOLVE_TARGETS_TOOL_NAME, query_terms)
        if not isinstance(query_terms, dict):
            return {"status": "error", "message": "query_terms must be an object."}
        return _resolve_revit_targets_internal(query_terms)

    # --- MCP Tool Definitions using @mcp_server.tool() ---
    @mcp_server.tool(name=REVIT_INFO_TOOL_NAME) # Name must match what LLM will use
    def get_revit_project_info_mcp_tool() -> dict:
        """Retrieves detailed information about the currently open Revit project."""
        app.logger.info(f"MCP Tool executed: {REVIT_INFO_TOOL_NAME}")
        return call_revit_listener(command_path='/project_info', method='GET')

    @mcp_server.tool(name=GET_SCHEMA_CONTEXT_TOOL_NAME)
    def get_revit_schema_context_tool(force_refresh: bool = False) -> dict:
        """Returns canonical Revit schema context (categories, levels, families, types, parameters)."""
        return get_revit_schema_context_mcp_tool(force_refresh=force_refresh)

    @mcp_server.tool(name=RESOLVE_TARGETS_TOOL_NAME)
    def resolve_revit_targets_tool(query_terms: dict) -> dict:
        """Resolves user terms to exact Revit category/level/family/type/parameter names with confidence."""
        return resolve_revit_targets_mcp_tool(query_terms=query_terms)

    @mcp_server.tool(name=GET_ELEMENTS_BY_CATEGORY_TOOL_NAME)
    def get_elements_by_category_mcp_tool(category_name: str) -> dict:
        """Retrieves all elements in the Revit model belonging to the specified category, returning their IDs and names. Automatically stores the results for later selection."""
        app.logger.info(f"MCP Tool executed: {GET_ELEMENTS_BY_CATEGORY_TOOL_NAME} with category_name: {category_name}")
        result = call_revit_listener(command_path='/get_elements_by_category', method='POST', payload_data={"category_name": category_name})

        if result.get("status") == "error" and "Invalid category_name" in str(result.get("message", "")):
            suggestions = _resolve_revit_targets_internal({"category_name": category_name})
            result["suggestions"] = suggestions
            resolved_category = suggestions.get("resolved", {}).get("category_name")
            if resolved_category:
                result["message"] = "{} Did you mean '{}' ?".format(result.get("message", ""), resolved_category)
        
        # Automatically store the results if successful
        if result.get("status") == "success" and "element_ids" in result:
            storage_key, result_handle = store_elements(category_name, result["element_ids"], result.get("count", len(result["element_ids"])))
            result["stored_as"] = storage_key
            result["result_handle"] = result_handle
            result["storage_message"] = f"Results stored as '{storage_key}' ({result_handle}) - use select_stored_elements with category_name or result_handle"
        
        return compact_result_payload(result, preserve_keys=["stored_as", "result_handle", "storage_message", "count", "category", "status", "message"])

    @mcp_server.tool(name=SELECT_ELEMENTS_TOOL_NAME)
    def select_elements_by_id_mcp_tool(element_ids: list[str] = None, result_handle: str = None) -> dict:
        """Selects one or more elements in Revit using their Element IDs."""
        app.logger.info(f"MCP Tool executed: {SELECT_ELEMENTS_TOOL_NAME}")

        resolved_ids, record, resolve_error = resolve_element_ids(element_ids=element_ids, result_handle=result_handle)
        if resolve_error:
            return resolve_error

        app.logger.info(
            "select_elements_by_id_mcp_tool using %s IDs%s",
            len(resolved_ids),
            " from handle '{}'".format(result_handle) if result_handle else ""
        )
        
        # Ensure element_ids is a list, even if a single string ID is passed by the LLM
        if isinstance(resolved_ids, str):
            app.logger.warning(f"select_elements_by_id_mcp_tool: element_ids was a string ('{resolved_ids}'), converting to list.")
            processed_element_ids = [resolved_ids]
        elif isinstance(resolved_ids, list) and all(isinstance(eid, str) for eid in resolved_ids):
            processed_element_ids = resolved_ids
        elif isinstance(resolved_ids, list): # List contains non-strings, attempt to convert or log error
            app.logger.warning(f"select_elements_by_id_mcp_tool: element_ids list contained non-string items: {resolved_ids}. Attempting to convert all to strings.")
            try:
                processed_element_ids = [str(eid) for eid in resolved_ids]
            except Exception as e_conv:
                app.logger.error(f"select_elements_by_id_mcp_tool: Failed to convert all items in element_ids to string: {e_conv}")
                return {"status": "error", "message": f"Invalid format for element_ids. All IDs must be strings. Received: {resolved_ids}"}
        else:
            app.logger.error(f"select_elements_by_id_mcp_tool: element_ids is not a string or a list of strings. Received type: {type(resolved_ids)}, value: {resolved_ids}")
            return {"status": "error", "message": f"Invalid input type for element_ids. Expected string or list of strings. Received: {type(resolved_ids)}"}

        result = call_revit_listener(command_path='/select_elements_by_id', method='POST', payload_data={"element_ids": processed_element_ids})
        if result.get("status") == "success" and result_handle:
            result["result_handle"] = result_handle
        return compact_result_payload(result)

    @mcp_server.tool(name=SELECT_STORED_ELEMENTS_TOOL_NAME)
    def select_stored_elements_mcp_tool(category_name: str = None, result_handle: str = None) -> dict:
        """Selects elements that were previously retrieved and stored by get_elements_by_category or filter_elements. Use the category name (e.g., 'windows', 'doors') to select the stored elements."""
        app.logger.info(f"MCP Tool executed: {SELECT_STORED_ELEMENTS_TOOL_NAME} with category_name: {category_name}, result_handle: {result_handle}")
        
        stored_data = None
        normalized_category = None
        if result_handle:
            stored_data = get_result_by_handle(result_handle)
            if not stored_data:
                return {
                    "status": "error",
                    "message": f"No stored elements found for result_handle '{result_handle}'.",
                    "available_categories": list(element_storage.keys())
                }
            normalized_category = stored_data.get("storage_key", stored_data.get("category", "unknown"))
        elif category_name:
            # Normalize the input category name
            normalized_category = category_name.lower().replace("ost_", "").replace(" ", "_")
            # Strategy 1: Try exact match first
            stored_data = get_stored_elements(normalized_category)
        else:
            return {"status": "error", "message": "Provide either category_name or result_handle."}
        
        # Strategy 2: If exact match fails, try to find partial matches
        if not stored_data and category_name:
            available_keys = list(element_storage.keys())
            
            # Look for keys that start with the category name (e.g., "windows" matching "windows_level_l5")
            potential_matches = [key for key in available_keys if key.startswith(normalized_category)]
            
            if potential_matches:
                # If multiple matches, prefer the most recent one (they're stored in order)
                best_match = potential_matches[-1]  # Take the last (most recent) match
                stored_data = get_stored_elements(best_match)
                app.logger.info(f"Found partial match: '{best_match}' for category '{category_name}'")
            else:
                # Strategy 3: Try fuzzy matching - look for the category name anywhere in the key
                fuzzy_matches = [key for key in available_keys if normalized_category in key]
                if fuzzy_matches:
                    best_match = fuzzy_matches[-1]  # Take the most recent
                    stored_data = get_stored_elements(best_match)
                    app.logger.info(f"Found fuzzy match: '{best_match}' for category '{category_name}'")
        
        if not stored_data:
            available_keys = list(element_storage.keys())
            return {
                "status": "error", 
                "message": f"No stored elements found for category '{category_name or result_handle}'. Available stored categories: {available_keys}",
                "available_categories": available_keys,
                "suggestion": "Try using list_stored_elements to see all available categories, or use the exact storage key name."
            }
        
        # Use the stored element IDs
        element_ids = stored_data["element_ids"]
        total_elements = len(element_ids)
        app.logger.info(f"Using {total_elements} stored element IDs for category '{category_name}' (matched to stored key)")

        if total_elements > MAX_ELEMENTS_FOR_SELECTION:
            app.logger.warning("Selection aborted: %s elements exceeds safe limit of %s", total_elements, MAX_ELEMENTS_FOR_SELECTION)
            return {
                "status": "limit_exceeded",
                "message": f"Selection would include {total_elements} elements which exceeds the safe limit of {MAX_ELEMENTS_FOR_SELECTION}.",
                "suggestion": "Please narrow your criteria (e.g., filter by level or parameter) before selecting.",
                "stored_count": stored_data.get("count", total_elements),
                "stored_key": stored_data.get("category", category_name),
                "selection_limit": MAX_ELEMENTS_FOR_SELECTION
            }

        # Prefer focused selection when available, but gracefully fall back for older route sets.
        result = call_revit_listener(command_path='/select_elements_focused', method='POST', payload_data={"element_ids": element_ids})
        if result.get("status") == "error":
            if _is_route_not_defined(result, "/select_elements_focused") or "select_elements_focused" in str(result.get("message", "")):
                app.logger.warning(
                    "Route '/select_elements_focused' is not available on the active Revit API. "
                    "Falling back to '/select_elements_by_id'."
                )
                fallback_result = call_revit_listener(
                    command_path='/select_elements_by_id',
                    method='POST',
                    payload_data={"element_ids": element_ids}
                )
                if fallback_result.get("status") == "success":
                    fallback_result["approach_note"] = "Fallback selection used because focused selection route was unavailable"
                result = fallback_result

        # Add storage info to the result
        if result.get("status") == "success":
            result["source"] = f"stored_{category_name or normalized_category}"
            result["stored_count"] = stored_data["count"]
            result["stored_at"] = stored_data["timestamp"]
            result["matched_key"] = stored_data.get("category", "unknown")
            result["result_handle"] = stored_data.get("result_handle")
            result["approach_note"] = "Focused selection - elements should remain active for user operations"

        return compact_result_payload(result)

    @mcp_server.tool(name=LIST_STORED_ELEMENTS_TOOL_NAME)
    def list_stored_elements_mcp_tool() -> dict:
        """Lists all currently stored element categories and their counts. Use this to see what elements are available for selection."""
        app.logger.info(f"MCP Tool executed: {LIST_STORED_ELEMENTS_TOOL_NAME}")
        
        stored_categories = list_stored_categories()
        
        return {
            "status": "success",
            "message": f"Found {len(stored_categories)} stored categories",
            "stored_categories": stored_categories,
            "total_categories": len(stored_categories)
        }

    @mcp_server.tool(name=FILTER_ELEMENTS_TOOL_NAME)
    def filter_elements_mcp_tool(category_name: str, level_name: str = None, parameters: list = None) -> dict:
        """Filters elements by category, level, and parameter conditions. Returns element IDs matching the criteria. Use this instead of get_elements_by_category when you need specific filtering."""
        app.logger.info(f"MCP Tool executed: {FILTER_ELEMENTS_TOOL_NAME} with category: {category_name}, level: {level_name}, parameters: {parameters}")

        resolution = _resolve_revit_targets_internal({
            "category_name": category_name,
            "level_name": level_name,
            "parameter_names": [p.get("name") for p in (parameters or []) if isinstance(p, dict) and p.get("name")]
        })
        if resolution.get("status") != "error":
            resolved_payload = resolution.get("resolved", {})
            if resolved_payload.get("category_name"):
                category_name = resolved_payload["category_name"]
            if resolved_payload.get("level_name"):
                level_name = resolved_payload["level_name"]
            param_map = resolved_payload.get("parameter_names", {})
            if param_map and isinstance(parameters, list):
                for p in parameters:
                    if isinstance(p, dict) and p.get("name") in param_map:
                        mapped = param_map[p["name"]]
                        resolved_name = mapped.get("resolved_name", p["name"])
                        confidence = float(mapped.get("confidence", 0.0))
                        if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                            p["name"] = resolved_name
                        else:
                            app.logger.info(
                                "Skipping low-confidence parameter remap for '%s' -> '%s' (confidence=%s)",
                                p["name"], resolved_name, confidence
                            )
        
        payload = {"category_name": category_name}
        if level_name:
            payload["level_name"] = level_name
        if parameters:
            payload["parameters"] = parameters
        
        result = call_revit_listener(command_path='/elements/filter', method='POST', payload_data=payload)

        if result.get("status") == "error":
            error_msg = str(result.get("message", ""))
            if "Invalid category_name" in error_msg or "Level '" in error_msg:
                resolver_input = {
                    "category_name": category_name,
                    "level_name": level_name,
                    "parameter_names": [p.get("name") for p in (parameters or []) if isinstance(p, dict) and p.get("name")]
                }
                suggestions = _resolve_revit_targets_internal(resolver_input)
                result["suggestions"] = suggestions
                if "Invalid category_name" in error_msg:
                    resolved_category = suggestions.get("resolved", {}).get("category_name")
                    alt_categories = suggestions.get("alternatives", {}).get("category_name", [])
                    if resolved_category:
                        result["message"] = "{} Did you mean '{}' ?".format(error_msg, resolved_category)
                    elif alt_categories:
                        result["message"] = "{} Available categories include: {}".format(error_msg, ", ".join(alt_categories[:10]))
                if "Level '" in error_msg:
                    resolved_level = suggestions.get("resolved", {}).get("level_name")
                    alt_levels = suggestions.get("alternatives", {}).get("level_name", [])
                    if resolved_level:
                        result["message"] = "{} Did you mean level '{}' ?".format(result.get("message", error_msg), resolved_level)
                    elif alt_levels:
                        result["message"] = "{} Available levels include: {}".format(result.get("message", error_msg), ", ".join(alt_levels[:10]))

        if result.get("status") == "error" and _is_route_not_defined(result, "/elements/filter"):
            # Compatibility fallback: if no advanced filters were requested, use category retrieval.
            if not level_name and not parameters:
                app.logger.warning("Route '/elements/filter' missing. Falling back to '/get_elements_by_category' for category-only request.")
                result = call_revit_listener(
                    command_path='/get_elements_by_category',
                    method='POST',
                    payload_data={"category_name": category_name}
                )
            else:
                return {
                    "status": "error",
                    "error_type": "route_not_defined",
                    "message": "The active Revit route set does not support '/elements/filter'. Reload/update the Revit extension to use advanced filtering."
                }
        
        # Automatically store the results if successful
        if result.get("status") == "success" and "element_ids" in result:
            # Create a descriptive storage key
            storage_key = category_name.lower().replace("ost_", "").replace(" ", "_")
            if level_name:
                storage_key += f"_level_{level_name.lower()}"
            if parameters:
                storage_key += "_filtered"
            
            stored_key, result_handle = store_elements(storage_key, result["element_ids"], result.get("count", len(result["element_ids"])))
            result["stored_as"] = stored_key
            result["result_handle"] = result_handle
            result["storage_message"] = f"Filtered results stored as '{stored_key}' ({result_handle}) - use select_stored_elements with category_name or result_handle"
        
        return compact_result_payload(result, preserve_keys=["stored_as", "result_handle", "storage_message", "count", "category", "status", "message"])

    def _matches_filter_value(candidate_value, expected_value, operator="contains", case_sensitive=False):
        candidate = "" if candidate_value in (None, "Not available") else str(candidate_value)
        expected = "" if expected_value is None else str(expected_value)
        op = str(operator or "contains").strip().lower()

        if not case_sensitive:
            candidate_cmp = candidate.lower()
            expected_cmp = expected.lower()
        else:
            candidate_cmp = candidate
            expected_cmp = expected

        if op in ("contains",):
            return expected_cmp in candidate_cmp
        if op in ("equals", "=="):
            return candidate_cmp == expected_cmp
        if op in ("not_equals", "!=", "not equal"):
            return candidate_cmp != expected_cmp
        if op in ("starts_with",):
            return candidate_cmp.startswith(expected_cmp)
        if op in ("ends_with",):
            return candidate_cmp.endswith(expected_cmp)
        return False

    @mcp_server.tool(name=FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME)
    def filter_stored_elements_by_parameter_mcp_tool(
        parameter_name: str,
        value: str,
        operator: str = "contains",
        result_handle: str = None,
        category_name: str = None,
        batch_size: int = None,
        case_sensitive: bool = False
    ) -> dict:
        """
        Server-side batch filter across a stored element set by parameter value.
        This keeps large element/property scans out of LLM context.
        """
        app.logger.info(
            "MCP Tool executed: %s (parameter=%s, operator=%s, value=%s, result_handle=%s, category_name=%s, batch_size=%s)",
            FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME,
            parameter_name,
            operator,
            value,
            result_handle,
            category_name,
            batch_size
        )

        if not parameter_name or value is None:
            return {"status": "error", "message": "parameter_name and value are required."}

        parameter_resolution = _resolve_revit_targets_internal({"parameter_names": [parameter_name]})
        param_map = parameter_resolution.get("resolved", {}).get("parameter_names", {})
        if parameter_name in param_map:
            mapped = param_map[parameter_name]
            confidence = float(mapped.get("confidence", 0.0))
            if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                parameter_name = mapped.get("resolved_name", parameter_name)

        resolved_ids, record, resolve_error = resolve_element_ids(
            element_ids=None,
            result_handle=result_handle,
            category_name=category_name
        )
        if resolve_error:
            return resolve_error

        total_ids = len(resolved_ids)
        if total_ids == 0:
            return {"status": "success", "count": 0, "message": "No source elements available for server-side filtering."}

        if batch_size is None:
            # Larger defaults for very large sets reduce per-request overhead.
            inferred_batch_size = 1000 if total_ids >= 5000 else DEFAULT_SERVER_FILTER_BATCH_SIZE
        else:
            inferred_batch_size = int(batch_size)

        safe_batch_size = int(inferred_batch_size)
        safe_batch_size = max(20, min(1000, safe_batch_size))
        total_batches = int((total_ids + safe_batch_size - 1) / safe_batch_size)

        matched_ids = []
        matched_samples = []

        for batch_index, start in enumerate(range(0, total_ids, safe_batch_size), 1):
            batch_ids = resolved_ids[start:start + safe_batch_size]
            batch_result = call_revit_listener(
                command_path='/elements/get_properties',
                method='POST',
                payload_data={
                    "element_ids": batch_ids,
                    "parameter_names": [parameter_name]
                }
            )

            if batch_result.get("status") == "error":
                if _is_route_not_defined(batch_result, "/elements/get_properties"):
                    return {
                        "status": "error",
                        "error_type": "route_not_defined",
                        "message": "The active Revit route set does not support '/elements/get_properties'. Reload/update the Revit extension to enable server-side filtering."
                    }
                return batch_result

            elements = batch_result.get("elements", []) or []
            for element_data in elements:
                element_id = str(element_data.get("element_id", "")).strip()
                properties = element_data.get("properties", {}) or {}
                current_value = properties.get(parameter_name, "Not available")
                if _matches_filter_value(current_value, value, operator=operator, case_sensitive=case_sensitive):
                    matched_ids.append(element_id)
                    if len(matched_samples) < MAX_RECORDS_IN_RESPONSE:
                        matched_samples.append({"element_id": element_id, parameter_name: current_value})

            if batch_index == 1 or batch_index % 5 == 0 or batch_index == total_batches:
                app.logger.info(
                    "Server-side filter progress: batch %s/%s, processed=%s/%s, matched=%s",
                    batch_index,
                    total_batches,
                    min(start + len(batch_ids), total_ids),
                    total_ids,
                    len(matched_ids)
                )

        source_key = record.get("storage_key") if isinstance(record, dict) else _normalize_storage_key(category_name or "elements")
        parameter_key = _normalize_storage_key(parameter_name)
        filtered_storage_seed = "{}_{}_filtered".format(source_key, parameter_key)
        stored_key, new_result_handle = store_elements(filtered_storage_seed, matched_ids, len(matched_ids))

        result = {
            "status": "success",
            "count": len(matched_ids),
            "source_count": total_ids,
            "processed_count": total_ids,
            "parameter_name": parameter_name,
            "operator": operator,
            "value": str(value),
            "case_sensitive": bool(case_sensitive),
            "matched_sample": matched_samples,
            "element_ids": matched_ids,
            "stored_as": stored_key,
            "result_handle": new_result_handle,
            "storage_message": "Filtered results stored as '{}' ({}) - use select_stored_elements with result_handle".format(stored_key, new_result_handle),
            "message": "Server-side parameter filtering matched {} of {} source elements.".format(len(matched_ids), total_ids)
        }
        return compact_result_payload(
            result,
            preserve_keys=["stored_as", "result_handle", "storage_message", "count", "source_count", "processed_count", "status", "message", "parameter_name", "operator", "value"]
        )

    @mcp_server.tool(name=GET_ELEMENT_PROPERTIES_TOOL_NAME)
    def get_element_properties_mcp_tool(
        element_ids: list[str] = None,
        parameter_names: list[str] = None,
        result_handle: str = None,
        include_all_parameters: bool = False,
        populated_only: bool = False
    ) -> dict:
        """Gets parameter values for specified elements. If parameter_names not provided, returns common parameters for the element category."""
        resolved_ids, record, resolve_error = resolve_element_ids(element_ids=element_ids, result_handle=result_handle)
        if resolve_error:
            return resolve_error
        requested_count = len(resolved_ids)
        truncated_for_safety = False
        if requested_count > MAX_ELEMENTS_FOR_PROPERTY_READ:
            resolved_ids = resolved_ids[:MAX_ELEMENTS_FOR_PROPERTY_READ]
            truncated_for_safety = True
            app.logger.warning(
                "%s requested %s elements; limiting property read to first %s for safety",
                GET_ELEMENT_PROPERTIES_TOOL_NAME,
                requested_count,
                MAX_ELEMENTS_FOR_PROPERTY_READ
            )

        app.logger.info(f"MCP Tool executed: {GET_ELEMENT_PROPERTIES_TOOL_NAME} with {len(resolved_ids)} elements")
        
        payload = {"element_ids": resolved_ids}
        if parameter_names:
            payload["parameter_names"] = parameter_names
        if include_all_parameters:
            payload["include_all_parameters"] = True
        if populated_only:
            payload["populated_only"] = True
        
        result = call_revit_listener(command_path='/elements/get_properties', method='POST', payload_data=payload)
        if result.get("status") == "error" and _is_route_not_defined(result, "/elements/get_properties"):
            return {
                "status": "error",
                "error_type": "route_not_defined",
                "message": "The active Revit route set does not support '/elements/get_properties'. Reload/update the Revit extension to enable property reads."
            }
        if result.get("status") == "success" and result_handle:
            result["result_handle"] = result_handle
        if result.get("status") == "success" and truncated_for_safety:
            result["requested_count"] = requested_count
            result["processed_count"] = len(resolved_ids)
            result["truncated_for_safety"] = True
            result["message"] = (
                "Retrieved properties for {} elements (safety-capped from {}). "
                "Narrow with filter_elements for full-accuracy bulk analysis."
            ).format(len(resolved_ids), requested_count)
        return compact_result_payload(result)

    @mcp_server.tool(name=UPDATE_ELEMENT_PARAMETERS_TOOL_NAME)
    def update_element_parameters_mcp_tool(
        updates: list[dict] = None,
        element_ids: list[str] = None,
        result_handle: str = None,
        parameter_name: str = None,
        new_value: str = None
    ) -> dict:
        """Updates parameter values for elements. Accepts either a detailed updates list or a simplified bulk form."""
        app.logger.info(f"MCP Tool executed: {UPDATE_ELEMENT_PARAMETERS_TOOL_NAME}")

        normalized_updates: list[dict] = []

        if updates:
            if not isinstance(updates, list) or not updates:
                return {"status": "error", "message": "'updates' must be a non-empty list of update payloads."}

            for update in updates:
                if not isinstance(update, dict):
                    return {"status": "error", "message": "Each update must be an object with element_id and parameters."}
                element_id = str(update.get('element_id', '')).strip()
                parameters = update.get('parameters')
                if not element_id or not parameters:
                    return {"status": "error", "message": "Each update requires element_id and parameters."}
                if not isinstance(parameters, dict) or not parameters:
                    return {"status": "error", "message": "'parameters' must be a non-empty object of parameter/value pairs."}
                normalized_updates.append({"element_id": element_id, "parameters": parameters})

        elif (element_ids or result_handle) and parameter_name and new_value is not None:
            resolved_ids, record, resolve_error = resolve_element_ids(element_ids=element_ids, result_handle=result_handle)
            if resolve_error:
                return resolve_error
            if not isinstance(resolved_ids, list) or not resolved_ids:
                return {"status": "error", "message": "'element_ids' must be a non-empty list when using the simplified form."}
            parameter_name = str(parameter_name).strip()
            if not parameter_name:
                return {"status": "error", "message": "parameter_name cannot be empty."}
            normalized_value = str(new_value)
            for eid in resolved_ids:
                element_id = str(eid).strip()
                if not element_id:
                    return {"status": "error", "message": "All element_ids must be non-empty strings."}
                normalized_updates.append({"element_id": element_id, "parameters": {parameter_name: normalized_value}})
        else:
            return {"status": "error", "message": "Provide either 'updates' or (element_ids, parameter_name, new_value)."}

        app.logger.info(f"Prepared {len(normalized_updates)} parameter update(s) for execution.")

        # Resolve parameter names for simplified bulk form.
        if parameter_name:
            parameter_resolution = _resolve_revit_targets_internal({
                "parameter_names": [parameter_name]
            })
            param_map = parameter_resolution.get("resolved", {}).get("parameter_names", {})
            if parameter_name in param_map:
                mapped = param_map[parameter_name]
                confidence = float(mapped.get("confidence", 0.0))
                if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                    parameter_name = mapped.get("resolved_name", parameter_name)

        result = call_revit_listener(
            command_path='/elements/update_parameters',
            method='POST',
            payload_data={"updates": normalized_updates}
        )
        if result.get("status") == "error" and _is_route_not_defined(result, "/elements/update_parameters"):
            return {
                "status": "error",
                "error_type": "route_not_defined",
                "message": "The active Revit route set does not support '/elements/update_parameters'. Reload/update the Revit extension to enable parameter updates."
            }
        if result.get("status") == "success" and result_handle:
            result["result_handle"] = result_handle
        return compact_result_payload(result)

    
    @mcp_server.tool(name=PLACE_VIEW_ON_SHEET_TOOL_NAME)
    def place_view_on_sheet_mcp_tool(view_name: str, exact_match: bool = False) -> dict:
        """Places a view on a new sheet by view name. Creates a new sheet with automatic numbering and places the view in the center. Supports fuzzy matching for view names."""
        app.logger.info(f"MCP Tool executed: {PLACE_VIEW_ON_SHEET_TOOL_NAME} with view_name: {view_name}, exact_match: {exact_match}")
        
        return call_revit_listener(command_path='/sheets/place_view', method='POST', payload_data={"view_name": view_name, "exact_match": exact_match})

    @mcp_server.tool(name=LIST_VIEWS_TOOL_NAME)
    def list_views_mcp_tool() -> dict:
        """Lists all views in the current document that can be placed on sheets. Returns view names, types, and whether they're already on sheets."""
        app.logger.info(f"MCP Tool executed: {LIST_VIEWS_TOOL_NAME}")
        
        return call_revit_listener(command_path='/sheets/list_views', method='GET')

    @mcp_server.tool(name=PLANNER_TOOL_NAME)
    def plan_and_execute_workflow_tool(user_request: str, execution_plan: list[dict]) -> dict:
        """
        Generic planner that executes a sequence of tools based on a planned workflow.
        The LLM should first analyze the user request, then provide a step-by-step execution plan.
        
        Args:
            user_request: The original user request
            execution_plan: List of planned steps, each containing:
                - tool: Tool name to execute
                - params: Parameters for the tool
                - description: What this step accomplishes
        """
        app.logger.info(f"MCP Tool executed: {PLANNER_TOOL_NAME} - Executing {len(execution_plan)} planned steps")
        
        workflow_results = {
            "user_request": user_request,
            "planned_steps": len(execution_plan),
            "executed_steps": [],
            "step_results": [],
            "final_status": "success",
            "summary": ""
        }
        
        # Available tool mapping
        available_tools = {
            "get_revit_project_info": get_revit_project_info_mcp_tool,
            "get_revit_schema_context": lambda **kwargs: get_revit_schema_context_tool(kwargs.get("force_refresh", False)),
            "resolve_revit_targets": lambda **kwargs: resolve_revit_targets_tool(kwargs.get("query_terms", {})),
            "get_elements_by_category": lambda **kwargs: get_elements_by_category_mcp_tool(kwargs.get("category_name")),
            "filter_elements": lambda **kwargs: filter_elements_mcp_tool(
                kwargs.get("category_name"), 
                kwargs.get("level_name"), 
                kwargs.get("parameters", [])
            ),
            "filter_stored_elements_by_parameter": lambda **kwargs: filter_stored_elements_by_parameter_mcp_tool(
                parameter_name=kwargs.get("parameter_name"),
                value=kwargs.get("value"),
                operator=kwargs.get("operator", "contains"),
                result_handle=kwargs.get("result_handle"),
                category_name=kwargs.get("category_name"),
                batch_size=kwargs.get("batch_size"),
                case_sensitive=kwargs.get("case_sensitive", False)
            ),
            "get_element_properties": lambda **kwargs: get_element_properties_mcp_tool(
                kwargs.get("element_ids", []),
                kwargs.get("parameter_names", []),
                kwargs.get("result_handle"),
                kwargs.get("include_all_parameters", False),
                kwargs.get("populated_only", False)
            ),
            "update_element_parameters": lambda **kwargs: update_element_parameters_mcp_tool(
                updates=kwargs.get("updates"),
                element_ids=kwargs.get("element_ids"),
                result_handle=kwargs.get("result_handle"),
                parameter_name=kwargs.get("parameter_name"),
                new_value=kwargs.get("new_value")
            ),
            "select_elements_by_id": lambda **kwargs: select_elements_by_id_mcp_tool(
                kwargs.get("element_ids", []),
                kwargs.get("result_handle")
            ),
            "select_stored_elements": lambda **kwargs: select_stored_elements_mcp_tool(
                kwargs.get("category_name"),
                kwargs.get("result_handle")
            ),
            "list_stored_elements": list_stored_elements_mcp_tool,
            "place_view_on_sheet": lambda **kwargs: place_view_on_sheet_mcp_tool(
                kwargs.get("view_name"), 
                kwargs.get("exact_match", False)
            ),
            "list_views": list_views_mcp_tool
        }
        
        try:
            for i, step in enumerate(execution_plan, 1):
                step_info = {
                    "step_number": i,
                    "tool": step.get("tool"),
                    "description": step.get("description", ""),
                    "status": "pending"
                }
                
                tool_name = step.get("tool")
                tool_params = step.get("params", {}).copy()  # Make a copy to avoid modifying the original
                
                # Substitute placeholders in parameters with values from previous steps
                def substitute_placeholders(obj):
                    """Recursively substitute placeholder values in the object."""
                    if isinstance(obj, str):
                        # Look for ${step_X_key} patterns and replace them
                        import re
                        pattern = r'\$\{step_(\d+)_([^}]+)\}'
                        
                        def replace_placeholder(match):
                            step_num = int(match.group(1))
                            key = match.group(2)
                            placeholder_key = f"step_{step_num}_{key}"
                            if placeholder_key in workflow_results:
                                value = workflow_results[placeholder_key]
                                # If the entire string is just the placeholder, return the actual value (preserving type)
                                if obj.strip() == match.group(0):
                                    return value
                                # Otherwise, convert to string for partial replacement
                                return str(value)
                            else:
                                app.logger.warning(f"Placeholder {placeholder_key} not found in workflow results")
                                return match.group(0)  # Return original if not found
                        
                        # Check if the entire string is just a placeholder
                        full_match = re.fullmatch(pattern, obj.strip())
                        if full_match:
                            step_num = int(full_match.group(1))
                            key = full_match.group(2)
                            placeholder_key = f"step_{step_num}_{key}"
                            if placeholder_key in workflow_results:
                                return workflow_results[placeholder_key]  # Return the actual value (preserving type)
                        
                        # Otherwise do normal substitution
                        return re.sub(pattern, replace_placeholder, obj)
                    elif isinstance(obj, dict):
                        return {k: substitute_placeholders(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [substitute_placeholders(item) for item in obj]
                    else:
                        return obj
                
                # Apply substitution to all parameters
                tool_params = substitute_placeholders(tool_params)

                # Enforce resolver-first behavior for filter/update operations
                if tool_name in ["filter_elements", "update_element_parameters", "filter_stored_elements_by_parameter"]:
                    resolver_input = {}
                    if tool_name == "filter_elements":
                        resolver_input["category_name"] = tool_params.get("category_name")
                        resolver_input["level_name"] = tool_params.get("level_name")
                        resolver_input["parameter_names"] = [
                            p.get("name") for p in (tool_params.get("parameters", []) or [])
                            if isinstance(p, dict) and p.get("name")
                        ]
                    elif tool_name == "filter_stored_elements_by_parameter":
                        resolver_input["parameter_names"] = [tool_params.get("parameter_name")]
                    elif tool_name == "update_element_parameters":
                        parameter_candidates = []
                        if tool_params.get("parameter_name"):
                            parameter_candidates.append(tool_params.get("parameter_name"))
                        if isinstance(tool_params.get("updates"), list):
                            for upd in tool_params.get("updates"):
                                if isinstance(upd, dict) and isinstance(upd.get("parameters"), dict):
                                    parameter_candidates.extend(list(upd.get("parameters").keys()))
                        resolver_input["parameter_names"] = list(set(parameter_candidates))

                    resolution = _resolve_revit_targets_internal(resolver_input)
                    step_info["resolution"] = resolution
                    if resolution.get("status") == "error":
                        step_info["status"] = "error"
                        step_info["error"] = "Target resolution failed before executing '{}'".format(tool_name)
                        step_info["result"] = resolution
                        workflow_results["executed_steps"].append(step_info)
                        workflow_results["step_results"].append(step_info["result"])
                        continue

                    resolved_payload = resolution.get("resolved", {})
                    if tool_name == "filter_elements":
                        if resolved_payload.get("category_name"):
                            tool_params["category_name"] = resolved_payload["category_name"]
                        if resolved_payload.get("level_name"):
                            tool_params["level_name"] = resolved_payload["level_name"]
                        param_map = resolved_payload.get("parameter_names", {})
                        if param_map and isinstance(tool_params.get("parameters"), list):
                            for p in tool_params["parameters"]:
                                if isinstance(p, dict) and p.get("name") in param_map:
                                    mapped = param_map[p["name"]]
                                    confidence = float(mapped.get("confidence", 0.0))
                                    if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                                        p["name"] = mapped.get("resolved_name", p["name"])
                    elif tool_name == "filter_stored_elements_by_parameter":
                        param_map = resolved_payload.get("parameter_names", {})
                        current_param = tool_params.get("parameter_name")
                        if current_param in param_map:
                            mapped = param_map[current_param]
                            confidence = float(mapped.get("confidence", 0.0))
                            if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                                tool_params["parameter_name"] = mapped.get("resolved_name", current_param)
                    elif tool_name == "update_element_parameters":
                        param_map = resolved_payload.get("parameter_names", {})
                        if tool_params.get("parameter_name") in param_map:
                            mapped = param_map[tool_params["parameter_name"]]
                            confidence = float(mapped.get("confidence", 0.0))
                            if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                                tool_params["parameter_name"] = mapped.get("resolved_name", tool_params["parameter_name"])
                        if isinstance(tool_params.get("updates"), list):
                            for upd in tool_params["updates"]:
                                if isinstance(upd, dict) and isinstance(upd.get("parameters"), dict):
                                    new_params = {}
                                    for k, v in upd["parameters"].items():
                                        if k in param_map:
                                            mapped = param_map[k]
                                            confidence = float(mapped.get("confidence", 0.0))
                                            if confidence >= MIN_CONFIDENCE_FOR_PARAMETER_REMAP:
                                                new_params[mapped.get("resolved_name", k)] = v
                                            else:
                                                new_params[k] = v
                                        else:
                                            new_params[k] = v
                                    upd["parameters"] = new_params
                
                app.logger.info(f"Executing step {i}: {tool_name} - {step.get('description', '')}")
                app.logger.debug(f"Step {i} parameters after substitution: {tool_params}")
                
                if tool_name not in available_tools:
                    step_info["status"] = "error"
                    step_info["error"] = f"Unknown tool: {tool_name}"
                    step_info["result"] = {"error": f"Tool '{tool_name}' not available"}
                else:
                    try:
                        # Execute the tool
                        tool_function = available_tools[tool_name]
                        if tool_name == "get_revit_project_info" or tool_name == "list_stored_elements":
                            # Tools that take no parameters
                            result = tool_function()
                        else:
                            # Tools that take parameters
                            result = tool_function(**tool_params)
                        
                        step_info["result"] = result

                        result_status = ""
                        if isinstance(result, dict):
                            result_status = str(result.get("status", "")).lower()

                        if result_status in ["error", "failed", "limit_exceeded"]:
                            step_info["status"] = "error"
                            if isinstance(result, dict) and "message" in result:
                                step_info["error"] = result.get("message")
                        else:
                            step_info["status"] = "completed"
                        
                        # Store results for potential use in subsequent steps
                        # This allows chaining where one step's output feeds into the next
                        if isinstance(result, dict) and "result_handle" in result:
                            workflow_results[f"step_{i}_result_handle"] = result["result_handle"]
                        if isinstance(result, dict) and "element_ids" in result:
                            workflow_results[f"step_{i}_element_ids"] = result["element_ids"]
                        if isinstance(result, dict) and "element_ids_sample" in result:
                            workflow_results[f"step_{i}_element_ids_sample"] = result["element_ids_sample"]
                        if isinstance(result, dict) and "count" in result:
                            workflow_results[f"step_{i}_count"] = result["count"]
                        if isinstance(result, dict) and "elements" in result:
                            workflow_results[f"step_{i}_elements"] = result["elements"]
                        
                    except Exception as tool_error:
                        step_info["status"] = "error" 
                        step_info["error"] = str(tool_error)
                        step_info["result"] = {"error": str(tool_error)}
                        app.logger.error(f"Step {i} failed: {tool_error}")
                
                workflow_results["executed_steps"].append(step_info)
                workflow_results["step_results"].append(step_info["result"])
            
            # Generate summary
            successful_steps = len([s for s in workflow_results["executed_steps"] if s["status"] == "completed"])
            failed_steps = len([s for s in workflow_results["executed_steps"] if s["status"] == "error"])
            
            if failed_steps == 0:
                workflow_results["final_status"] = "success"
                workflow_results["summary"] = f"Successfully completed all {successful_steps} planned steps"
            elif successful_steps > 0:
                workflow_results["final_status"] = "partial"
                workflow_results["summary"] = f"Completed {successful_steps} steps, {failed_steps} steps failed"
            else:
                workflow_results["final_status"] = "failed"
                workflow_results["summary"] = f"All {failed_steps} steps failed"
            
            app.logger.info(f"Workflow completed: {workflow_results['summary']}")
            return workflow_results
            
        except Exception as e:
            workflow_results["final_status"] = "error"
            workflow_results["error"] = str(e)
            workflow_results["summary"] = f"Workflow execution failed: {str(e)}"
            app.logger.error(f"Workflow execution error: {e}")
            return workflow_results

    app.logger.info("MCP tools defined and decorated.")

    # --- LLM Tool Specifications (Manual for now, for existing LLM API calls) ---
    # These definitions tell the LLMs (OpenAI, Anthropic, Google) about the tools.
    # The 'name' in these specs MUST match the 'name' in @mcp_server.tool() and the constants.
    REVIT_INFO_TOOL_DESCRIPTION_FOR_LLM = "Retrieves detailed information about the currently open Revit project, such as project name, file path, Revit version, Revit build number, and active document title."
    GET_SCHEMA_CONTEXT_TOOL_DESCRIPTION_FOR_LLM = "Returns canonical Revit schema context (exact levels, category names, family/type names, common parameter names). Use this before filtering or parameter updates."
    GET_SCHEMA_CONTEXT_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "force_refresh": {"type": "boolean", "description": "If true, refresh schema context from Revit even if cached."}
        }
    }
    RESOLVE_TARGETS_TOOL_DESCRIPTION_FOR_LLM = "Resolves user terms (category/level/family/type/parameter names) to exact Revit names with confidence and alternatives. Always call this before filter_elements or update_element_parameters."
    RESOLVE_TARGETS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "query_terms": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string"},
                    "level_name": {"type": "string"},
                    "family_name": {"type": "string"},
                    "type_name": {"type": "string"},
                    "parameter_names": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["query_terms"]
    }
    GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM = "Retrieves and stores all elements in the current Revit model for the specified category. Use this ONLY when the user wants to find/get elements. This automatically stores results for later selection. After calling this, if the user wants to select the elements, use select_stored_elements."
    GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object", "properties": {"category_name": {"type": "string", "description": "The name of the Revit category to retrieve elements from (e.g., 'OST_Windows', 'OST_Doors', 'OST_Walls' or simplified like 'Windows', 'Doors', 'Walls')."}}, "required": ["category_name"]
    }
    SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM = "DEPRECATED - Prefer select_stored_elements. Selects elements by exact IDs or by a stored result_handle."
    SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "element_ids": {"type": "array", "items": {"type": "string"}, "description": "Array of Element IDs to select. Avoid large lists when possible."},
            "result_handle": {"type": "string", "description": "Handle from get_elements_by_category/filter_elements to select without passing all IDs."}
        }
    }
    SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM = "Selects elements previously retrieved by get_elements_by_category or filter_elements. Prefer using result_handle to avoid passing large element lists."
    SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "category_name": {"type": "string", "description": "Stored category key (e.g., 'windows', 'doors')."},
            "result_handle": {"type": "string", "description": "Preferred. Handle returned by get_elements_by_category/filter_elements."}
        }
    }
    LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM = "Lists all currently stored element categories and their counts. Use this to see what elements are available for selection using select_stored_elements."
    LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object", "properties": {}
    }
    FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM = "Filters elements by category, level, and parameter conditions. Use this when you need to find specific elements with certain criteria (e.g., windows on Level 5 with specific sill height). More powerful than get_elements_by_category for specific searches. IMPORTANT: When filtering for parameter updates, always follow with get_element_properties to verify current values, then update_element_parameters to make changes. Chain these tools together in one conversation turn."
    FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object", 
        "properties": {
            "category_name": {"type": "string", "description": "The Revit category (e.g., 'OST_Windows', 'Windows')"},
            "level_name": {"type": "string", "description": "Optional level name to filter by (e.g., 'Level 1', 'L5')"},
            "parameters": {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Parameter name (e.g., 'Sill Height', 'Width')"},
                        "value": {"type": "string", "description": "Parameter value to match (e.g., '2\\' 3\\\"', '900')"},
                        "condition": {"type": "string", "enum": ["equals", "contains", "greater_than", "less_than"], "description": "Comparison condition"}
                    },
                    "required": ["name", "value"]
                },
                "description": "Optional parameter filters"
            }
        }, 
        "required": ["category_name"]
    }
    FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_DESCRIPTION_FOR_LLM = "Filters a previously stored element set using server-side batched parameter reads. Use this after get_elements_by_category/filter_elements to avoid loading large property datasets into model context."
    FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "result_handle": {"type": "string", "description": "Preferred. Source handle returned by get_elements_by_category/filter_elements."},
            "category_name": {"type": "string", "description": "Alternative to result_handle. Stored category key."},
            "parameter_name": {"type": "string", "description": "Exact parameter name to evaluate (e.g., 'Part Number')."},
            "value": {"type": "string", "description": "Value to compare against."},
            "operator": {
                "type": "string",
                "enum": ["contains", "equals", "not_equals", "starts_with", "ends_with"],
                "description": "Comparison operator. Default is 'contains'."
            },
            "batch_size": {"type": "integer", "description": "Optional server-side batch size (20-1000)."},
            "case_sensitive": {"type": "boolean", "description": "Whether string matching is case-sensitive. Default false."}
        },
        "required": ["parameter_name", "value"]
    }
    GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM = "Gets parameter values for specified elements. Prefer result_handle over raw element_ids for large result sets. Use include_all_parameters=true with populated_only=true to discover which parameters actually have values."
    GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "element_ids": {"type": "array", "items": {"type": "string"}, "description": "Array of element IDs to get properties for (use only for small sets)."},
            "result_handle": {"type": "string", "description": "Preferred. Handle from a previous search/filter result."},
            "parameter_names": {"type": "array", "items": {"type": "string"}, "description": "Optional specific parameter names."},
            "include_all_parameters": {"type": "boolean", "description": "If true, returns all discoverable instance/type parameter names and values."},
            "populated_only": {"type": "boolean", "description": "If true, omit empty/'Not available' values. Works best with include_all_parameters=true."}
        },
        "required": []
    }
    UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM = "Updates parameter values for elements. Prefer result_handle + parameter_name + new_value for bulk updates after filtering."
    UPDATE_ELEMENT_PARAMETERS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "element_id": {"type": "string", "description": "Element ID to update"},
                        "parameters": {
                            "type": "object",
                            "description": "Object with parameter names as keys and new values as values (e.g., {'Sill Height': '2\' 6\"', 'Comments': 'Updated'})"
                        }
                    },
                    "required": ["element_id", "parameters"]
                },
                "description": "Array of element updates"
            },
            "element_ids": {"type": "array", "items": {"type": "string"}, "description": "Element IDs for simplified bulk updates (small sets)."},
            "result_handle": {"type": "string", "description": "Preferred handle for bulk updates after filtering."},
            "parameter_name": {"type": "string", "description": "Parameter name to set (e.g., 'Install Level')"},
            "new_value": {"type": "string", "description": "New value to apply to the specified parameter"}
        },
        "description": "Provide either detailed 'updates', or (result_handle + parameter_name + new_value), or (element_ids + parameter_name + new_value)."
    }


    # Sheet and view management tool descriptions and parameters
    PLACE_VIEW_ON_SHEET_TOOL_DESCRIPTION_FOR_LLM = "Places a view onto a new sheet by view name. Creates a new sheet with automatic numbering based on view type (D001 for details, S001 for sections, P001 for floor plans, etc.) and places the view in the center of the sheet. Supports fuzzy matching for view names when exact_match=False."
    PLACE_VIEW_ON_SHEET_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "view_name": {"type": "string", "description": "Name of the view to place on the sheet. Can be partial name if exact_match=False"},
            "exact_match": {"type": "boolean", "description": "Whether to require exact view name match (default: False for fuzzy matching)"}
        },
        "required": ["view_name"]
    }
    
    LIST_VIEWS_TOOL_DESCRIPTION_FOR_LLM = "Lists all views in the current Revit document that can be placed on sheets. Returns view names, types, IDs, and whether they're already placed on sheets. Use this to discover available views before placing them."
    LIST_VIEWS_TOOL_PARAMETERS_FOR_LLM = {"type": "object", "properties": {}}

    PLANNER_TOOL_DESCRIPTION_FOR_LLM = "Executes a sequence of tools based on a planned workflow. The LLM should first analyze the user request, then provide a step-by-step execution plan."
    PLANNER_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "user_request": {"type": "string", "description": "The original user request"},
            "execution_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "The tool to execute"},
                        "params": {"type": "object", "description": "Parameters for the tool"},
                        "description": {"type": "string", "description": "What this step accomplishes"}
                    },
                    "required": ["tool", "params"]
                },
                "description": "List of planned steps"
            }
        },
        "required": ["user_request", "execution_plan"]
    }

    REVIT_TOOLS_SPEC_FOR_LLMS = {
        "openai": [
            {"type": "function", "function": {"name": REVIT_INFO_TOOL_NAME, "description": REVIT_INFO_TOOL_DESCRIPTION_FOR_LLM, "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": GET_SCHEMA_CONTEXT_TOOL_NAME, "description": GET_SCHEMA_CONTEXT_TOOL_DESCRIPTION_FOR_LLM, "parameters": GET_SCHEMA_CONTEXT_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": RESOLVE_TARGETS_TOOL_NAME, "description": RESOLVE_TARGETS_TOOL_DESCRIPTION_FOR_LLM, "parameters": RESOLVE_TARGETS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": GET_ELEMENTS_BY_CATEGORY_TOOL_NAME, "description": GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM, "parameters": GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": SELECT_ELEMENTS_TOOL_NAME, "description": SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": SELECT_STORED_ELEMENTS_TOOL_NAME, "description": SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": LIST_STORED_ELEMENTS_TOOL_NAME, "description": LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": FILTER_ELEMENTS_TOOL_NAME, "description": FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME, "description": FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_DESCRIPTION_FOR_LLM, "parameters": FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": GET_ELEMENT_PROPERTIES_TOOL_NAME, "description": GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM, "parameters": GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": UPDATE_ELEMENT_PARAMETERS_TOOL_NAME, "description": UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM, "parameters": UPDATE_ELEMENT_PARAMETERS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": PLACE_VIEW_ON_SHEET_TOOL_NAME, "description": PLACE_VIEW_ON_SHEET_TOOL_DESCRIPTION_FOR_LLM, "parameters": PLACE_VIEW_ON_SHEET_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": LIST_VIEWS_TOOL_NAME, "description": LIST_VIEWS_TOOL_DESCRIPTION_FOR_LLM, "parameters": LIST_VIEWS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": PLANNER_TOOL_NAME, "description": PLANNER_TOOL_DESCRIPTION_FOR_LLM, "parameters": PLANNER_TOOL_PARAMETERS_FOR_LLM}},
        ],
        "anthropic": [
            {"name": REVIT_INFO_TOOL_NAME, "description": REVIT_INFO_TOOL_DESCRIPTION_FOR_LLM, "input_schema": {"type": "object", "properties": {}}},
            {"name": GET_SCHEMA_CONTEXT_TOOL_NAME, "description": GET_SCHEMA_CONTEXT_TOOL_DESCRIPTION_FOR_LLM, "input_schema": GET_SCHEMA_CONTEXT_TOOL_PARAMETERS_FOR_LLM},
            {"name": RESOLVE_TARGETS_TOOL_NAME, "description": RESOLVE_TARGETS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": RESOLVE_TARGETS_TOOL_PARAMETERS_FOR_LLM},
            {"name": GET_ELEMENTS_BY_CATEGORY_TOOL_NAME, "description": GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM, "input_schema": GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM},
            {"name": SELECT_ELEMENTS_TOOL_NAME, "description": SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": SELECT_STORED_ELEMENTS_TOOL_NAME, "description": SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": LIST_STORED_ELEMENTS_TOOL_NAME, "description": LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": FILTER_ELEMENTS_TOOL_NAME, "description": FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME, "description": FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_DESCRIPTION_FOR_LLM, "input_schema": FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_PARAMETERS_FOR_LLM},
            {"name": GET_ELEMENT_PROPERTIES_TOOL_NAME, "description": GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM, "input_schema": GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM},
            {"name": UPDATE_ELEMENT_PARAMETERS_TOOL_NAME, "description": UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": UPDATE_ELEMENT_PARAMETERS_TOOL_PARAMETERS_FOR_LLM},
            {"name": PLACE_VIEW_ON_SHEET_TOOL_NAME, "description": PLACE_VIEW_ON_SHEET_TOOL_DESCRIPTION_FOR_LLM, "input_schema": PLACE_VIEW_ON_SHEET_TOOL_PARAMETERS_FOR_LLM},
            {"name": LIST_VIEWS_TOOL_NAME, "description": LIST_VIEWS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": LIST_VIEWS_TOOL_PARAMETERS_FOR_LLM},
            {"name": PLANNER_TOOL_NAME, "description": PLANNER_TOOL_DESCRIPTION_FOR_LLM, "input_schema": PLANNER_TOOL_PARAMETERS_FOR_LLM},
        ],
        "google": [
            google_types.Tool(function_declarations=[
                google_types.FunctionDeclaration(name=REVIT_INFO_TOOL_NAME, description=REVIT_INFO_TOOL_DESCRIPTION_FOR_LLM, parameters={"type": "object", "properties": {}}),
                google_types.FunctionDeclaration(name=GET_SCHEMA_CONTEXT_TOOL_NAME, description=GET_SCHEMA_CONTEXT_TOOL_DESCRIPTION_FOR_LLM, parameters=GET_SCHEMA_CONTEXT_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=RESOLVE_TARGETS_TOOL_NAME, description=RESOLVE_TARGETS_TOOL_DESCRIPTION_FOR_LLM, parameters=RESOLVE_TARGETS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=GET_ELEMENTS_BY_CATEGORY_TOOL_NAME, description=GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM, parameters=GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=SELECT_ELEMENTS_TOOL_NAME, description=SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=SELECT_STORED_ELEMENTS_TOOL_NAME, description=SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=LIST_STORED_ELEMENTS_TOOL_NAME, description=LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=FILTER_ELEMENTS_TOOL_NAME, description=FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME, description=FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_DESCRIPTION_FOR_LLM, parameters=FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=GET_ELEMENT_PROPERTIES_TOOL_NAME, description=GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM, parameters=GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=UPDATE_ELEMENT_PARAMETERS_TOOL_NAME, description=UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM, parameters=UPDATE_ELEMENT_PARAMETERS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=PLACE_VIEW_ON_SHEET_TOOL_NAME, description=PLACE_VIEW_ON_SHEET_TOOL_DESCRIPTION_FOR_LLM, parameters=PLACE_VIEW_ON_SHEET_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=LIST_VIEWS_TOOL_NAME, description=LIST_VIEWS_TOOL_DESCRIPTION_FOR_LLM, parameters=LIST_VIEWS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=PLANNER_TOOL_NAME, description=PLANNER_TOOL_DESCRIPTION_FOR_LLM, parameters=PLANNER_TOOL_PARAMETERS_FOR_LLM),
            ])
        ]
    }
    app.logger.info("Manual tool specs for LLMs defined.")

    ANTHROPIC_MODEL_ID_MAP = {
        # Current model IDs
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-opus-4-6": "claude-opus-4-6",
        "claude-haiku-4-5": "claude-haiku-4-5",
        # Backward compatibility with older saved UI values
        "claude-4-sonnet": "claude-sonnet-4-6",
        "claude-4-opus": "claude-opus-4-6",
        "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    }
    app.logger.info("Configuration loaded.")

    def _run_schema_warmup():
        """Warm schema cache in a background thread so startup never blocks."""
        try:
            schema_warmup = get_revit_schema_context_mcp_tool(force_refresh=True)
            if schema_warmup.get("status") == "success":
                app.logger.info("Schema context cache warmup succeeded.")
            else:
                app.logger.warning("Schema context cache warmup skipped: %s", schema_warmup.get("message"))
        except Exception as schema_warmup_err:
            app.logger.warning("Schema context cache warmup failed: %s", schema_warmup_err)

    warm_schema_on_startup = os.environ.get("REVITMCP_WARM_SCHEMA_ON_STARTUP", "true").strip().lower() in ("1", "true", "yes", "on")
    if warm_schema_on_startup:
        threading.Thread(target=_run_schema_warmup, name="revitmcp-schema-warmup", daemon=True).start()
        app.logger.info("Schema context warmup launched in background thread.")
    else:
        app.logger.info("Schema context warmup disabled via REVITMCP_WARM_SCHEMA_ON_STARTUP.")

    @app.route('/', methods=['GET'])
    def chat_ui():
        app.logger.info("Serving chat_ui (index.html)")
        return render_template('index.html')

    @app.route('/test_log', methods=['GET'])
    def test_log_route():
        app.logger.info("--- ACCESSED /test_log route successfully (app.logger.info) ---")
        return jsonify({"status": "success", "message": "Test log route accessed. Check server console."}), 200

    @app.route('/chat_api', methods=['POST'])
    def chat_api():
        data = request.json
        conversation_history = data.get('conversation')
        api_key = data.get('apiKey')
        selected_model_ui_name = data.get('model')
        # user_message_content = conversation_history[-1]['content'].strip() # Not directly used anymore for dispatch
        
        # Add planning guidance system prompt
        planning_system_prompt = {
            "role": "system", 
            "content": """You are a Revit automation assistant with planning capabilities.

PLANNING APPROACH:
For complex requests, use the plan_and_execute_workflow tool which allows you to:
1. Analyze the user request 
2. Plan a sequence of steps using available tools
3. Execute all steps in one operation
4. Return complete results

AVAILABLE TOOLS FOR PLANNING:
- get_revit_project_info: Get project information (no params)
- get_revit_schema_context: Get canonical categories/levels/families/parameters (optional force_refresh)
- resolve_revit_targets: Resolve user terms to exact Revit names (params: query_terms)
- get_elements_by_category: Get all elements by category (params: category_name)
- filter_elements: Advanced filtering (params: category_name, level_name, parameters)
- filter_stored_elements_by_parameter: Server-side batched parameter filtering on stored sets (params: result_handle/category_name, parameter_name, value, operator)
- get_element_properties: Get parameter values (params: result_handle OR element_ids, parameter_names, include_all_parameters, populated_only)
- update_element_parameters: Update parameters (params: updates OR result_handle + parameter_name + new_value)
- select_elements_by_id: Select specific elements (params: element_ids or result_handle)
- select_stored_elements: Select stored elements (params: result_handle or category_name)
- list_stored_elements: List available stored categories (no params)
- place_view_on_sheet: Place view on new sheet (params: view_name, exact_match)
- list_views: List all available views (no params)

EXECUTION PLAN FORMAT:
[
  {
    "tool": "filter_elements",
    "params": {"category_name": "Windows", "level_name": "L5", "parameters": [{"name": "Sill Height", "value": "2' 3\"", "condition": "equals"}]},
    "description": "Find windows on L5 with sill height 2'3\""
  },
  {
    "tool": "update_element_parameters", 
    "params": {"result_handle": "${step_1_result_handle}", "parameter_name": "Sill Height", "new_value": "2' 6\""},
    "description": "Update sill height to 2'6\""
  }
]

WORKFLOW EXAMPLES:
- Parameter updates: filter_elements → update_element_parameters → select_stored_elements
- Property inspection: get_elements_by_category → filter_stored_elements_by_parameter → get_element_properties
- Element discovery: get_elements_by_category → filter_stored_elements_by_parameter → select_stored_elements

IMPORTANT:
- Always resolve terms with resolve_revit_targets before filter_elements or update_element_parameters.
- For large sets, prefer filter_stored_elements_by_parameter before get_element_properties.
- Do not use raw user category/level names directly.

Use plan_and_execute_workflow for multi-step operations to provide complete results in one response."""
        }
        
        def execute_tool_call(tool_name, function_args):
            """Execute a tool safely and return its result dictionary."""
            normalized_args = function_args or {}
            app.logger.info(f"Executing tool '{tool_name}' with args: {normalized_args}")

            try:
                if tool_name == REVIT_INFO_TOOL_NAME:
                    tool_result_data = get_revit_project_info_mcp_tool()
                elif tool_name == GET_SCHEMA_CONTEXT_TOOL_NAME:
                    tool_result_data = get_revit_schema_context_tool(force_refresh=normalized_args.get("force_refresh", False))
                elif tool_name == RESOLVE_TARGETS_TOOL_NAME:
                    tool_result_data = resolve_revit_targets_tool(query_terms=normalized_args.get("query_terms", {}))
                elif tool_name == GET_ELEMENTS_BY_CATEGORY_TOOL_NAME:
                    tool_result_data = get_elements_by_category_mcp_tool(category_name=normalized_args.get("category_name"))
                elif tool_name == SELECT_ELEMENTS_TOOL_NAME:
                    tool_result_data = select_elements_by_id_mcp_tool(
                        element_ids=normalized_args.get("element_ids", []),
                        result_handle=normalized_args.get("result_handle")
                    )
                elif tool_name == SELECT_STORED_ELEMENTS_TOOL_NAME:
                    tool_result_data = select_stored_elements_mcp_tool(
                        category_name=normalized_args.get("category_name"),
                        result_handle=normalized_args.get("result_handle")
                    )
                elif tool_name == LIST_STORED_ELEMENTS_TOOL_NAME:
                    tool_result_data = list_stored_elements_mcp_tool()
                elif tool_name == FILTER_ELEMENTS_TOOL_NAME:
                    tool_result_data = filter_elements_mcp_tool(
                        category_name=normalized_args.get("category_name"),
                        level_name=normalized_args.get("level_name"),
                        parameters=normalized_args.get("parameters", [])
                    )
                elif tool_name == FILTER_STORED_ELEMENTS_BY_PARAMETER_TOOL_NAME:
                    tool_result_data = filter_stored_elements_by_parameter_mcp_tool(
                        parameter_name=normalized_args.get("parameter_name"),
                        value=normalized_args.get("value"),
                        operator=normalized_args.get("operator", "contains"),
                        result_handle=normalized_args.get("result_handle"),
                        category_name=normalized_args.get("category_name"),
                        batch_size=normalized_args.get("batch_size"),
                        case_sensitive=normalized_args.get("case_sensitive", False)
                    )
                elif tool_name == GET_ELEMENT_PROPERTIES_TOOL_NAME:
                    tool_result_data = get_element_properties_mcp_tool(
                        element_ids=normalized_args.get("element_ids", []),
                        parameter_names=normalized_args.get("parameter_names", []),
                        result_handle=normalized_args.get("result_handle"),
                        include_all_parameters=normalized_args.get("include_all_parameters", False),
                        populated_only=normalized_args.get("populated_only", False)
                    )
                elif tool_name == UPDATE_ELEMENT_PARAMETERS_TOOL_NAME:
                    tool_result_data = update_element_parameters_mcp_tool(
                        updates=normalized_args.get("updates"),
                        element_ids=normalized_args.get("element_ids"),
                        result_handle=normalized_args.get("result_handle"),
                        parameter_name=normalized_args.get("parameter_name"),
                        new_value=normalized_args.get("new_value")
                    )
                elif tool_name == PLACE_VIEW_ON_SHEET_TOOL_NAME:
                    tool_result_data = place_view_on_sheet_mcp_tool(
                        view_name=normalized_args.get("view_name"),
                        exact_match=normalized_args.get("exact_match", False)
                    )
                elif tool_name == LIST_VIEWS_TOOL_NAME:
                    tool_result_data = list_views_mcp_tool()
                elif tool_name == PLANNER_TOOL_NAME:
                    tool_result_data = plan_and_execute_workflow_tool(
                        user_request=normalized_args.get("user_request"),
                        execution_plan=normalized_args.get("execution_plan", [])
                    )
                else:
                    app.logger.warning(f"Unknown tool '{tool_name}' requested by LLM.")
                    tool_result_data = {"status": "error", "message": f"Unknown tool '{tool_name}' requested by LLM."}
            except Exception as tool_exc:
                app.logger.error(f"Exception while executing tool '{tool_name}': {tool_exc}", exc_info=True)
                tool_result_data = {"status": "error", "message": f"Exception while executing tool '{tool_name}': {tool_exc}"}

            return tool_result_data

        final_response_to_frontend = {}
        image_output_for_frontend = None # To store image data if a tool returns it
        model_reply_text = "" # The final text reply from the LLM
        error_message_for_frontend = None

        try:
            if selected_model_ui_name == 'echo_model':
                model_reply_text = f"Echo: {conversation_history[-1]['content']}"
            
            # --- OpenAI Models ---
            elif selected_model_ui_name.startswith('gpt-') or selected_model_ui_name.startswith('o'):
                client = openai.OpenAI(api_key=api_key)
                messages_for_llm = [planning_system_prompt] + [{"role": "assistant" if msg['role'] == 'bot' else msg['role'], "content": msg['content']} for msg in conversation_history]
                max_tool_iterations = 5
                iteration = 0

                while iteration < max_tool_iterations:
                    iteration += 1
                    app.logger.debug(f"OpenAI (iteration {iteration}): Sending messages: {messages_for_llm}")
                    completion = client.chat.completions.create(
                        model=selected_model_ui_name,
                        messages=messages_for_llm,
                        tools=REVIT_TOOLS_SPEC_FOR_LLMS['openai'],
                        tool_choice="auto"
                    )
                    response_message = completion.choices[0].message
                    messages_for_llm.append(response_message)
                    tool_calls = response_message.tool_calls or []

                    if tool_calls:
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            try:
                                raw_arguments = tool_call.function.arguments or "{}"
                                function_args = json.loads(raw_arguments)
                            except json.JSONDecodeError as e:
                                app.logger.error(f"OpenAI: Failed to parse function arguments for {function_name}: {tool_call.function.arguments}. Error: {e}")
                                tool_result_data = {"status": "error", "message": f"Invalid arguments from LLM for tool {function_name}."}
                            else:
                                tool_result_data = execute_tool_call(function_name, function_args)

                            messages_for_llm.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(tool_result_data)
                            })
                        continue

                    model_reply_text = response_message.content or ""
                    break
                else:
                    app.logger.warning("OpenAI: Reached tool iteration limit without final response.")
                    model_reply_text = "Reached tool execution limit without a final response."

            # --- Anthropic Models ---
            elif selected_model_ui_name.startswith('claude-'):
                client = anthropic.Anthropic(api_key=api_key)
                actual_anthropic_model_id = ANTHROPIC_MODEL_ID_MAP.get(selected_model_ui_name, selected_model_ui_name)
                system_prompt_content = planning_system_prompt["content"]
                messages_for_llm = [{"role": "assistant" if msg['role'] == 'bot' else msg['role'], "content": msg['content']} for msg in conversation_history]
                max_tool_iterations = 5
                iteration = 0

                while iteration < max_tool_iterations:
                    iteration += 1
                    app.logger.debug(f"Anthropic (iteration {iteration}): Sending messages: {messages_for_llm}")
                    response = client.messages.create(
                        model=actual_anthropic_model_id,
                        max_tokens=3000,
                        system=system_prompt_content,
                        messages=messages_for_llm,
                        tools=REVIT_TOOLS_SPEC_FOR_LLMS['anthropic'],
                        tool_choice={"type": "auto"}
                    )
                    messages_for_llm.append({"role": "assistant", "content": response.content})

                    tool_results_for_turn = []
                    for response_block in response.content:
                        if getattr(response_block, 'type', None) == 'tool_use':
                            tool_name = response_block.name
                            function_args = response_block.input if isinstance(response_block.input, dict) else {}
                            app.logger.info(f"Anthropic: Tool use requested: {tool_name}, Input: {function_args}")
                            tool_result_data = execute_tool_call(tool_name, function_args)
                            tool_results_for_turn.append({
                                "type": "tool_result",
                                "tool_use_id": response_block.id,
                                "content": json.dumps(tool_result_data)
                            })

                    if tool_results_for_turn:
                        messages_for_llm.append({"role": "user", "content": tool_results_for_turn})
                        continue

                    text_parts = [block.text for block in response.content if getattr(block, 'type', None) == 'text' and getattr(block, 'text', None)]
                    if text_parts:
                        model_reply_text = ''.join(text_parts)
                    else:
                        model_reply_text = "Anthropic model responded without text content after tool execution."
                    break
                else:
                    app.logger.warning("Anthropic: Reached tool iteration limit without final response.")
                    model_reply_text = "Reached tool execution limit without a final response."

            # --- Google Gemini Models ---
            elif selected_model_ui_name.startswith('gemini-'):
                genai.configure(api_key=api_key)
                gemini_tool_config = google_types.ToolConfig(
                    function_calling_config=google_types.FunctionCallingConfig(
                        mode=google_types.FunctionCallingConfig.Mode.AUTO
                    )
                )
                model = genai.GenerativeModel(
                    selected_model_ui_name,
                    tools=REVIT_TOOLS_SPEC_FOR_LLMS['google'],
                    tool_config=gemini_tool_config,
                    system_instruction=planning_system_prompt["content"]
                )

                gemini_history_for_chat = []
                for msg in conversation_history:
                    role = 'user' if msg['role'] == 'user' else 'model'
                    gemini_history_for_chat.append({'role': role, 'parts': [google_types.Part(text=msg['content'])]})

                if gemini_history_for_chat:
                    current_user_prompt_parts = gemini_history_for_chat.pop()['parts']
                else:
                    current_user_prompt_parts = [google_types.Part(text=conversation_history[-1]['content'])]

                chat_session = model.start_chat(history=gemini_history_for_chat)
                app.logger.debug(f"Google: Sending prompt parts: {current_user_prompt_parts} with history count: {len(chat_session.history)}")

                gemini_response = chat_session.send_message(current_user_prompt_parts)
                max_tool_iterations = 5

                for iteration in range(1, max_tool_iterations + 1):
                    candidate = gemini_response.candidates[0]
                    function_part = next((part for part in candidate.content.parts if getattr(part, 'function_call', None)), None)

                    if function_part:
                        function_name = function_part.function_call.name
                        function_args = dict(function_part.function_call.args)
                        app.logger.info(f"Google: Function call requested: {function_name} with args {function_args}")
                        tool_result_data = execute_tool_call(function_name, function_args)
                        function_response_part = google_types.Part(
                            function_response=google_types.FunctionResponse(
                                name=function_name,
                                response=tool_result_data
                            )
                        )
                        app.logger.debug(f"Google: Resending with tool response for iteration {iteration}.")
                        gemini_response = chat_session.send_message([function_response_part])
                        continue

                    text_output = ''.join(part.text for part in candidate.content.parts if getattr(part, 'text', None))
                    model_reply_text = text_output or gemini_response.text
                    break
                else:
                    app.logger.warning("Google: Reached tool iteration limit without final response.")
                    model_reply_text = "Reached tool execution limit without a final response."
            else:
                error_message_for_frontend = f"Model '{selected_model_ui_name}' is not recognized or supported."

        except openai.APIConnectionError as e:
            error_message_for_frontend = f"OpenAI Connection Error: {e}. Please check network or API key."
            app.logger.error(f"OpenAI Connection Error: {e}", exc_info=True)
        except openai.AuthenticationError as e:
            error_message_for_frontend = f"OpenAI Authentication Error: {e}. Invalid API Key?"
            app.logger.error(f"OpenAI Authentication Error: {e}", exc_info=True)
        except openai.RateLimitError as e:
            error_message_for_frontend = f"OpenAI Rate Limit Error: {e}. Please try again later."
            app.logger.error(f"OpenAI Rate Limit Error: {e}", exc_info=True)
        except openai.APIError as e:
            error_message_for_frontend = f"OpenAI API Error: {e} (Status: {e.status_code if hasattr(e, 'status_code') else 'N/A'})."
            app.logger.error(f"OpenAI API Error: {e}", exc_info=True)
        except anthropic.APIConnectionError as e:
            error_message_for_frontend = f"Anthropic Connection Error: {e}. Please check network or API key."
            app.logger.error(f"Anthropic Connection Error: {e}", exc_info=True)
        except anthropic.AuthenticationError as e:
            error_message_for_frontend = f"Anthropic Authentication Error: {e}. Invalid API Key?"
            app.logger.error(f"Anthropic Authentication Error: {e}", exc_info=True)
        except anthropic.RateLimitError as e:
            error_message_for_frontend = f"Anthropic Rate Limit Error: {e}. Please try again later."
            app.logger.error(f"Anthropic Rate Limit Error: {e}", exc_info=True)
        except anthropic.APIError as e: # Catch generic Anthropic API errors
            error_message_for_frontend = f"Anthropic API Error: {e} (Status: {e.status_code if hasattr(e, 'status_code') else 'N/A'})."
            app.logger.error(f"Anthropic API Error: {e}", exc_info=True)
        except Exception as e: # General fallback for other LLM or unexpected errors
            error_message_for_frontend = f"An unexpected error occurred: {str(e)}"
            app.logger.error(f"Chat API error: {type(e).__name__} - {str(e)}", exc_info=True)

        final_response_to_frontend["reply"] = model_reply_text
        if image_output_for_frontend:
            final_response_to_frontend["image_output"] = image_output_for_frontend
        
        if error_message_for_frontend and not model_reply_text:
            return jsonify({"error": error_message_for_frontend}), 500
        elif error_message_for_frontend:
            final_response_to_frontend["error_detail"] = error_message_for_frontend
            return jsonify(final_response_to_frontend) # Include error if model also gave partial reply
        else:
            return jsonify(final_response_to_frontend)

    @app.route('/send_revit_command', methods=['POST'])
    def send_revit_command():
        client_request_data = request.json
        if not client_request_data or "command" not in client_request_data:
            return jsonify({"status": "error", "message": "Invalid request. 'command' is required."}), 400
        revit_command_payload = client_request_data
        actual_revit_listener_url = "http://localhost:8001" 
        app.logger.info(f"External Server (/send_revit_command): Forwarding {revit_command_payload} to {actual_revit_listener_url}")
        try:
            response_from_revit = requests.post(actual_revit_listener_url, json=revit_command_payload, headers={'Content-Type': 'application/json'}, timeout=30)
            response_from_revit.raise_for_status()
            revit_response_data = response_from_revit.json()
            app.logger.info(f"External Server: Response from Revit Listener: {revit_response_data}")
            return jsonify(revit_response_data), response_from_revit.status_code
        except requests.exceptions.ConnectionError as e:
            msg = f"Could not connect to Revit Listener at {actual_revit_listener_url}. Error: {e}"
            app.logger.error(msg)
            return jsonify({"status": "error", "message": msg}), 503
        except requests.exceptions.Timeout as e:
            msg = f"Request to Revit Listener timed out. Error: {e}"
            app.logger.error(msg)
            return jsonify({"status": "error", "message": msg}), 504
        except requests.exceptions.RequestException as e:
            msg = f"Error communicating with Revit Listener. Error: {e}"
            app.logger.error(msg)
            details = "No response details."
            if hasattr(e, 'response') and e.response is not None:
                try: details = e.response.json()
                except ValueError: details = e.response.text
            status = e.response.status_code if hasattr(e, 'response') and e.response is not None else 500
            return jsonify({"status": "error", "message": msg, "details": details}), status
        except Exception as e:
            msg = f"Unexpected error in /send_revit_command. Error: {e}"
            app.logger.error(msg, exc_info=True)
            return jsonify({"status": "error", "message": msg}), 500

    # Add a pause for debugging console window issues
    print("--- server.py script execution reached near end (before __main__ check) ---")
    # input("Press Enter to continue launching Flask server...") # Python 3

    if __name__ == '__main__':
        runtime_surface = resolve_runtime_surface(sys.argv[1:])
        startup_logger.info("--- Runtime surface selected: %s ---", runtime_surface)
        print(f"--- Runtime surface selected: {runtime_surface} ---")

        try:
            if runtime_surface == 'mcp':
                startup_logger.info("--- Starting FastMCP server over stdio transport ---")
                mcp_server.run(transport='stdio')
                startup_logger.info("FastMCP server exited normally.")
            else:
                startup_logger.info(f"--- Starting Flask development server on host {HOST}, port {PORT} ---")
                print(f"--- Debug mode for app.run is: {DEBUG_MODE} ---")
                app.run(debug=DEBUG_MODE, port=PORT, host=HOST)
                startup_logger.info("Flask app.run() exited normally.")
        except OSError as e_os:
            startup_logger.error(f"OS Error during server startup (app.run): {e_os}", exc_info=True)
            print(f"OS Error: {e_os}")
        except Exception as e_main_run:
            startup_logger.error(f"Unexpected error during server startup (app.run in __main__): {e_main_run}", exc_info=True)
            print(f"Unexpected error: {e_main_run}")

except Exception as e_global:
    startup_logger.error("!!!!!!!!!! GLOBAL SCRIPT EXECUTION ERROR !!!!!!!!!!", exc_info=True)
    sys.stderr.write(f"GLOBAL SCRIPT ERROR: {e_global}\n{traceback.format_exc()}\n")
    sys.stderr.write(f"Check '{STARTUP_LOG_FILE}' for details.\n")
finally:
    startup_logger.info("--- Server script execution finished or encountered a global error ---")
    # input("Server.py finished or errored. Press Enter to close window...") # Python 3 
