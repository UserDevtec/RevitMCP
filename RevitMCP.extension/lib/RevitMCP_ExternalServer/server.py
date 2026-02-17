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
    # This stores element IDs from searches so they can be accurately referenced later
    element_storage = {}  # Format: {"category_name": {"element_ids": [...], "count": N, "timestamp": "..."}}
    
    MAX_ELEMENTS_FOR_SELECTION = 250

    def store_elements(category_name: str, element_ids: list, count: int) -> str:
        """Store element IDs for a category and return a storage key."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        storage_key = category_name.lower().replace(" ", "_")
        element_storage[storage_key] = {
            "element_ids": element_ids,
            "count": count,
            "category": category_name,
            "timestamp": timestamp
        }
        app.logger.info(f"Stored {count} element IDs for category '{category_name}' with key '{storage_key}'")
        return storage_key
    
    def get_stored_elements(storage_key: str) -> dict:
        """Retrieve stored element IDs by storage key."""
        storage_key = storage_key.lower().replace(" ", "_")
        if storage_key in element_storage:
            return element_storage[storage_key]
        return None
    
    def list_stored_categories() -> dict:
        """List all currently stored categories."""
        return {key: {"category": data["category"], "count": data["count"], "timestamp": data["timestamp"]} 
                for key, data in element_storage.items()}

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
    GET_ELEMENT_PROPERTIES_TOOL_NAME = "get_element_properties"
    UPDATE_ELEMENT_PARAMETERS_TOOL_NAME = "update_element_parameters"
    
    # Sheet and view management tools
    PLACE_VIEW_ON_SHEET_TOOL_NAME = "place_view_on_sheet"
    LIST_VIEWS_TOOL_NAME = "list_views"

    # Replace batch workflow with generic planner
    PLANNER_TOOL_NAME = "plan_and_execute_workflow"

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

    # --- MCP Tool Definitions using @mcp_server.tool() ---
    @mcp_server.tool(name=REVIT_INFO_TOOL_NAME) # Name must match what LLM will use
    def get_revit_project_info_mcp_tool() -> dict:
        """Retrieves detailed information about the currently open Revit project."""
        app.logger.info(f"MCP Tool executed: {REVIT_INFO_TOOL_NAME}")
        return call_revit_listener(command_path='/project_info', method='GET')

    @mcp_server.tool(name=GET_ELEMENTS_BY_CATEGORY_TOOL_NAME)
    def get_elements_by_category_mcp_tool(category_name: str) -> dict:
        """Retrieves all elements in the Revit model belonging to the specified category, returning their IDs and names. Automatically stores the results for later selection."""
        app.logger.info(f"MCP Tool executed: {GET_ELEMENTS_BY_CATEGORY_TOOL_NAME} with category_name: {category_name}")
        result = call_revit_listener(command_path='/get_elements_by_category', method='POST', payload_data={"category_name": category_name})
        
        # Automatically store the results if successful
        if result.get("status") == "success" and "element_ids" in result:
            storage_key = store_elements(category_name, result["element_ids"], result.get("count", len(result["element_ids"])))
            result["stored_as"] = storage_key
            result["storage_message"] = f"Results stored as '{storage_key}' - use select_stored_elements to select these elements"
        
        return result

    @mcp_server.tool(name=SELECT_ELEMENTS_TOOL_NAME)
    def select_elements_by_id_mcp_tool(element_ids: list[str]) -> dict:
        """Selects one or more elements in Revit using their Element IDs."""
        app.logger.info(f"MCP Tool executed: {SELECT_ELEMENTS_TOOL_NAME} with element_ids: {element_ids}")
        
        # Ensure element_ids is a list, even if a single string ID is passed by the LLM
        if isinstance(element_ids, str):
            app.logger.warning(f"select_elements_by_id_mcp_tool: element_ids was a string ('{element_ids}'), converting to list.")
            processed_element_ids = [element_ids]
        elif isinstance(element_ids, list) and all(isinstance(eid, str) for eid in element_ids):
            processed_element_ids = element_ids
        elif isinstance(element_ids, list): # List contains non-strings, attempt to convert or log error
            app.logger.warning(f"select_elements_by_id_mcp_tool: element_ids list contained non-string items: {element_ids}. Attempting to convert all to strings.")
            try:
                processed_element_ids = [str(eid) for eid in element_ids]
            except Exception as e_conv:
                app.logger.error(f"select_elements_by_id_mcp_tool: Failed to convert all items in element_ids to string: {e_conv}")
                return {"status": "error", "message": f"Invalid format for element_ids. All IDs must be strings. Received: {element_ids}"}
        else:
            app.logger.error(f"select_elements_by_id_mcp_tool: element_ids is not a string or a list of strings. Received type: {type(element_ids)}, value: {element_ids}")
            return {"status": "error", "message": f"Invalid input type for element_ids. Expected string or list of strings. Received: {type(element_ids)}"}

        return call_revit_listener(command_path='/select_elements_by_id', method='POST', payload_data={"element_ids": processed_element_ids})

    @mcp_server.tool(name=SELECT_STORED_ELEMENTS_TOOL_NAME)
    def select_stored_elements_mcp_tool(category_name: str) -> dict:
        """Selects elements that were previously retrieved and stored by get_elements_by_category or filter_elements. Use the category name (e.g., 'windows', 'doors') to select the stored elements."""
        app.logger.info(f"MCP Tool executed: {SELECT_STORED_ELEMENTS_TOOL_NAME} with category_name: {category_name}")
        
        # Normalize the input category name
        normalized_category = category_name.lower().replace("ost_", "").replace(" ", "_")
        
        # Strategy 1: Try exact match first
        stored_data = get_stored_elements(normalized_category)
        
        # Strategy 2: If exact match fails, try to find partial matches
        if not stored_data:
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
                "message": f"No stored elements found for category '{category_name}'. Available stored categories: {available_keys}",
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
            result["source"] = f"stored_{category_name}"
            result["stored_count"] = stored_data["count"]
            result["stored_at"] = stored_data["timestamp"]
            result["matched_key"] = stored_data.get("category", "unknown")
            result["approach_note"] = "Focused selection - elements should remain active for user operations"

        return result

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
        
        payload = {"category_name": category_name}
        if level_name:
            payload["level_name"] = level_name
        if parameters:
            payload["parameters"] = parameters
        
        result = call_revit_listener(command_path='/elements/filter', method='POST', payload_data=payload)

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
            
            stored_key = store_elements(storage_key, result["element_ids"], result.get("count", len(result["element_ids"])))
            result["stored_as"] = stored_key
            result["storage_message"] = f"Filtered results stored as '{stored_key}' - use select_stored_elements to select these elements"
        
        return result

    @mcp_server.tool(name=GET_ELEMENT_PROPERTIES_TOOL_NAME)
    def get_element_properties_mcp_tool(element_ids: list[str], parameter_names: list[str] = None) -> dict:
        """Gets parameter values for specified elements. If parameter_names not provided, returns common parameters for the element category."""
        app.logger.info(f"MCP Tool executed: {GET_ELEMENT_PROPERTIES_TOOL_NAME} with {len(element_ids)} elements")
        
        payload = {"element_ids": element_ids}
        if parameter_names:
            payload["parameter_names"] = parameter_names
        
        result = call_revit_listener(command_path='/elements/get_properties', method='POST', payload_data=payload)
        if result.get("status") == "error" and _is_route_not_defined(result, "/elements/get_properties"):
            return {
                "status": "error",
                "error_type": "route_not_defined",
                "message": "The active Revit route set does not support '/elements/get_properties'. Reload/update the Revit extension to enable property reads."
            }
        return result

    @mcp_server.tool(name=UPDATE_ELEMENT_PARAMETERS_TOOL_NAME)
    def update_element_parameters_mcp_tool(
        updates: list[dict] = None,
        element_ids: list[str] = None,
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

        elif element_ids and parameter_name and new_value is not None:
            if not isinstance(element_ids, list) or not element_ids:
                return {"status": "error", "message": "'element_ids' must be a non-empty list when using the simplified form."}
            parameter_name = str(parameter_name).strip()
            if not parameter_name:
                return {"status": "error", "message": "parameter_name cannot be empty."}
            normalized_value = str(new_value)
            for eid in element_ids:
                element_id = str(eid).strip()
                if not element_id:
                    return {"status": "error", "message": "All element_ids must be non-empty strings."}
                normalized_updates.append({"element_id": element_id, "parameters": {parameter_name: normalized_value}})
        else:
            return {"status": "error", "message": "Provide either 'updates' or (element_ids, parameter_name, new_value)."}

        app.logger.info(f"Prepared {len(normalized_updates)} parameter update(s) for execution.")

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
        return result

    
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
            "get_elements_by_category": lambda **kwargs: get_elements_by_category_mcp_tool(kwargs.get("category_name")),
            "filter_elements": lambda **kwargs: filter_elements_mcp_tool(
                kwargs.get("category_name"), 
                kwargs.get("level_name"), 
                kwargs.get("parameters", [])
            ),
            "get_element_properties": lambda **kwargs: get_element_properties_mcp_tool(
                kwargs.get("element_ids", []), 
                kwargs.get("parameter_names", [])
            ),
            "update_element_parameters": lambda **kwargs: update_element_parameters_mcp_tool(
                updates=kwargs.get("updates"),
                element_ids=kwargs.get("element_ids"),
                parameter_name=kwargs.get("parameter_name"),
                new_value=kwargs.get("new_value")
            ),
            "select_elements_by_id": lambda **kwargs: select_elements_by_id_mcp_tool(
                kwargs.get("element_ids", [])
            ),
            "select_stored_elements": lambda **kwargs: select_stored_elements_mcp_tool(
                kwargs.get("category_name")
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
                        if "element_ids" in result:
                            workflow_results[f"step_{i}_element_ids"] = result["element_ids"]
                        if "count" in result:
                            workflow_results[f"step_{i}_count"] = result["count"]
                        if "elements" in result:
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
    GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM = "Retrieves and stores all elements in the current Revit model for the specified category. Use this ONLY when the user wants to find/get elements. This automatically stores results for later selection. After calling this, if the user wants to select the elements, use select_stored_elements."
    GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object", "properties": {"category_name": {"type": "string", "description": "The name of the Revit category to retrieve elements from (e.g., 'OST_Windows', 'OST_Doors', 'OST_Walls' or simplified like 'Windows', 'Doors', 'Walls')."}}, "required": ["category_name"]
    }
    SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM = "DEPRECATED - Use select_stored_elements instead. Selects elements by exact IDs. Only use this if you have specific element IDs that are NOT from get_elements_by_category."
    SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object", "properties": {"element_ids": {"type": "array", "items": {"type": "string"}, "description": "An array of Element IDs (as strings) of the Revit elements to be selected."}}, "required": ["element_ids"]
    }
    SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM = "Selects elements that were previously retrieved by get_elements_by_category. Use this when the user wants to SELECT elements after using get_elements_by_category. When user says 'select windows', 'select doors', 'select them', etc., use this tool with the category name. Automatically zooms to show selected elements without changing the user's view."
    SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object", "properties": {"category_name": {"type": "string", "description": "The category name of the stored elements to select (e.g., 'windows', 'doors', 'walls'). Use the same category name that was used with get_elements_by_category."}}, "required": ["category_name"]
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
    GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM = "Gets parameter values for specified elements. Use this to read current values before updating or to display element properties to the user. IMPORTANT: This is typically used as a middle step - after filtering elements and before updating them. When user requests updates, chain: filter -> get_properties -> update_parameters in one turn."
    GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM = {
        "type": "object",
        "properties": {
            "element_ids": {"type": "array", "items": {"type": "string"}, "description": "Array of element IDs to get properties for"},
            "parameter_names": {"type": "array", "items": {"type": "string"}, "description": "Optional array of specific parameter names to retrieve (e.g., ['Sill Height', 'Width']). If not provided, gets common parameters."}
        },
        "required": ["element_ids"]
    }
    UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM = "Updates parameter values for elements. Use this to modify element properties like sill height, dimensions, comments, etc. You can either provide a detailed updates list or use the simplified form (element_ids + parameter_name + new_value) to push the same value to many elements. Always use this within a transaction context. IMPORTANT: This is typically the final step in an update workflow. When user requests parameter updates, chain: filter_elements -> get_element_properties -> update_element_parameters -> select_stored_elements in one conversation turn."
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
            "element_ids": {"type": "array", "items": {"type": "string"}, "description": "Element IDs to update when using the simplified bulk update form"},
            "parameter_name": {"type": "string", "description": "Parameter name to set (e.g., 'Install Level')"},
            "new_value": {"type": "string", "description": "New value to apply to the specified parameter"}
        },
        "description": "Provide either a detailed 'updates' array or element_ids + parameter_name + new_value for bulk updates."
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
            {"type": "function", "function": {"name": GET_ELEMENTS_BY_CATEGORY_TOOL_NAME, "description": GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM, "parameters": GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": SELECT_ELEMENTS_TOOL_NAME, "description": SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": SELECT_STORED_ELEMENTS_TOOL_NAME, "description": SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": LIST_STORED_ELEMENTS_TOOL_NAME, "description": LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": FILTER_ELEMENTS_TOOL_NAME, "description": FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "parameters": FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": GET_ELEMENT_PROPERTIES_TOOL_NAME, "description": GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM, "parameters": GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": UPDATE_ELEMENT_PARAMETERS_TOOL_NAME, "description": UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM, "parameters": UPDATE_ELEMENT_PARAMETERS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": PLACE_VIEW_ON_SHEET_TOOL_NAME, "description": PLACE_VIEW_ON_SHEET_TOOL_DESCRIPTION_FOR_LLM, "parameters": PLACE_VIEW_ON_SHEET_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": LIST_VIEWS_TOOL_NAME, "description": LIST_VIEWS_TOOL_DESCRIPTION_FOR_LLM, "parameters": LIST_VIEWS_TOOL_PARAMETERS_FOR_LLM}},
            {"type": "function", "function": {"name": PLANNER_TOOL_NAME, "description": PLANNER_TOOL_DESCRIPTION_FOR_LLM, "parameters": PLANNER_TOOL_PARAMETERS_FOR_LLM}},
        ],
        "anthropic": [
            {"name": REVIT_INFO_TOOL_NAME, "description": REVIT_INFO_TOOL_DESCRIPTION_FOR_LLM, "input_schema": {"type": "object", "properties": {}}},
            {"name": GET_ELEMENTS_BY_CATEGORY_TOOL_NAME, "description": GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM, "input_schema": GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM},
            {"name": SELECT_ELEMENTS_TOOL_NAME, "description": SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": SELECT_STORED_ELEMENTS_TOOL_NAME, "description": SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": LIST_STORED_ELEMENTS_TOOL_NAME, "description": LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": FILTER_ELEMENTS_TOOL_NAME, "description": FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM},
            {"name": GET_ELEMENT_PROPERTIES_TOOL_NAME, "description": GET_ELEMENT_PROPERTIES_TOOL_DESCRIPTION_FOR_LLM, "input_schema": GET_ELEMENT_PROPERTIES_TOOL_PARAMETERS_FOR_LLM},
            {"name": UPDATE_ELEMENT_PARAMETERS_TOOL_NAME, "description": UPDATE_ELEMENT_PARAMETERS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": UPDATE_ELEMENT_PARAMETERS_TOOL_PARAMETERS_FOR_LLM},
            {"name": PLACE_VIEW_ON_SHEET_TOOL_NAME, "description": PLACE_VIEW_ON_SHEET_TOOL_DESCRIPTION_FOR_LLM, "input_schema": PLACE_VIEW_ON_SHEET_TOOL_PARAMETERS_FOR_LLM},
            {"name": LIST_VIEWS_TOOL_NAME, "description": LIST_VIEWS_TOOL_DESCRIPTION_FOR_LLM, "input_schema": LIST_VIEWS_TOOL_PARAMETERS_FOR_LLM},
            {"name": PLANNER_TOOL_NAME, "description": PLANNER_TOOL_DESCRIPTION_FOR_LLM, "input_schema": PLANNER_TOOL_PARAMETERS_FOR_LLM},
        ],
        "google": [
            google_types.Tool(function_declarations=[
                google_types.FunctionDeclaration(name=REVIT_INFO_TOOL_NAME, description=REVIT_INFO_TOOL_DESCRIPTION_FOR_LLM, parameters={"type": "object", "properties": {}}),
                google_types.FunctionDeclaration(name=GET_ELEMENTS_BY_CATEGORY_TOOL_NAME, description=GET_ELEMENTS_BY_CATEGORY_TOOL_DESCRIPTION_FOR_LLM, parameters=GET_ELEMENTS_BY_CATEGORY_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=SELECT_ELEMENTS_TOOL_NAME, description=SELECT_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=SELECT_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=SELECT_STORED_ELEMENTS_TOOL_NAME, description=SELECT_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=SELECT_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=LIST_STORED_ELEMENTS_TOOL_NAME, description=LIST_STORED_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=LIST_STORED_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
                google_types.FunctionDeclaration(name=FILTER_ELEMENTS_TOOL_NAME, description=FILTER_ELEMENTS_TOOL_DESCRIPTION_FOR_LLM, parameters=FILTER_ELEMENTS_TOOL_PARAMETERS_FOR_LLM),
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
- get_elements_by_category: Get all elements by category (params: category_name)
- filter_elements: Advanced filtering (params: category_name, level_name, parameters)
- get_element_properties: Get parameter values (params: element_ids, parameter_names)
- update_element_parameters: Update parameters (params: updates OR element_ids + parameter_name + new_value)
- select_elements_by_id: Select specific elements (params: element_ids)
- select_stored_elements: Select stored elements (params: category_name)
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
    "params": {"updates": [{"element_id": "${step_1_element_ids}", "parameters": {"Sill Height": "2' 6\""}}]},
    "description": "Update sill height to 2'6\""
  }
]

WORKFLOW EXAMPLES:
- Parameter updates: filter_elements → update_element_parameters → select_stored_elements
- Property inspection: filter_elements → get_element_properties
- Element discovery: get_elements_by_category → get_element_properties → select_stored_elements

Use plan_and_execute_workflow for multi-step operations to provide complete results in one response."""
        }
        
        def execute_tool_call(tool_name, function_args):
            """Execute a tool safely and return its result dictionary."""
            normalized_args = function_args or {}
            app.logger.info(f"Executing tool '{tool_name}' with args: {normalized_args}")

            try:
                if tool_name == REVIT_INFO_TOOL_NAME:
                    tool_result_data = get_revit_project_info_mcp_tool()
                elif tool_name == GET_ELEMENTS_BY_CATEGORY_TOOL_NAME:
                    tool_result_data = get_elements_by_category_mcp_tool(category_name=normalized_args.get("category_name"))
                elif tool_name == SELECT_ELEMENTS_TOOL_NAME:
                    tool_result_data = select_elements_by_id_mcp_tool(element_ids=normalized_args.get("element_ids", []))
                elif tool_name == SELECT_STORED_ELEMENTS_TOOL_NAME:
                    tool_result_data = select_stored_elements_mcp_tool(category_name=normalized_args.get("category_name"))
                elif tool_name == LIST_STORED_ELEMENTS_TOOL_NAME:
                    tool_result_data = list_stored_elements_mcp_tool()
                elif tool_name == FILTER_ELEMENTS_TOOL_NAME:
                    tool_result_data = filter_elements_mcp_tool(
                        category_name=normalized_args.get("category_name"),
                        level_name=normalized_args.get("level_name"),
                        parameters=normalized_args.get("parameters", [])
                    )
                elif tool_name == GET_ELEMENT_PROPERTIES_TOOL_NAME:
                    tool_result_data = get_element_properties_mcp_tool(
                        element_ids=normalized_args.get("element_ids", []),
                        parameter_names=normalized_args.get("parameter_names", [])
                    )
                elif tool_name == UPDATE_ELEMENT_PARAMETERS_TOOL_NAME:
                    tool_result_data = update_element_parameters_mcp_tool(
                        updates=normalized_args.get("updates"),
                        element_ids=normalized_args.get("element_ids"),
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
