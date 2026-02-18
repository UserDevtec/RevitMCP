# RevitMCP: Element-related HTTP routes
# -*- coding: UTF-8 -*-

from pyrevit import routes, script, DB


def _resolve_built_in_category(category_name, route_logger):
    if not category_name:
        return None

    if hasattr(DB.BuiltInCategory, category_name):
        return getattr(DB.BuiltInCategory, category_name)

    if not category_name.startswith("OST_"):
        possible_ost = "OST_" + category_name.replace(" ", "")
        if hasattr(DB.BuiltInCategory, possible_ost):
            route_logger.info("Interpreted category '{}' as '{}'.".format(category_name, possible_ost))
            return getattr(DB.BuiltInCategory, possible_ost)

    return None


def _find_parameter(element, param_name):
    # First try direct lookup by displayed parameter name.
    p = element.LookupParameter(param_name)
    if p:
        return p

    # Then try built-in enum name conversion.
    try:
        enum_name = param_name.replace(" ", "_").upper()
        if hasattr(DB.BuiltInParameter, enum_name):
            return element.get_Parameter(getattr(DB.BuiltInParameter, enum_name))
    except Exception:
        pass

    # Fallback to manual scan.
    for item in element.Parameters:
        if item and item.Definition and item.Definition.Name == param_name:
            return item

    return None


def _get_parameter_display_value(param, doc):
    if not param or not param.HasValue:
        return "Not available"

    try:
        if param.StorageType == DB.StorageType.String:
            return param.AsString() or ""

        if param.StorageType == DB.StorageType.Double:
            return param.AsValueString() or str(param.AsDouble())

        if param.StorageType == DB.StorageType.Integer:
            return str(param.AsInteger())

        if param.StorageType == DB.StorageType.ElementId:
            elem_id = param.AsElementId()
            if elem_id != DB.ElementId.InvalidElementId:
                ref_elem = doc.GetElement(elem_id)
                if ref_elem:
                    name = getattr(ref_elem, "Name", None)
                    return name if name else str(elem_id.IntegerValue)
                return str(elem_id.IntegerValue)
            return "None"

        return param.AsValueString() or ""
    except Exception:
        return "Not available"


def _parse_length_to_internal_feet(value):
    value_str = str(value).strip()

    if "'" in value_str or '"' in value_str:
        feet = 0.0
        inches = 0.0

        if "'" in value_str:
            feet_part = value_str.split("'")[0].strip()
            feet = float(feet_part) if feet_part else 0.0

        if '"' in value_str:
            inches_part = value_str
            if "'" in value_str:
                inches_part = value_str.split("'")[1]
            inches_part = inches_part.replace('"', '').strip()
            inches = float(inches_part) if inches_part else 0.0

        return feet + (inches / 12.0)

    return float(value_str)


def _set_parameter_value(param, new_value):
    if param.StorageType == DB.StorageType.String:
        param.Set(str(new_value))
        return True, None

    if param.StorageType == DB.StorageType.Double:
        try:
            param.Set(_parse_length_to_internal_feet(new_value))
            return True, None
        except Exception:
            return False, "Could not convert '{}' to number".format(new_value)

    if param.StorageType == DB.StorageType.Integer:
        try:
            param.Set(int(float(new_value)))
            return True, None
        except Exception:
            return False, "Could not convert '{}' to integer".format(new_value)

    return False, "Unsupported parameter type: {}".format(param.StorageType)


def register_routes(api):
    """Register all element-related routes with the API"""

    @api.route('/get_elements_by_category', methods=['POST'])
    def handle_get_elements_by_category(request):
        """
        Handles POST requests to /revit-mcp-v1/get_elements_by_category
        Returns elements in Revit by category name (without selecting them).
        Uses FilteredElementCollector with ToElementIds() for better performance.
        """
        route_logger = script.get_logger()

        if not request:
            route_logger.error("Critical error: 'request' object is None.")
            return routes.Response(status=500, data={"error": "Internal server error: 'request' object is None.", "details": "The 'request' object was not provided to the handler by pyRevit."})

        try:
            # Access the JSON payload from the request
            payload = request.data if hasattr(request, 'data') else None
            route_logger.info("Successfully accessed request.data. Type: {}, Value: {}".format(type(payload), payload))

            if not payload or not isinstance(payload, dict):
                route_logger.error("Request body (payload) is missing or not a valid JSON object.")
                return routes.Response(status=400, data={"error": "Request body is missing or not a valid JSON object."})

            category_name_payload = payload.get('category_name')

            if not category_name_payload:
                route_logger.error("Missing 'category_name' in JSON body for /get_elements_by_category.")
                return routes.Response(status=400, data={"error": "Missing 'category_name' in JSON request body."})

            # Get document from __revit__
            current_uiapp = __revit__
            if not hasattr(current_uiapp, 'ActiveUIDocument') or not current_uiapp.ActiveUIDocument:
                route_logger.error("Error in /get_elements_by_category: No active UI document.")
                return routes.Response(status=503, data={"error": "No active Revit UI document found."})
            doc = current_uiapp.ActiveUIDocument.Document

            # Category validation and element collection
            built_in_category = _resolve_built_in_category(category_name_payload, route_logger)
            if not built_in_category:
                route_logger.error("Invalid category_name: '{}'. Not a recognized BuiltInCategory.".format(category_name_payload))
                return routes.Response(status=400, data={"error": "Invalid category_name: '{}'.".format(category_name_payload)})

            # Use ToElementIds() for better performance - no need for ToElements()
            element_ids_collector = DB.FilteredElementCollector(doc)\
                                       .OfCategory(built_in_category)\
                                       .WhereElementIsNotElementType()\
                                       .ToElementIds()

            element_ids = []
            route_logger.info("Found {} elements in category '{}' using ToElementIds()".format(element_ids_collector.Count, category_name_payload))

            # Convert ElementId objects to string list
            for element_id in element_ids_collector:
                element_ids.append(str(element_id.IntegerValue))

            if not element_ids:
                route_logger.info("No elements found for category '{}' to return.".format(category_name_payload))
                return {"status": "success", "message": "No elements found for category '{}'".format(category_name_payload), "count": 0, "element_ids": []}

            route_logger.info("Successfully retrieved {} element IDs for category '{}'.".format(len(element_ids), category_name_payload))
            return {
                "status": "success",
                "message": "{} elements found for category '{}'.".format(len(element_ids), category_name_payload),
                "count": len(element_ids),
                "category": category_name_payload,
                "element_ids": element_ids
            }

        except Exception as e_main_logic:
            route_logger.critical("Critical error in /get_elements_by_category: {}".format(e_main_logic), exc_info=True)
            return routes.Response(status=500, data={"error": "Internal server error during main logic.", "details": str(e_main_logic)})

    @api.route('/elements/filter', methods=['POST'])
    def handle_filter_elements(request):
        route_logger = script.get_logger()

        try:
            payload = request.data if hasattr(request, 'data') else None
            if not payload or not isinstance(payload, dict):
                return routes.Response(status=400, data={"error": "Invalid JSON payload"})

            category_name = payload.get('category_name')
            if not category_name:
                return routes.Response(status=400, data={"error": "Missing 'category_name'"})

            level_name = payload.get('level_name')
            parameter_filters = payload.get('parameters', [])

            current_uiapp = __revit__
            if not hasattr(current_uiapp, 'ActiveUIDocument') or not current_uiapp.ActiveUIDocument:
                return routes.Response(status=503, data={"error": "No active UI document"})

            doc = current_uiapp.ActiveUIDocument.Document

            built_in_category = _resolve_built_in_category(category_name, route_logger)
            if not built_in_category:
                return routes.Response(status=400, data={"error": "Invalid category_name: '{}'.".format(category_name)})

            collector = DB.FilteredElementCollector(doc).OfCategory(built_in_category).WhereElementIsNotElementType()

            if level_name:
                levels = DB.FilteredElementCollector(doc).OfClass(DB.Level).ToElements()
                target_level = None
                for level in levels:
                    if level.Name == level_name or level.Name.lower() == str(level_name).lower():
                        target_level = level
                        break

                if not target_level:
                    level_names = [lvl.Name for lvl in levels]
                    return routes.Response(
                        status=400,
                        data={
                            "error": "Level '{}' not found".format(level_name),
                            "available_levels": level_names[:50]
                        }
                    )

                collector = collector.WherePasses(DB.ElementLevelFilter(target_level.Id))

            elements = collector.ToElements()
            route_logger.info("Found {} elements before parameter filtering".format(len(elements)))

            filtered_elements = []
            for element in elements:
                include_element = True

                for param_filter in parameter_filters:
                    param_name = param_filter.get('name')
                    expected_value = param_filter.get('value')
                    condition = (param_filter.get('operator') or param_filter.get('condition') or 'equals').lower()

                    if not param_name or expected_value is None:
                        continue

                    param = _find_parameter(element, param_name)
                    if not param:
                        include_element = False
                        break

                    current_value = _get_parameter_display_value(param, doc)
                    current_text = str(current_value or "")
                    expected_text = str(expected_value)

                    current_cmp = current_text.lower()
                    expected_cmp = expected_text.lower()

                    if condition in ("equals", "=="):
                        if current_cmp != expected_cmp:
                            include_element = False
                            break
                    elif condition in ("contains",):
                        if expected_cmp not in current_cmp:
                            include_element = False
                            break
                    elif condition in ("not_equals", "!=", "not equal"):
                        if current_cmp == expected_cmp:
                            include_element = False
                            break
                    elif condition in ("starts_with",):
                        if not current_cmp.startswith(expected_cmp):
                            include_element = False
                            break
                    elif condition in ("ends_with",):
                        if not current_cmp.endswith(expected_cmp):
                            include_element = False
                            break
                    elif condition in ("greater_than", ">"):
                        try:
                            if float(current_text.split()[0]) <= float(expected_text.split()[0]):
                                include_element = False
                                break
                        except Exception:
                            include_element = False
                            break
                    elif condition in ("less_than", "<"):
                        try:
                            if float(current_text.split()[0]) >= float(expected_text.split()[0]):
                                include_element = False
                                break
                        except Exception:
                            include_element = False
                            break
                    else:
                        include_element = False
                        break

                if include_element:
                    filtered_elements.append(element)

            element_ids = [str(elem.Id.IntegerValue) for elem in filtered_elements]

            return {
                "status": "success",
                "message": "Found {} elements matching filter criteria.".format(len(element_ids)),
                "count": len(element_ids),
                "category": category_name,
                "level": level_name if level_name else "Any",
                "parameter_filters": parameter_filters,
                "element_ids": element_ids
            }

        except Exception as e:
            route_logger.critical("Error in /elements/filter: {}".format(e), exc_info=True)
            return routes.Response(status=500, data={"error": "Internal server error", "details": str(e)})

    @api.route('/elements/get_properties', methods=['POST'])
    def handle_get_element_properties(request):
        route_logger = script.get_logger()

        try:
            payload = request.data if hasattr(request, 'data') else None
            if not payload or not isinstance(payload, dict):
                return routes.Response(status=400, data={"error": "Invalid JSON payload"})

            element_ids_list = payload.get('element_ids')
            if not element_ids_list:
                return routes.Response(status=400, data={"error": "Missing 'element_ids'"})

            parameter_names = payload.get('parameter_names', [])

            current_uiapp = __revit__
            if not hasattr(current_uiapp, 'ActiveUIDocument') or not current_uiapp.ActiveUIDocument:
                return routes.Response(status=503, data={"error": "No active UI document"})

            doc = current_uiapp.ActiveUIDocument.Document
            results = []

            for id_str in element_ids_list:
                try:
                    element = doc.GetElement(DB.ElementId(int(id_str)))
                    if not element:
                        results.append({"element_id": id_str, "error": "Element not found", "properties": {}})
                        continue

                    properties = {}

                    if not parameter_names:
                        category_name = element.Category.Name if element.Category else ""
                        common_params = ["Level", "Family and Type", "Comments"]
                        if category_name == "Windows":
                            common_params.extend(["Sill Height", "Head Height", "Width", "Height"])
                        elif category_name == "Doors":
                            common_params.extend(["Width", "Height", "Finish"])
                        elif category_name == "Walls":
                            common_params.extend(["Base Constraint", "Top Constraint", "Height"])
                        names_to_use = common_params
                    else:
                        names_to_use = parameter_names

                    for param_name in names_to_use:
                        param = _find_parameter(element, param_name)
                        properties[param_name] = _get_parameter_display_value(param, doc) if param else "Not available"

                    # Enrich with stable, human-readable identity values.
                    try:
                        if hasattr(element, "Symbol") and element.Symbol:
                            symbol = element.Symbol
                            family = symbol.Family
                            properties["Family Name"] = family.Name if family else "Not available"
                            properties["Type Name"] = symbol.Name
                            if family and symbol.Name:
                                properties["Family and Type"] = "{} : {}".format(family.Name, symbol.Name)
                    except Exception:
                        pass

                    properties["Element_Name"] = getattr(element, 'Name', 'No Name')
                    properties["Category"] = element.Category.Name if element.Category else "No Category"

                    results.append({"element_id": id_str, "properties": properties})

                except Exception as elem_error:
                    route_logger.warning("Error processing element {}: {}".format(id_str, elem_error))
                    results.append({"element_id": id_str, "error": str(elem_error), "properties": {}})

            return {
                "status": "success",
                "message": "Retrieved properties for {} elements".format(len(results)),
                "count": len(results),
                "elements": results
            }

        except Exception as e:
            route_logger.critical("Error in /elements/get_properties: {}".format(e), exc_info=True)
            return routes.Response(status=500, data={"error": "Internal server error", "details": str(e)})

    @api.route('/elements/update_parameters', methods=['POST'])
    def handle_update_element_parameters(request):
        route_logger = script.get_logger()

        try:
            payload = request.data if hasattr(request, 'data') else None
            if not payload or not isinstance(payload, dict):
                return routes.Response(status=400, data={"error": "Invalid JSON payload"})

            updates_list = payload.get('updates')
            if not updates_list:
                return routes.Response(status=400, data={"error": "Missing 'updates' list"})

            current_uiapp = __revit__
            if not hasattr(current_uiapp, 'ActiveUIDocument') or not current_uiapp.ActiveUIDocument:
                return routes.Response(status=503, data={"error": "No active UI document"})

            doc = current_uiapp.ActiveUIDocument.Document
            results = []

            transaction = DB.Transaction(doc, "Update Element Parameters")
            transaction.Start()
            try:
                for update_data in updates_list:
                    element_id_str = update_data.get('element_id')
                    parameters_to_update = update_data.get('parameters', {})

                    if not element_id_str or not parameters_to_update:
                        results.append({
                            "element_id": element_id_str,
                            "status": "error",
                            "message": "Missing element_id or parameters",
                            "updated_params": [],
                            "errors": {}
                        })
                        continue

                    try:
                        element = doc.GetElement(DB.ElementId(int(element_id_str)))
                        if not element:
                            results.append({
                                "element_id": element_id_str,
                                "status": "error",
                                "message": "Element not found",
                                "updated_params": [],
                                "errors": {}
                            })
                            continue

                        updated_params = []
                        errors = {}

                        for param_name, new_value in parameters_to_update.items():
                            param = _find_parameter(element, param_name)
                            if not param:
                                errors[param_name] = "Parameter not found"
                                continue

                            if param.IsReadOnly:
                                errors[param_name] = "Parameter is read-only"
                                continue

                            ok, err = _set_parameter_value(param, new_value)
                            if ok:
                                updated_params.append(param_name)
                            else:
                                errors[param_name] = err

                        if updated_params and not errors:
                            status = "success"
                            message = "All parameters updated successfully"
                        elif updated_params and errors:
                            status = "partial"
                            message = "Some parameters updated, some failed"
                        else:
                            status = "error"
                            message = "No parameters were updated"

                        results.append({
                            "element_id": element_id_str,
                            "status": status,
                            "message": message,
                            "updated_params": updated_params,
                            "errors": errors
                        })

                    except Exception as elem_error:
                        route_logger.warning("Error updating element {}: {}".format(element_id_str, elem_error))
                        results.append({
                            "element_id": element_id_str,
                            "status": "error",
                            "message": str(elem_error),
                            "updated_params": [],
                            "errors": {}
                        })

                transaction.Commit()
            except Exception as transaction_error:
                transaction.RollBack()
                return routes.Response(status=500, data={"error": "Transaction failed", "details": str(transaction_error)})

            success_count = len([r for r in results if r["status"] == "success"])
            partial_count = len([r for r in results if r["status"] == "partial"])
            error_count = len([r for r in results if r["status"] == "error"])

            return {
                "status": "success",
                "message": "Parameter update completed: {} success, {} partial, {} errors".format(success_count, partial_count, error_count),
                "summary": {
                    "total": len(results),
                    "success": success_count,
                    "partial": partial_count,
                    "errors": error_count
                },
                "results": results
            }

        except Exception as e:
            route_logger.critical("Error in /elements/update_parameters: {}".format(e), exc_info=True)
            return routes.Response(status=500, data={"error": "Internal server error", "details": str(e)})
