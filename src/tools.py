"""
Tool definitions for the agent using @tool decorator
Think of these as the agent's Swiss Army knife 🔧 - each tool has a specific purpose!
"""

from typing import Dict, Any, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field
import re
import json
from datetime import datetime


class ToolLogger:
    """Logs tool usage with automatic persistence"""

    def __init__(self, logs_dir: str = "./logs", session_id: str = None):
        self.logs = []
        self.logs_dir = logs_dir
        self.session_id = session_id

        # Make sure logs directory exists
        import os
        os.makedirs(logs_dir, exist_ok=True)

        # Create session-specific log file if session_id provided
        if session_id:
            self.log_file = os.path.join(logs_dir, f"session_{session_id}.json")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(logs_dir, f"tool_usage_{timestamp}.json")

    def log_tool_use(self, tool_name: str, input_data: Dict[str, Any], output: Any):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "input": input_data,
            "output": str(output),
        }
        self.logs.append(log_entry)

        # Automatically save to persistent file
        self._auto_save()
        return log_entry

    def _auto_save(self):
        """Automatically save logs to persistent file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to auto-save logs: {e}")

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs

    def save_logs(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)


def create_calculator_tool(logger: ToolLogger):
    """
    Creates a simple, safe calculator tool.
    Validates expression, evaluates it, logs usage, and returns result.
    """

    @tool("calculator")
    def calculator(expression: str) -> str:
        """
        Safely evaluate a basic mathematical expression.
        Allowed operators: +, -, *, /, %, **, //
        Uses AST validation before evaluation.
        """

        import ast, operator as op

        # Allowed operators for safety (no names, no calls)
        allowed_ops = (
            ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Mod, ast.Pow, ast.FloorDiv
        )
        allowed_unary = (ast.UAdd, ast.USub)

        def validate_ast(node):
            """Recursively ensure expression only contains safe arithmetic."""
            if isinstance(node, ast.Expression):
                return validate_ast(node.body)

            elif isinstance(node, ast.BinOp):
                if not isinstance(node.op, allowed_ops):
                    raise ValueError("Operator not allowed.")
                validate_ast(node.left)
                validate_ast(node.right)

            elif isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, allowed_unary):
                    raise ValueError("Unary operator not allowed.")
                validate_ast(node.operand)

            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)):
                    raise ValueError("Only numbers allowed.")
            else:
                raise ValueError(f"Illegal expression element: {type(node).__name__}")

        try:
            # Remove commas in numbers (e.g. 1,200 + 5)
            cleaned = expression.replace(",", "")

            # Parse and validate AST
            tree = ast.parse(cleaned, mode="eval")
            validate_ast(tree)

            # Evaluate safely
            result = eval(compile(tree, "<calc>", "eval"), {}, {})

            # Normalize integer-like floats
            if isinstance(result, float) and result.is_integer():
                result = int(result)

            # Log success
            logger.log_tool_use(
                "calculator",
                {"expression": expression},
                {"result": result}
            )

            return f"Result: {result}"

        except Exception as e:
            # Log failure
            logger.log_tool_use(
                "calculator",
                {"expression": expression},
                {"error": str(e)}
            )
            return f"Error: {e}"

    return calculator



def create_document_search_tool(retriever, logger: ToolLogger):
    """
    Creates a document search tool.
    """

    @tool
    def document_search(
            query: str,
            search_type: Literal["keyword", "type", "amount", "amount_range", "all"] = "keyword",
            doc_type: Optional[str] = None,
            min_amount: Optional[float] = None,
            max_amount: Optional[float] = None,
            comparison: Optional[Literal["over", "under", "between", "exact", "approximate"]] = None,
            amount: Optional[float] = None
    ) -> str:
        """
        Search for relevant documents using various criteria. Handles natural language amount queries.

        Args:
            query: Search query (e.g., "invoices over $50,000", "contracts", "insurance claims")
            search_type: Type of search - 'keyword', 'type', 'amount', 'amount_range' or 'all'
            doc_type: Document type filter (e.g., 'invoice', 'contract', 'claim')
            min_amount: Minimum amount (for range queries or "over" queries)
            max_amount: Maximum amount (for range queries or "under" queries)
            comparison: Type of amount comparison - 'over', 'under', 'between', 'exact', 'approximate'
            amount: Single amount value for comparisons (used with 'over', 'under', 'exact', 'approximate')

        Examples:
            - "Find documents over $50,000" → comparison='over', amount=50000
            - "Show invoices under $10,000" → search_type='type', doc_type='invoice', comparison='under', amount=10000
            - "Documents between $20,000 and $80,000" → min_amount=20000, max_amount=80000
            - "Contracts around $100,000" → comparison='approximate', amount=100000

        Returns:
            Formatted search results with document details
        """
        try:
            results = []

            # Handle different search types
            if search_type == "all":
                results = retriever.retrieve_all()

            if search_type == "keyword":
                results = retriever.retrieve_by_keyword(query)

            elif search_type == "type" and doc_type:
                results = retriever.retrieve_by_type(doc_type)
                # If amount criteria also specified, filter further
                if comparison or min_amount is not None or max_amount is not None:
                    amount_results = _handle_amount_search(
                        retriever, comparison, amount, min_amount, max_amount, query
                    )
                    # Intersect results
                    result_ids = {r.doc_id for r in amount_results}
                    results = [r for r in results if r.doc_id in result_ids]

            elif search_type == "amount" or search_type == "amount_range":
                results = _handle_amount_search(
                    retriever, comparison, amount, min_amount, max_amount, query
                )

            else:
                # Try to intelligently parse the query
                query_lower = query.lower()

                # Check if it's an amount query
                if any(word in query_lower for word in
                       ['over', 'under', 'above', 'below', 'between', 'around', 'exactly', '$']):
                    results = retriever._parse_and_retrieve_by_amount(query)
                # Check if it's a type query
                elif any(word in query_lower for word in ['invoice', 'contract', 'claim']):
                    for doc_type in ['invoice', 'contract', 'claim']:
                        if doc_type in query_lower:
                            results = retriever.retrieve_by_type(doc_type)
                            break
                else:
                    # Default to keyword search
                    results = retriever.retrieve_by_keyword(query)

            # Format results with amount information
            if not results:
                formatted = "No documents found matching your search criteria."
            else:
                formatted = f"Found {len(results)} document(s):\n\n"
                for i, chunk in enumerate(results, 1):
                    formatted += f"Document {i} (ID: {chunk.doc_id}):\n"
                    formatted += f"Title: {chunk.metadata.get('title', 'Unknown')}\n"
                    formatted += f"Type: {chunk.metadata.get('doc_type', 'Unknown')}\n"

                    # Include amount information if available
                    amount_value = None
                    for field in ['total', 'amount', 'value']:
                        if field in chunk.metadata:
                            amount_value = chunk.metadata[field]
                            formatted += f"Amount: ${amount_value:,.2f}\n"
                            break

                    if hasattr(chunk, 'relevance_score'):
                        formatted += f"Relevance Score: {chunk.relevance_score:.2f}\n"

                    formatted += f"Preview: {chunk.content[:200]}...\n"
                    formatted += "-" * 50 + "\n"

            # Log the tool use
            logger.log_tool_use(
                "document_search",
                {
                    "query": query,
                    "search_type": search_type,
                    "doc_type": doc_type,
                    "min_amount": min_amount,
                    "max_amount": max_amount,
                    "comparison": comparison,
                    "amount": amount
                },
                {"results_count": len(results)}
            )

            return formatted

        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            logger.log_tool_use(
                "document_search",
                {"query": query, "search_type": search_type},
                {"error": error_msg}
            )
            return error_msg

    def _handle_amount_search(retriever, comparison, amount, min_amount, max_amount, query):
        """Helper function to handle amount-based searches"""
        if comparison:
            if comparison == "over" and amount is not None:
                return retriever.retrieve_by_amount_range(min_amount=amount)
            elif comparison == "under" and amount is not None:
                return retriever.retrieve_by_amount_range(max_amount=amount)
            elif comparison == "exact" and amount is not None:
                return retriever.retrieve_by_exact_amount(amount)
            elif comparison == "approximate" and amount is not None:
                return retriever.retrieve_by_approximate_amount(amount)
            elif comparison == "between" and min_amount is not None and max_amount is not None:
                return retriever.retrieve_by_amount_range(min_amount=min_amount, max_amount=max_amount)

        # Handle direct min/max specifications
        if min_amount is not None or max_amount is not None:
            return retriever.retrieve_by_amount_range(min_amount=min_amount, max_amount=max_amount)

        # Try parsing from query
        return retriever._parse_and_retrieve_by_amount(query)

    # Store helper function as attribute
    document_search._handle_amount_search = _handle_amount_search

    return document_search


def create_document_reader_tool(retriever, logger: ToolLogger):
    """
    Creates a tool to read full document content.
    """

    @tool
    def document_reader(doc_id: str) -> str:
        """
        Read the full content of a specific document by its ID.

        Args:
            doc_id: The exact document ID to read (e.g., 'INV-001', 'CON-001')

        Returns:
            The full content of the document or an error message if not found
        """
        try:
            doc = retriever.get_document_by_id(doc_id)
            if doc:
                # Include amount information in the output
                amount_info = ""
                for field in ['total', 'amount', 'value']:
                    if field in doc.metadata:
                        amount_info = f"\nAmount: ${doc.metadata[field]:,.2f}"
                        break

                result = f"Document {doc_id}:{amount_info}\n\n{doc.content}"
                logger.log_tool_use(
                    "document_reader",
                    {"doc_id": doc_id},
                    {"found": True, "doc_type": doc.metadata.get('doc_type')}
                )
                return result
            else:
                logger.log_tool_use(
                    "document_reader",
                    {"doc_id": doc_id},
                    {"found": False}
                )
                return f"Document with ID {doc_id} not found."
        except Exception as e:
            error_msg = f"Error reading document: {str(e)}"
            logger.log_tool_use(
                "document_reader",
                {"doc_id": doc_id},
                {"error": error_msg}
            )
            return error_msg

    return document_reader


def create_document_statistics_tool(retriever, logger: ToolLogger):
    """
    Creates a tool to get statistics about the document collection.
    """

    @tool
    def document_statistics() -> str:
        """
        Get statistics about all documents in the system.

        Returns:
            Summary statistics including document counts, amount totals, and averages
        """
        try:
            stats = retriever.get_statistics()

            formatted = "DOCUMENT COLLECTION STATISTICS:\n\n"
            formatted += f"Total Documents: {stats['total_documents']}\n"
            formatted += f"Documents with Amounts: {stats['documents_with_amounts']}\n"
            formatted += f"\nDocument Types:\n"

            for doc_type, count in stats['document_types'].items():
                formatted += f"  - {doc_type.capitalize()}: {count}\n"

            if stats['documents_with_amounts'] > 0:
                formatted += f"\nFinancial Summary:\n"
                formatted += f"  - Total Amount: ${stats['total_amount']:,.2f}\n"
                formatted += f"  - Average Amount: ${stats['average_amount']:,.2f}\n"
                formatted += f"  - Minimum Amount: ${stats['min_amount']:,.2f}\n"
                formatted += f"  - Maximum Amount: ${stats['max_amount']:,.2f}\n"

            logger.log_tool_use(
                "document_statistics",
                {},
                {"stats": stats}
            )

            return formatted

        except Exception as e:
            error_msg = f"Error getting statistics: {str(e)}"
            logger.log_tool_use(
                "document_statistics",
                {},
                {"error": error_msg}
            )
            return error_msg

    return document_statistics


def get_all_tools(retriever, logger: ToolLogger) -> List:
    """
    Get all available tools for the agent.
    """
    return [
        create_calculator_tool(logger),
        create_document_search_tool(retriever, logger),
        create_document_reader_tool(retriever, logger),
        create_document_statistics_tool(retriever, logger)
    ]