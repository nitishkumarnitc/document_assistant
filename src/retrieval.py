import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from schemas import DocumentChunk


@dataclass
class Document:
    """Represents a document in our system"""
    doc_id: str
    title: str
    content: str
    doc_type: str  # 'invoice', 'contract', 'claim'
    metadata: Dict[str, Any]


class SimulatedRetriever:
    """
    Simulates document retrieval without using vector databases.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self._load_sample_documents()

    def _load_sample_documents(self):
        """Load sample documents into memory"""
        sample_docs = [
            Document(
                doc_id="INV-001",
                title="Invoice #12345",
                content="""
                Invoice #12345
                Date: 2024-01-15
                Client: Acme Corporation

                Services Rendered:
                - Consulting Services: $5,000
                - Software Development: $12,500
                - Support & Maintenance: $2,500

                Subtotal: $20,000
                Tax (10%): $2,000

                Payment Terms: Net 30 days
                """,
                doc_type="invoice",
                metadata={"client": "Acme Corporation", "date": "2024-01-15"}
            ),
            Document(
                doc_id="CON-001",
                title="Service Agreement",
                content="""
                SERVICE AGREEMENT

                This Service Agreement is entered into on January 1, 2024, between:
                - Provider: DocDacity Solutions Inc.
                - Client: Healthcare Partners LLC

                Services:
                1. Document Processing Platform Access
                2. 24/7 Technical Support
                3. Monthly Data Analytics Reports
                4. Compliance Monitoring

                Duration: 12 months
                Monthly Fee: $15,000
                Total Contract Value: $180,000

                Termination: Either party may terminate with 60 days written notice.
                """,
                doc_type="contract",
                metadata={"value": 180000, "duration_months": 12, "client": "Healthcare Partners LLC"}
            ),
            Document(
                doc_id="CLM-001",
                title="Insurance Claim #78901",
                content="""
                INSURANCE CLAIM FORM
                Claim Number: 78901
                Date of Incident: 2024-02-10
                Policy Number: POL-456789

                Claimant: John Doe
                Type of Claim: Medical Expense Reimbursement

                Expenses:
                - Hospital Visit: $1,200
                - Diagnostic Tests: $800
                - Medication: $150
                - Follow-up Consultation: $300

                Total Claim Amount: $2,450

                Status: Under Review
                """,
                doc_type="claim",
                metadata={"amount": 2450, "status": "Under Review", "claimant": "John Doe"}
            ),
            Document(
                doc_id="INV-002",
                title="Invoice #12346",
                content="""
                Invoice #12346
                Date: 2024-02-20
                Client: TechStart Inc.

                Products:
                - Enterprise License (Annual): $50,000
                - Implementation Services: $15,000
                - Training Package: $5,000

                Subtotal: $70,000
                Discount (10%): -$7,000
                Tax (10%): $6,300
                Total Due: $69,300

                Payment Terms: Net 45 days
                """,
                doc_type="invoice",
                metadata={"total": 69300, "client": "TechStart Inc.", "date": "2024-02-20"}
            ),
            Document(
                doc_id="INV-003",
                title="Invoice #12347",
                content="""
                Invoice #12347
                Date: 2024-03-01
                Client: Global Corp

                Services:
                - Annual Subscription: $120,000
                - Premium Support: $30,000
                - Custom Development: $45,000

                Subtotal: $195,000
                Tax (10%): $19,500
                Total Due: $214,500

                Payment Terms: Net 60 days
                """,
                doc_type="invoice",
                metadata={"total": 214500, "client": "Global Corp", "date": "2024-03-01"}
            )
        ]

        for doc in sample_docs:
            self.documents[doc.doc_id] = doc

    def add_document(self, document: Document):
        """Add a document to the retriever"""
        self.documents[document.doc_id] = document

    def _get_document_amount(self, doc: Document) -> Optional[float]:
        """
        Extract the amount from a document's metadata.
        Checks multiple possible fields for flexibility.
        """
        # Priority order for amount fields
        amount_fields = ['total', 'amount', 'value', 'total_amount', 'total_value']

        for field in amount_fields:
            if field in doc.metadata and doc.metadata[field] is not None:
                try:
                    return float(doc.metadata[field])
                except (ValueError, TypeError):
                    continue

        return None

    def retrieve_all(self) -> List[DocumentChunk]:
        """Retrieve all documents as DocumentChunks"""
        results = []
        for doc in self.documents.values():
            results.append(DocumentChunk(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata={
                    "title": doc.title,
                    "doc_type": doc.doc_type,
                    **doc.metadata
                },
                relevance_score=1.0
            ))
        return results

    def retrieve_by_keyword(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """
        Simple keyword-based retrieval
        """
        query_lower = query.lower()
        keywords = query_lower.split()

        results = []
        for doc in self.documents.values():
            content_lower = doc.content.lower()
            title_lower = doc.title.lower()

            # Calculate simple relevance score
            score = 0.0
            for keyword in keywords:
                # Check title (higher weight)
                if keyword in title_lower:
                    score += 2.0
                # Check content
                score += content_lower.count(keyword) * 0.5
                # Check metadata
                for value in doc.metadata.values():
                    if keyword in str(value).lower():
                        score += 1.0

            if score > 0:
                results.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata={
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        **doc.metadata
                    },
                    relevance_score=score
                ))

        # Sort by relevance and return top_k
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def retrieve_by_type(self, doc_type: str) -> List[DocumentChunk]:
        """Retrieve all documents of a specific type"""
        results = []
        for doc in self.documents.values():
            if doc.doc_type.lower() == doc_type.lower():
                results.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata={
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        **doc.metadata
                    },
                    relevance_score=1.0
                ))
        return results

    def retrieve_by_amount_range(
            self,
            min_amount: Optional[float] = None,
            max_amount: Optional[float] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve documents within a specific amount range.
        Now supports:
        - Both min and max: Documents between min and max
        - Only min: Documents >= min (e.g., "over $50,000")
        - Only max: Documents <= max (e.g., "under $10,000")
        - Neither: Returns all documents with amounts
        """
        if min_amount is None and max_amount is None:
            # If no bounds specified, return all documents with amounts
            return self._retrieve_all_with_amounts()

        results = []
        for doc in self.documents.values():
            amount = self._get_document_amount(doc)

            if amount is not None:
                # Check if amount matches criteria
                matches = True

                if min_amount is not None and amount < min_amount:
                    matches = False

                if max_amount is not None and amount > max_amount:
                    matches = False

                if matches:
                    results.append(DocumentChunk(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        metadata={
                            "title": doc.title,
                            "doc_type": doc.doc_type,
                            **doc.metadata
                        },
                        relevance_score=1.0
                    ))

        # Sort by amount for better organization
        results.sort(key=lambda x: self._get_document_amount_from_chunk(x), reverse=True)
        return results

    def retrieve_by_exact_amount(self, amount: float, tolerance: float = 0.01) -> List[DocumentChunk]:
        """
        Retrieve documents with an exact amount (with small tolerance for float comparison).
        """
        results = []
        for doc in self.documents.values():
            doc_amount = self._get_document_amount(doc)

            if doc_amount is not None and abs(doc_amount - amount) <= tolerance:
                results.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata={
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        **doc.metadata
                    },
                    relevance_score=1.0
                ))

        return results

    def retrieve_by_approximate_amount(
            self,
            amount: float,
            percentage: float = 10.0
    ) -> List[DocumentChunk]:
        """
        Retrieve documents with amounts approximately equal to the target.
        Default tolerance is ±10%.
        """
        tolerance = amount * (percentage / 100)
        min_amount = amount - tolerance
        max_amount = amount + tolerance

        results = []
        for doc in self.documents.values():
            doc_amount = self._get_document_amount(doc)

            if doc_amount is not None and min_amount <= doc_amount <= max_amount:
                # Calculate relevance based on how close the amount is
                distance = abs(doc_amount - amount)
                relevance = 1.0 - (distance / tolerance)  # Closer amounts get higher scores

                results.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata={
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        **doc.metadata
                    },
                    relevance_score=relevance
                ))

        # Sort by relevance (closest amounts first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def retrieve_by_amount(
            self,
            query: str,
            comparison_type: Optional[str] = None,
            amount: Optional[float] = None,
            min_amount: Optional[float] = None,
            max_amount: Optional[float] = None
    ) -> List[DocumentChunk]:
        """
        Flexible amount-based retrieval that understands natural language queries.

        Examples:
        - "over $50,000" → comparison_type="greater", amount=50000
        - "under $10,000" → comparison_type="less", amount=10000
        - "between $20,000 and $80,000" → min_amount=20000, max_amount=80000
        - "around $25,000" → comparison_type="approximate", amount=25000
        - "exactly $100,000" → comparison_type="exact", amount=100000
        """
        # If specific comparison type is provided, use it
        if comparison_type:
            if comparison_type in ["greater", "over", "above", "more than"]:
                return self.retrieve_by_amount_range(min_amount=amount)
            elif comparison_type in ["less", "under", "below", "less than"]:
                return self.retrieve_by_amount_range(max_amount=amount)
            elif comparison_type in ["exact", "exactly", "equal", "equals"]:
                return self.retrieve_by_exact_amount(amount)
            elif comparison_type in ["approximate", "around", "about", "roughly"]:
                return self.retrieve_by_approximate_amount(amount)
            elif comparison_type in ["between", "range"]:
                return self.retrieve_by_amount_range(min_amount=min_amount, max_amount=max_amount)

        # Otherwise, try to parse the query
        return self._parse_and_retrieve_by_amount(query)

    def _parse_and_retrieve_by_amount(self, query: str) -> List[DocumentChunk]:
        """
        Parse natural language amount queries and retrieve accordingly.
        """
        query_lower = query.lower()

        # Extract amounts from query
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amounts = [float(m.replace(',', '').replace('$', '')) for m in re.findall(amount_pattern, query)]

        # Check for comparison keywords
        if any(word in query_lower for word in ['over', 'above', 'more than', 'greater than', '>']):
            if amounts:
                return self.retrieve_by_amount_range(min_amount=amounts[0])

        elif any(word in query_lower for word in ['under', 'below', 'less than', '<']):
            if amounts:
                return self.retrieve_by_amount_range(max_amount=amounts[0])

        elif any(word in query_lower for word in ['between', 'range', 'from']):
            if len(amounts) >= 2:
                return self.retrieve_by_amount_range(
                    min_amount=min(amounts[0], amounts[1]),
                    max_amount=max(amounts[0], amounts[1])
                )

        elif any(word in query_lower for word in ['around', 'about', 'approximately', 'roughly', '~']):
            if amounts:
                return self.retrieve_by_approximate_amount(amounts[0])

        elif any(word in query_lower for word in ['exactly', 'exact', 'precisely', '=']):
            if amounts:
                return self.retrieve_by_exact_amount(amounts[0])

        # Default: if amounts mentioned, look for documents containing those amounts
        if amounts:
            return self.retrieve_by_amount_range(
                min_amount=min(amounts) * 0.9,
                max_amount=max(amounts) * 1.1
            )

        # Fallback to keyword search
        return self.retrieve_by_keyword(query)

    def _retrieve_all_with_amounts(self) -> List[DocumentChunk]:
        """Retrieve all documents that have amount information"""
        results = []
        for doc in self.documents.values():
            if self._get_document_amount(doc) is not None:
                results.append(DocumentChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata={
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        **doc.metadata
                    },
                    relevance_score=1.0
                ))
        return results

    def _get_document_amount_from_chunk(self, chunk: DocumentChunk) -> float:
        """Extract amount from a DocumentChunk for sorting"""
        amount_fields = ['total', 'amount', 'value', 'total_amount', 'total_value']

        for field in amount_fields:
            if field in chunk.metadata:
                try:
                    return float(chunk.metadata[field])
                except (ValueError, TypeError):
                    continue

        return 0.0

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific document by ID"""
        if doc_id in self.documents:
            doc = self.documents[doc_id]
            return DocumentChunk(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata={
                    "title": doc.title,
                    "doc_type": doc.doc_type,
                    **doc.metadata
                },
                relevance_score=1.0
            )
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        """
        total_docs = len(self.documents)
        docs_with_amounts = 0
        total_amount = 0.0
        amounts = []
        doc_types = {}

        for doc in self.documents.values():
            # Count by type
            doc_types[doc.doc_type] = doc_types.get(doc.doc_type, 0) + 1

            # Amount statistics
            amount = self._get_document_amount(doc)
            if amount is not None:
                docs_with_amounts += 1
                total_amount += amount
                amounts.append(amount)

        stats = {
            "total_documents": total_docs,
            "documents_with_amounts": docs_with_amounts,
            "total_amount": total_amount,
            "average_amount": total_amount / docs_with_amounts if docs_with_amounts > 0 else 0,
            "document_types": doc_types
        }

        if amounts:
            stats["min_amount"] = min(amounts)
            stats["max_amount"] = max(amounts)

        return stats
