import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from print_color import print

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.assistant import DocumentAssistant


def print_header():
    """Print a nice header"""
    print("\n" + "=" * 60)
    print("DocDacity Intelligent Document Assistant", color='blue')
    print("=" * 60 + "\n")


def print_help():
    """Print help information"""
    print("\nAVAILABLE COMMANDS:", color='blue')
    print("  /help     - Show this help message")
    print("  /docs     - List available documents")
    print("  /quit     - Exit the assistant")
    print("\nExample queries:")
    print("  - What's the total amount in invoice INV-001?")
    print("  - Summarize all contracts")
    print("  - Calculate the sum of all invoice totals")
    print("  - Find documents with amounts over $50,000")
    print()


def list_documents(assistant: DocumentAssistant):
    """List all available documents"""
    print("\nAVAILABLE DOCUMENTS:", color='blue')
    print("-" * 40)

    for doc_id, doc in assistant.retriever.documents.items():
        print(f"ID: {doc_id}")
        print(f"Title: {doc.title}")
        print(f"Type: {doc.doc_type}")
        if 'total' in doc.metadata:
            print(f"Total: ${doc.metadata['total']:,.2f}")
        elif 'amount' in doc.metadata:
            print(f"Amount: ${doc.metadata['amount']:,.2f}")
        elif 'value' in doc.metadata:
            print(f"Value: ${doc.metadata['value']:,.2f}")
        print("-" * 40)


def main():
    """Main interactive loop"""
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    # Print header
    print_header()

    # Create assistant
    print(" INITIALIZING ASSISTANT...", color='green')
    assistant = DocumentAssistant(
        openai_api_key=api_key,
        model_name="gpt-4o",
        temperature=0.1
    )

    # Start session
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip() or "demo_user"
    session_id = assistant.start_session(user_id)
    print(f"Session started: {session_id}")

    # Show help
    print_help()

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nEnter Message: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!", color='blue')
                break
            elif user_input.lower() == "/help":
                print_help()
                continue
            elif user_input.lower() == "/docs":
                list_documents(assistant)
                continue

            # Process the message
            print("\nProcessing...", color='yellow')
            print("user_input:", user_input)
            result = assistant.process_message(user_input)

            if result["success"]:
                print("\n🤖 Assistant:", end=" ")

                if result.get("response"):
                    print(result["response"])
                if result.get("intent"):
                    intent = result["intent"]
                    print(f"\nINTENT: {intent['intent_type']}", color='green')
                if result.get("active_documents"):
                    print(f"\nSOURCES: {', '.join(result['active_documents'])}", color='blue')
                if result.get("tools_used"):
                    print(f"\nTOOLS USED: {', '.join(result['tools_used'])}", color='magenta')
                if result.get("summary"):
                    print(f"\nCONVERSATION SUMMARY: {result['summary']}", color='cyan')


            else:
                print(f"\nError: {result.get('error', 'Unknown error')}", color='red')

        except KeyboardInterrupt:
            print("\n\nGoodbye!", color='blue')
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}", color='red')


if __name__ == "__main__":
    main()
