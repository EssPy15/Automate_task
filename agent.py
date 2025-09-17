from llm_client import extract_event_info

def ai_agent(email_text: str):
    """Agent that processes incoming email and creates a structured plan."""
    structured_output = extract_event_info(email_text)
    print("Raw model output:", structured_output)

    # Normally, you’d parse and add to a calendar.
    # Here, we simulate:
    print("✅ Event added to planner!")

if __name__ == "__main__":
    email_text = "Midterm exam for CS201 will be held on Oct 3, 2025 at 10am."
    ai_agent(email_text)
