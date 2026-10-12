# DecodeLabs AI Internship - Project 1: Rule-Based AI Chatbot

# Knowledge Base (5+ Intents as Key-Value Pairs)
knowledge_base = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! Welcome to DecodeLabs AI Chatbot.",
    "how are you": "I am operating at 100% efficiency, thank you!",
    "what is your name": "I am DecodeBot, a deterministic rule-based AI system.",
    "who created you": "I was created as part of Project 1 at DecodeLabs!"
}

print("=======================================")
print("     WELCOME TO DECODEBOT (AI P1)      ")
print("  Type 'exit' at any time to end chat. ")
print("=======================================\n")

# 1. Continuous Input Loop
while True:
    # Phase 1: Input & Sanitization
    raw_input = input("You: ")
    clean_input = raw_input.lower().strip()
    
    # Phase 2: Exit Strategy (Kill Command)
    if clean_input == "exit":
        print("DecodeBot: Goodbye! Session closed.")
        break
        
    # Phase 3: Dictionary Lookup + Fallback Response
    response = knowledge_base.get(clean_input, "DecodeBot: I am sorry, I do not understand that command.")
    print(f"DecodeBot: {response}\n")