import json


with open('cric.json', 'r') as file:
    data = json.load(file)


qa_dict = {item['question'].lower(): item['answer'] for item in data['questions']}

def chatbot_response(question):
    """Return the answer to the given question from the JSON data."""
    return qa_dict.get(question.lower(), "Sorry, I don't have an answer to that question.")


def main():
    print("Welcome to the Cricket Chatbot! Ask me anything about cricket.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")

if _name_ == "_main_":
    main()