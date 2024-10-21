response_dictionary = {
    "hi": "Hello there! How can I assist you today?",
    "how are you": "I'm a bot, so I don't have feelings, but I'm functioning properly!",
    "bye": "Goodbye! Have a great day!",
    "help": "Sure, I can help you. What do you need assistance with?",

    # Product Information
    "Tell me about product X": "Product X is a high-performance gadget with features like A, B, and C. It is designed for durability and offers a 1-year warranty.",
    "Do you have smartwatches?": "Yes, we offer a variety of smartwatches from top brands such as Apple, Samsung, and Fitbit. You can browse the selection on our website.",

    # Shipping Details
    "How long does shipping take?": "Shipping typically takes between 3-7 business days, depending on your location and the shipping method chosen.",
    "What shipping methods are available?": "We offer standard, expedited, and overnight shipping. You can select your preferred method during checkout.",

    # Return Policy
    "What is your return policy?": "We have a 30-day return policy. If you are not satisfied with your purchase, you can return the item within 30 days for a full refund.",
    "How do I return a product?": "To return a product, please visit our returns page, enter your order number, and follow the instructions to generate a return shipping label.",

    # Technical Support
    "My gadget won’t turn on": "Please try holding the power button for 10 seconds to perform a hard reset. If the issue persists, contact our technical support team.",
    "How do I reset my device?": "To reset your device, navigate to the settings menu, select 'Reset', and follow the on-screen instructions. Ensure you back up your data before resetting."
}

def chatbot_response(user_input):
    user_input = user_input.lower()
    if user_input in response_dictionary:
        return response_dictionary[user_input]
    else:
        return "I'm sorry, I don't understand that. Please try asking a different question or request."
    

def main():
    print("Welcome to the Chatbot! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
