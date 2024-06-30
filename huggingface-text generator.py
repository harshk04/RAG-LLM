from gradio_client import Client

def get_prompts(prompt):
    client = Client("harshk04/TextGeneration")
    result = client.predict(
            message=f"I want you to convert this text in a description of sign language. Give me instructions for how I can say the sentence '{prompt}' in sign language. Please return just the instructions and no additional text.",
            system_message="You are a friendly chatbot",
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            api_name="/chat"
    )

    list_prompts = result.split("\n")
    list_prompts = [prompt[3:] for prompt in list_prompts if prompt and prompt[0].isdigit()]

    return list_prompts

if __name__ == '__main__':
    prompt = 'Good Morning'
    print(get_prompts(prompt))
