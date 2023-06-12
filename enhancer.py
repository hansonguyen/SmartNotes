import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

"""
Returns AI response with summary, additional info, and more resources
"""
def enhance(filename: str, notes: str, ctx: str) -> str:
    # System prompt to initialize ChatGPT
    system_prompt = "Act as a note enhancer. You will receive the output of an OCR reading of handwritten school notes. I will provide a related topic list in the form 'Context: [topic]'. Your task is to generate a concise and easy-to-read summary of the notes, along with explanations so that someone without knowledge could understand. Include a 'More Resources' section that provides additional learning materials. Structure your output with the headings 'Summary', 'Explanations', and 'More Resources'."
    print("Enhancing notes...")

    # Perform chat completion
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": f"Context: [{ctx}] Input: [{notes}]" }
            ]
        )
        print("Writing file...")
        output_file = "./outputs/ai_response.json"
        with open(output_file, 'w') as json_file:
            json.dump(res, json_file, indent=2)

        message = res["choices"][0]["message"]["content"]

        # Write response to file
        with open(f'./enhanced_notes/{filename.split(".")[0]}_enhanced.txt', 'w') as f:
            f.write(message)
        print(f'Notes enhanced in {filename.split(".")[0]}_enhanced.txt')
        return message

    except Exception as e:
        return e