import os
import sys
import json
import requests
from crewai import Agent, Task, Crew, LLM

# 1. Initialize Gemini Flash as the core reasoning engine
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Parse incoming payload data from n8n forwarded by the GitHub Action
try:
    # GitHub will pass a compressed JSON string containing both query and phone number
    input_data = json.loads(sys.argv[1])
    user_input_payload = input_data.get("user_query", "")
    user_phone = input_data.get("phone_number", "")
except Exception:
    user_input_payload = "Please convert the text into clean exam notes."
    user_phone = "default"

# 2. Define your Role-Based Cloud Agents
analyst_agent = Agent(
    role="Core Concept Analyst",
    goal="Extract every key equation, strict scientific term, and core concept from raw text.",
    backstory="You are an advanced text distillation tool. Your layout strips out conversational text and filters for absolute data.",
    llm=llm,
    verbose=False
)

exam_coach_agent = Agent(
    role="Exam Preparation Coach",
    goal="Structure raw technical data blocks into concise, high-yield structured study modules.",
    backstory="You are an expert physics and mathematics academic tutor. You structure arguments cleanly into notes optimized for standard student exam marking criteria.",
    llm=llm,
    verbose=False
)

# 3. Define the Agent Execution Pipeline Tasks
extract_task = Task(
    description=f"Thoroughly analyze this raw text and pull every single vital fact, formula, and definition: {user_input_payload}",
    expected_output="A clean bulleted list containing only technical scientific facts, concepts, and formulas.",
    agent=analyst_agent
)

format_task = Task(
    description="Take the extracted structured points and reformat them into a gorgeous, highly scannable Markdown study guide with logical section headers.",
    expected_output="A beautiful, comprehensive Markdown document optimized for studying.",
    agent=exam_coach_agent
)

# 4. Assemble the Autonomous Cloud Crew
crew = Crew(
    agents=[analyst_agent, exam_coach_agent],
    tasks=[extract_task, format_task]
)

# Run the crew and capture output text string
result = crew.kickoff()
output_text = str(result)

# 5. Send data back to the n8n Loop-Back Webhook
payload = {
    "status": "success",
    "phone_number": user_phone,
    "final_notes": output_text
}

n8n_webhook_url = "https://professionaltasker.app.n8n.cloud/webhook-test/receive-agent-output"

try:
    response = requests.post(n8n_webhook_url, json=payload, timeout=30)
    print(f"Callback successful. Status Code: {response.status_code}")
except Exception as e:
    print(f"Failed to deliver callback to n8n: {e}")
