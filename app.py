import yaml
import gradio as gr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from agent.graph_agent import build_graph, SYSTEM_PROMPT

graph = build_graph()

# -------------------------------
# Interaction Logic
# -------------------------------
def interact_with_graph(user_input, history, thread_id="1"):
    config = {"configurable": {"thread_id": thread_id, "recursion_limit": 10}}

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))

    ai_responses = []
    for event in graph.stream({"messages": messages}, config):
        for value in event.values():
            for msg in value.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.content:
                        ai_responses.append(msg.content)
                    for tool_call in getattr(msg, 'tool_calls', []):
                        args_str = yaml.dump(tool_call["args"], sort_keys=False)
                        ai_responses.append(
                            f"**⚙️ Tool Used: {tool_call['name']}**\n\n```yaml\n{args_str}\n```"
                        )
                elif isinstance(msg, ToolMessage):
                    ai_responses.append(f"**📈 Result:**\n{msg.content}")

    response = "\n\n".join(ai_responses) if ai_responses else "🤔 I couldn’t generate a response."
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
    ]
    return response, new_history


# -------------------------------
# Modern UI
# -------------------------------
css = """
body {background: linear-gradient(120deg, #eef2f3, #8e9eab);}
.chatbox {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    padding: 20px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue"), title="💎 Data Analysis Chat") as demo:
    gr.HTML("<h2 style='text-align:center;'>💬 Talk to Your Data</h2>")
    chatbot = gr.Chatbot(
        label="Chat Assistant",
        height=550,
        type="messages",
        show_copy_button=True,
        avatar_images=("https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
                       "https://cdn-icons-png.flaticon.com/512/4712/4712034.png"),
    )
    with gr.Row():
        user_input = gr.Textbox(placeholder="Ask about your CSVs...", show_label=False, scale=4)
        send = gr.Button("Send 🚀", variant="primary", scale=1)

    clear_btn = gr.ClearButton(components=[user_input, chatbot], value="Clear 🧹")

    def handle_submit(message, chat_history):
        if not message.strip():
            return chat_history, ""
        response, new_history = interact_with_graph(message, chat_history)
        return new_history, ""

    send.click(handle_submit, [user_input, chatbot], [chatbot, user_input])
    user_input.submit(handle_submit, [user_input, chatbot], [chatbot, user_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
