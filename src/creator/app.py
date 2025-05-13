"""To run:
python -m src.creator.app
Open: http://127.0.0.1:7860
To run auto reload:
gradio app.py
Need to import modules with absolute path
and cd into src/creator directory 
"""
import sys
import os
from pathlib import Path
import gradio as gr
import asyncio
import shutil
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.backend.utils.logging import setup_logging
from src.creator.chatbot_creator import ChatbotCreator
from src.creator.utils.utils import state as app_state
from src.creator.utils.utils import load_config
from src.creator.utils.css import css

from src.creator.services.config_service import ConfigService
from src.creator.services.doc_service import DocumentService
from src.creator.services.prompt_creation_service import PromptCreationService
from src.creator.services.prompt_optimization_service import PromptOptimizationService
from src.creator.services.simulation_service import SimulationService


cfg = load_config()
logger = logging.getLogger(__name__)
setup_logging()


config_service = ConfigService(cfg)
simulation_service = SimulationService(config_service)
doc_service = DocumentService(config_service, simulation_service)
prompt_creation_service = PromptCreationService(config_service)
prompt_optimization_service = PromptOptimizationService(
    config_service,
    prompt_creation_service
)
simulation_service = SimulationService(config_service)

# creator = ChatbotCreator(cfg)
# Initialize state
if "mode" not in app_state:
    app_state["mode"] = cfg.creator.mode

# Initialize "regeneration_count" if it doesn't exist
if "regeneration_count" not in app_state:
    app_state["regeneration_count"] = 0


# Utility wrapper for async
def run_async(coro):
    return asyncio.run(coro)


def toggle_mode():
    """Toggle between create and optimize modes."""
    current_mode = cfg.creator.mode
    new_mode = "optimize" if current_mode == "create" else "create"
    
    # Update both app_state and config
    app_state["mode"] = new_mode
    cfg.creator.mode = new_mode
    
    logger.info(f"🔄 Switched to {new_mode} mode")
    # Return updated UI elements - fixed to avoid "label" attribute
    next_text = "▶️ Next: Review Prompts" if new_mode == "optimize" else "▶️ Next: Create Prompts"
    # Return updated UI elements
    return (
        f"Current Mode: {new_mode.upper()}",
        gr.update(visible=new_mode == "optimize"),
        next_text
    )


# ---------Screen 1: File Upload & Model Selection------------
def upload_file(file):
    if file is None:
        return "⚠️ No file selected."
    filename = os.path.basename(file.name)
    target = Path(cfg.creator.data_ingest_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file.name, target)
    return f"✅ File '{filename}' saved to {target}"


def set_model(resoning_model, response_model) -> str:
    """Update model across all services"""
    config_service.update_model(resoning_model, response_model)
    app_state["reasoning_model"] = resoning_model
    app_state["response_model"] = response_model
    logger.info(f"Updated models to Reasoning: {resoning_model} "
                f"and Response: {response_model}")
    return f"Models set to: {resoning_model} and {response_model}"


def summarize_and_proceed(file):
    if file is None:
        return (
            "⚠️ Please upload a file first.",
            gr.update(visible=False),
            gr.update(visible=True)
        )
    try:
        summary = run_async(doc_service.summarize_document())
        logger.info(f"Document summarized successfully: {summary[: 200]}")
        app_state["summary"] = summary
        # In optimize mode, check if we need to load prompts
        if cfg.creator.mode == "optimize":
            logger.info("Optimize mode detected, prompts will be loaded in the next step")
        return (
            "✅ Document summarized successfully. Ready to proceed.",
            gr.update(visible=True)
        )
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return (
            f"❌ Error summarizing document: {str(e)}",
            gr.update(visible=False)
        )


def load_existing_prompts():
    try:
        # Check if we have a document summary
        if not app_state.get("summary"):
            return (
                "⚠️ Please upload and summarize a document first.",
                gr.update(visible=False)
            )
        
        prompts = config_service.get_current_prompts()
        app_state["sim_prompt"] = prompts["simulator"]
        app_state["chat_prompt"] = prompts["chatbot"]
        app_state["reasoning_prompt"] = prompts["reasoning"]
        return (
            "✅ Loaded existing prompts. Ready to proceed.",
            gr.update(visible=True)
        )
    except Exception as e:
        logger.error(f"Error loading existing prompts: {e}")
        return (
            f"❌ Error loading existing prompts: {str(e)}",
            gr.update(visible=False)
        )


# Screen 2: -------Chat Interface for Prompt Creation-------

# Function to handle chat submissions
def handle_chat_message(message, history):
    """Process a chat message and update prompts if complete."""
    if not message:
        return "", history, "", "", gr.update(visible=False), ""
    history_for_display = history.copy() if history else []
    try:
        assistant_response, updated_history, is_complete, sim_prompt, chat_prompt = run_async(
            prompt_creation_service.create_prompt(
                message=message,
                input_doc_summary=app_state.get("summary", ""),
                history=[(h[0], h[1]) for h in history] if history else []
            )
        )
        logger.info(f"Assistant response: {assistant_response}")
        logger.info(f"Updated history: {updated_history}")
        logger.info(f"Is complete: {is_complete}")
        logger.info(f"Simulator prompt: {sim_prompt}")
        logger.info(f"Chatbot prompt: {chat_prompt}")

        history_for_display = []
        history_for_display = history.copy() if history else []
        
        # Add new exchange
        history_for_display.append({"role": "user", "content": message})
        history_for_display.append({"role": "assistant", "content": assistant_response})
        
        logger.info(f"history for display: {history_for_display}")
        
        if is_complete:
            app_state["prompt_creation_complete"] = True
            app_state["sim_prompt"] = sim_prompt
            app_state["chat_prompt"] = chat_prompt
            
            reasoning_prompt = run_async(
                prompt_creation_service.create_reasoning_prompt(
                    chatbot_prompt=chat_prompt,
                    input_doc_summary=app_state.get("summary", "")
                )
            )
            app_state["reasoning_prompt"] = reasoning_prompt

            return (
                "",
                history_for_display,
                sim_prompt,
                chat_prompt,
                gr.update(visible=True),
                "✅ Prompts created successfully!"
            )
        # If not complete, just update chat
        return (
            "",  # clear input box
            history_for_display,  # history_for_display,
            "",  # No simulator prompt to show
            "",  # No chatbot prompt to show
            gr.update(visible=True),
            "Test Response"  # No status message
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return "", history, "", "", gr.update(visible=False), f"❌ Error: {str(e)}"


def check_prompt_completion():
    """Check if prompts are complete and update UI accordingly."""
    is_complete = app_state.get("prompt_creation_complete", False)
    
    if is_complete:
        sim_prompt = app_state.get("sim_prompt", "")
        chat_prompt = app_state.get("chat_prompt", "")
        
        return (
            sim_prompt,
            chat_prompt,
            "✅ Prompts created successfully!",
            gr.update(visible=True)  # Show next button
        )
    else:
        return (
            sim_prompt_box.value,  # Maintain current values
            chat_prompt_box.value,
            "",
            gr.update(visible=False)
        )
    

def generate_simulations():
    try:
        # Clear previous conversations
        app_state["conversations"] = []
        app_state["current_conv_index"] = 0
        app_state["feedback_history"] = []

        if not app_state.get("sim_prompt") or not app_state.get("chat_prompt"):
            return (
                "⚠️ No prompts available. Please create prompts first.",
                gr.update(visible=False)
            )
        # Get current prompts
        current_prompts = {
            "simulator": app_state["sim_prompt"],
            "chatbot": app_state["chat_prompt"],
            "reasoning": app_state.get("reasoning_prompt", "")
        }
        
        # Generate conversations using the simulator
        conversations = run_async(
            simulation_service.generate_simulations(
                prompts=current_prompts,
                num_simulations=10  # or get from config
            )
        )
        app_state["conversations"] = conversations

        if not app_state["conversations"]:
            return (
                "⚠️ No conversations were generated.",
                gr.update(visible=False)
            )
        return (
            f"✅ Generated {len(conversations)} conversation sets. Ready for feedback!",
            gr.update(visible=True)
        )
    except Exception as e:
        logger.error(f"Error generating simulations: {e}")
        return (
            f"❌ Error generating simulations: {str(e)}",
            gr.update(visible=False)
        )


def prepare_conversation_for_display(index=None):
    if index is not None:
        app_state["current_conv_index"] = index
    
    if not app_state["conversations"] or app_state["current_conv_index"] >= len(app_state["conversations"]):
        return (
            "No conversation to display",
            "",
            "",
            "0/0",
            gr.update(visible=False)
        )
    
    current_conv = app_state["conversations"][app_state["current_conv_index"]]
    
    # Format the conversation for display
    chat_html = ""
    for turn in current_conv:
        chat_html += "<div style='margin-bottom: 15px;'>"
        chat_html += f"<p><strong>User:</strong> {turn['customer_inquiry']}</p>"
        chat_html += f"<p><strong>Bot:</strong> {turn['bot_response']}</p>"
        chat_html += "</div>"
    
    conversation_id = f"Conversation {app_state['current_conv_index'] + 1}/10"
    
    # Determine if we should show next button
    show_next = app_state["current_conv_index"] < len(app_state["conversations"]) - 1
    
    return (
        chat_html,
        "",
        "",
        conversation_id,
        gr.update(visible=show_next)
    )


def submit_feedback(feedback_text):
    if not feedback_text.strip():
        return (
            "⚠️ Please provide feedback before proceeding.",
            gr.update(visible=True)
        )
    current_idx = app_state["current_conv_index"]
    current_conv = app_state["conversations"][current_idx]
    
    # Save feedback
    app_state["feedback_history"].append({
        "conversation_index": current_idx,
        "conversation": current_conv,
        "feedback": feedback_text
    })
    
    # Move to next conversation
    app_state["current_conv_index"] += 1
    
    # Check if we've gone through all conversations
    if app_state["current_conv_index"] >= len(app_state["conversations"]):
        return (
            "✅ All feedback collected! Ready for final decision.",
            gr.update(visible=False, _js="() => { document.getElementById('check_transition').click(); return true; }")
        )
    else:
        return (
            "✅ Feedback submitted! Moving to next conversation.",
            gr.update(visible=True)
        )


def process_all_feedback():
    if len(app_state["feedback_history"]) < len(app_state["conversations"]):
        return (
            "⚠️ Please provide feedback for all conversations first.",
            None,
            None
        )
    # Format feedback for optimizer
    feedback_list = []
    for fb in app_state["feedback_history"]:
        for turn in fb["conversation"]:
            feedback_list.append({
                "customer_inquiry": turn["customer_inquiry"],
                "bot_response": turn["bot_response"],
                "feedback": fb["feedback"]
            })
    
    app_state["regeneration_count"] += 1
    
    return (
        f"✅ All feedback processed! This is regeneration #{app_state['regeneration_count']}.\n"
        "Would you like to regenerate the prompts or deploy the current version?",
        gr.update(visible=True),
        gr.update(visible=True)
    )


def regenerate_prompts():
    try:
        # Check if we have collected feedback for all conversations
        if len(app_state.get("feedback_history", [])) < len(app_state.get("conversations", [])):
            return (
                "⚠️ Please provide feedback for all conversations first.",
                1
            )
        # Get current prompts
        current_prompts = {
            "simulator": app_state.get("sim_prompt", ""),
            "chatbot": app_state.get("chat_prompt", ""),
            "reasoning": app_state.get("reasoning_prompt", "")
        }
        
        # Use the feedback to optimize prompts
        chat_prompt, sim_prompt, reasoning_prompt, _ = run_async(
            prompt_optimization_service.optimize_prompts(
                input_doc_summary=app_state.get("summary", ""),
                feedback_history=app_state.get("feedback_history", []),
                current_prompts=current_prompts
            )
        )
        
        # Update app_state with new prompts
        app_state["chat_prompt"] = chat_prompt
        app_state["sim_prompt"] = sim_prompt
        app_state["reasoning_prompt"] = reasoning_prompt
        
        # Reset for new feedback round
        app_state["conversations"] = []
        app_state["current_conv_index"] = 0
        app_state["feedback_history"] = []
        
        # Generate new simulations with the optimized prompts
        new_prompts = {
            "simulator": sim_prompt,
            "chatbot": chat_prompt,
            "reasoning": reasoning_prompt
        }
        conversations = run_async(
            simulation_service.generate_simulations(
                prompts=new_prompts,
                num_simulations=10  # or get from config
            )
        )
        app_state["conversations"] = conversations
        app_state["current_conv_index"] = 0  # Reset index
        if not app_state["conversations"]:
            return (
                "⚠️ No conversations were generated during optimization.",
                1
            )
        return (
            "🔄 Prompts regenerated and new conversations created. Ready for feedback!",
            3
        )
    except Exception as e:
        logger.error(f"Error regenerating prompts: {e}")
        return f"❌ Error regenerating prompts: {str(e)}", 1
    

def deploy_prompts():
    try:
        # Save final prompts
        config_service.update_prompts(
            simulator_prompt=app_state["sim_prompt"],
            chatbot_prompt=app_state["chat_prompt"],
            reasoning_prompt=app_state["reasoning_prompt"]
        )
        return "🚀 Deployment in progress! Your chatbot prompts are being deployed.", 5
    except Exception as e:
        logger.error(f"Error deploying prompts: {e}")
        return f"❌ Error deploying prompts: {str(e)}", 1


def handle_screen_transition_after_regenerate(new_screen):
    if new_screen == 3:
        return gr.update(visible=False), gr.update(visible=True)
    return None, None


def handle_conversation_display_after_regenerate():
    return prepare_conversation_for_display(0)


# ---------Gradio App Setup------------
combined_css = css + """
/* Additional custom CSS */
.my-new-class {
    font-size: 18px;
}

/* Styled dividers */
hr {
  margin: 24px 0;
  border: 0;
  height: 1px;
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0));
}
"""

# Main Gradio App
with gr.Blocks(
    title="🤖 Interactive Prompt Creator",
    css=combined_css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    )
) as demo:
    # Application state (hidden)
    current_screen = gr.State(value=1)
    # Screen 1: Initial Setup with Mode Toggle
    with gr.Group(visible=True, elem_classes=["section"]) as screen1:
        gr.Markdown("# 📁 Step 1: Initial Setup")
        
        with gr.Row(elem_classes=["gradio-row"]):
            mode_toggle = gr.Button("🔄 Toggle Mode")
            mode_display = gr.Textbox(
                f"Current Mode: {cfg.creator.mode.upper()}",
                label="Mode",
                interactive=False,
                elem_classes=["status-box"]
            )
        gr.Markdown("---")

        # Document upload and summarization (needed for both modes)
        with gr.Group(elem_classes=["section"]) as doc_upload_group:
            gr.Markdown("### 📂 Upload Your Document")
            with gr.Row(elem_classes=["gradio-row"]):
                file_upload = gr.File(label="Upload Input Documents (txt, pdf, etc.)")
            
            upload_msg = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes=["status-box"]
            )
            with gr.Group(elem_classes=["section"]):
                gr.Markdown("### 🤖 Select Models")
                with gr.Row(equal_height=True, elem_classes=["gradio-row"]):
                    reasoning_model_dropdown = gr.Dropdown(
                        choices=["gpt-4o-mini", "gpt-4o"],
                        value=app_state.get("reasoning_model", "gpt-4o-mini"),
                        label="Reasoning Agent Model"
                    )
                    response_model_dropdown = gr.Dropdown(
                        choices=["gpt-4o-mini", "gpt-4o"],
                        value=app_state.get("response_model", "gpt-4o-mini"),
                        label="Response Agent Model"
                    )
                
                model_status = gr.Textbox(
                    label="Model Status",
                    interactive=False,
                    elem_classes=["status-box"]
                )
            with gr.Group(elem_classes=["section"]):
                gr.Markdown("### 📊 Process Documents")
                with gr.Row(elem_classes=["gradio-row"]):
                    summarize_btn = gr.Button(
                        "📝 Summarize & Embed Documents",
                        elem_classes=["primary"]
                    )
                    summary_status = gr.Textbox(
                        label="Summary Status",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
            
        # Optimize mode load prompts button
        with gr.Group(visible=cfg.creator.mode == "optimize", elem_classes=["section"]) as optimize_group:
            gr.Markdown("### 📝 Load Existing Prompts")
            with gr.Row(elem_classes=["gradio-row"]):
                load_prompts_btn = gr.Button(
                    "🔄 Load Existing Prompts", 
                    elem_classes=["primary"]
                )
                load_status = gr.Textbox(
                    label="Load Status",
                    interactive=False,
                    elem_classes=["status-box"]
                )
        gr.Markdown("---")

        with gr.Row(elem_classes=["gradio-row"]):
            next_btn_text = f"▶️ Next: {'Create Prompts' if cfg.creator.mode == 'create' else 'Review Prompts'}"
            next_to_screen2 = gr.Button(
                next_btn_text,
                visible=False,
                elem_classes=["success"]
            )

    # Screen 2: Chat Interface for Prompt Creation
    with gr.Group() as screen2:
        gr.Markdown("## 💬 Conversation")
        chat_history = gr.Chatbot(
            label="Conversation with Meta Prompt Creator",
            type='messages'
        )
        
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Talk to the prompt creator...",
                label="Your message",
                show_label=False,
                lines=2
            )
            chat_submit_btn = gr.Button(
                "Send",
                # elem_classes=["primary"]
            )
            
        with gr.Row():
            prompt_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")

        # Prompt display and editing
        with gr.Row():
            sim_prompt_box = gr.Textbox(label="Simulator Prompt", lines=8)
            chat_prompt_box = gr.Textbox(label="Chatbot Response Prompt", lines=8)
        
        with gr.Row():
            generate_sims_btn = gr.Button(
                "🤖 Generate Simulated Conversations",
                elem_classes=["primary"]
            )
            sim_status = gr.Textbox(label="Simulation Status", interactive=False)
        
        with gr.Row():
            next_to_screen3 = gr.Button("▶️ Next: Review Conversations", visible=False)
    
    # Screen 3: Conversation Review & Feedback
    with gr.Group(visible=False) as screen3:
        gr.Markdown("# 🔍 Step 3: Review Conversations & Provide Feedback")
        
        with gr.Row():
            conversation_id = gr.Textbox(label="", interactive=False)
        
        with gr.Row():
            conversation_display = gr.HTML(
                "",
                label="Conversation",
                elem_classes=["conversation-container"]
            )
        
        with gr.Row():
            feedback_input = gr.Textbox(
                label="Your Feedback", 
                placeholder="Provide detailed feedback on this conversation...",
                lines=4
            )
        
        with gr.Row():
            submit_feedback_btn = gr.Button(
                "📤 Submit Feedback",
                elem_classes=["primary"]
            )
            feedback_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            next_conversation_btn = gr.Button("▶️ Next Conversation", visible=True)
    
    # Screen 4: Final Decision
    with gr.Group(visible=False) as screen4:
        gr.Markdown("# 🏁 Step 4: Final Decision")
        
        with gr.Row():
            decision_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            regenerate_btn = gr.Button(
                "🔄 Regenerate Prompts",
                elem_classes=["primary"]
            )
            deploy_btn = gr.Button(
                "🚀 Deploy Prompts", 
                elem_classes=["success"]
            )
    
    # Screen 5: Deployment
    with gr.Group(visible=False) as screen5:
        gr.Markdown("# 🚀 Deployment")
        
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; padding: 50px;">
                    <h1 style="font-size: 72px;">🚀</h1>
                    <h2>Deploying Your Chatbot</h2>
                    <p>Your optimized prompts are being deployed. This may take a few moments.</p>
                </div>
            """)
        
        with gr.Row():
            deployment_status = gr.Textbox(label="Deployment Status", interactive=False)
    
    # --- Event Handlers ---
    # Mode toggle handler
    mode_toggle.click(
        toggle_mode,
        outputs=[mode_display, optimize_group, next_to_screen2]
    )
    
    # Screen 1 Events: File upload and model selection
    file_upload.change(upload_file, inputs=file_upload, outputs=upload_msg)
    reasoning_model_dropdown.change(
        set_model,
        inputs=[reasoning_model_dropdown, response_model_dropdown], 
        outputs=[model_status]
    )
    response_model_dropdown.change(
        set_model,
        inputs=[reasoning_model_dropdown, response_model_dropdown],
        outputs=[model_status]
    )
    summarize_btn.click(
        summarize_and_proceed,
        inputs=[file_upload],
        outputs=[summary_status, next_to_screen2]
    )

    # Load existing prompts button (only shown in optimize mode)
    load_prompts_btn.click(
        load_existing_prompts,
        outputs=[load_status, next_to_screen2]
    )
    next_to_screen2.click(
        lambda: 2,
        outputs=[current_screen]
    )
    
    # Screen 2 Events: Create prompts, then generate simulations
    chat_submit_btn.click(
        handle_chat_message,
        inputs=[user_input, chat_history],
        outputs=[user_input, chat_history, sim_prompt_box, chat_prompt_box,
                 next_to_screen3, prompt_status]
    )
    generate_sims_btn.click(
        generate_simulations,
        outputs=[sim_status, next_to_screen3]
    )
    next_to_screen3.click(
        lambda: 3,
        outputs=[current_screen]
    )
    
    # Screen 3 Events: Go through conversations and submit feedback
    submit_feedback_btn.click(
        submit_feedback,
        inputs=[feedback_input],
        outputs=[feedback_status, next_conversation_btn]
    )
    next_conversation_btn.click(
        lambda: prepare_conversation_for_display(),
        outputs=[conversation_display, feedback_input, feedback_status, conversation_id, next_conversation_btn]
    )
    # Add a separate check button (hidden) that's triggered programmatically
    check_transition = gr.Button(visible=False, elem_id="check_transition")
    check_transition.click(
        lambda: 4 if app_state["current_conv_index"] >= len(app_state["conversations"]) else None,
        outputs=[current_screen]
    )
    
    # Screen 4 Events: Regenerate or Deploy
    regenerate_btn.click(
        regenerate_prompts,
        outputs=[decision_status, current_screen]
    )
    # Deploy button handling
    deploy_btn.click(
        deploy_prompts,
        outputs=[deployment_status, current_screen]
    )

    # Screen transition handlers based on current_screen value
    current_screen.change(
        lambda x: gr.update(visible=x == 1),
        inputs=[current_screen],
        outputs=[screen1]
    )
    current_screen.change(
        lambda x: gr.update(visible=x == 2),
        inputs=[current_screen],
        outputs=[screen2]
    )
    current_screen.change(
        lambda x: gr.update(visible=x == 3),
        inputs=[current_screen],
        outputs=[screen3]
    )
    current_screen.change(
        lambda x: gr.update(visible=x == 4),
        inputs=[current_screen],
        outputs=[screen4]
    )
    current_screen.change(
        lambda x: gr.update(visible=x == 5),
        inputs=[current_screen],
        outputs=[screen5]
    )

    # Test button handler
    debug_btn.click(
        lambda: [["Test message", "This is a test response that should appear in the chat."]], 
        outputs=[chat_history]
    )

    # Handle specific transitions that need additional actions
    current_screen.change(
        lambda x: process_all_feedback() if x == 4 else (None, None, None),
        inputs=[current_screen],
        outputs=[decision_status, regenerate_btn, deploy_btn]
    )

    current_screen.change(
        lambda x: prepare_conversation_for_display(0) if x == 3 else (None, None, None, None, None),
        inputs=[current_screen],
        outputs=[conversation_display, feedback_input, feedback_status, conversation_id, next_conversation_btn]
    )


# Launch the app
demo.launch()