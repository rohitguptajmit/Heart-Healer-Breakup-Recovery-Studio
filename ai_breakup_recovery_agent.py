import logging
import os
import tempfile
from pathlib import Path

from typing import List, Optional

import streamlit as st

from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Configure logging for errors only
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def initialize_agents(api_key: str) -> tuple[Agent, Agent, Agent, Agent]:
    """
    Initialize all agents using OpenAI instead of Gemini.
    """
    try:
        # OpenAIChat wrapper from Agno – supports text + images with GPT-4o mini
        model = OpenAIChat(
            id="gpt-4o-mini",
            api_key=api_key,
        )

        # 1) Emotional support agent
        therapist_agent = Agent(
            model=model,
            name="Supportive Listener",
            instructions=[
                "You are a warm, empathetic breakup-support companion who:",
                "1. Listens deeply and validates the user's emotions",
                "2. Uses gentle, light humour where appropriate",
                "3. Normalizes their feelings and shares relatable breakup perspectives",
                "4. Offers reassuring, encouraging words and hope",
                "5. Can use both the text and screenshots to understand emotional context",
                "Always be kind, non-judgmental, and emotionally safe And limit the answer in 70 Words"
            ],
            markdown=True,
        )

        # 2) Closure-writing agent
        closure_agent = Agent(
            model=model,
            name="Closure Coach",
            instructions=[
                "You help the user write for emotional closure, not to actually send.",
                "1. Draft unsent messages to their ex to release emotions",
                "2. Allow honest, raw but non-abusive expression",
                "3. Organize content with simple headings or sections",
                "4. Gently steer toward self-respect and letting go",
                "Focus on emotional release and healthy closure  And limit the answer in 50 Words."
            ],
            markdown=True,
        )

        # 3) Routine / healing plan agent
        routine_planner_agent = Agent(
            model=model,
            name="Routine Architect",
            instructions=[
                "You design short, practical breakup-recovery routines.",
                "1. Create a 7-day healing plan with daily tasks",
                "2. Mix self-care, movement, journaling, and social connection",
                "3. Add ideas for digital / social media boundaries",
                "4. Suggest mood-lifting playlist themes or song ideas",
                "Keep it simple, doable, and encouraging.  And limit the answer in 80 Words"
            ],
            markdown=True,
        )

        # 4) Direct reality-check agent
        brutal_honesty_agent = Agent(
            model=model,
            name="Reality Check Partner",
            tools=[DuckDuckGoTools()],
            instructions=[
                "You give honest but respectful perspective about the breakup.",
                "1. Explain what likely went wrong in clear language",
                "2. Avoid sugar-coating, but never insult the user",
                "3. Highlight patterns to avoid in the future",
                "4. List reasons why moving on is good for them",
                "Be direct, but always constructive and growth-focused.  And limit the answer in 50 Words"
            ],
            markdown=True,
        )

        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent

    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return None, None, None, None


# --------- Streamlit UI ---------

st.set_page_config(
    page_title="💟 Heart Healer – Breakup Recovery Studio",
    page_icon="💟",
    layout="wide",
)

# Sidebar for API key input
with st.sidebar:
    st.header("🔑 API Settings")

    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""

    api_key = st.text_input(
        "Enter your OpenAI API Key",
        value=st.session_state.api_key_input,
        type="password",
        help="You can create a key from the OpenAI dashboard (API keys section).",
        key="api_key_widget",
    )

    if api_key != st.session_state.api_key_input:
        st.session_state.api_key_input = api_key

    st.caption(
        "Tip: You can also export `OPENAI_API_KEY` as an environment variable. "
        "This field is just for convenience."
    )

    if api_key:
        st.success("API key detected – you’re ready to go ✅")
    else:
        st.warning("Please enter your OpenAI API key to use the app.")


# Main content
st.title("💟 Heart Healer – Breakup Recovery Studio")
st.markdown(
    """
### You don’t have to process this alone.
Share what happened, drop in a few chat screenshots if you’d like, and your AI support team will build a gentle, structured recovery plan for you.
"""
)

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Tell Your Side of the Story")
    user_input = st.text_area(
        "How are you feeling? What would you like to share?",
        height=170,
        placeholder="Write whatever feels comfortable – there’s no right or wrong here.",
    )

with col2:
    st.subheader("📷 (Optional) Add Chat Screenshots")
    uploaded_files = st.file_uploader(
        "Upload screenshots (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="screenshots",
    )

    if uploaded_files:
        st.caption("Preview of your uploaded screenshots:")
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)


def process_images(files) -> List[AgnoImage]:
    """
    Save uploaded Streamlit files to temp paths and wrap them as Agno Images.
    """
    processed_images: List[AgnoImage] = []
    if not files:
        return processed_images

    temp_dir = tempfile.gettempdir()

    for file in files:
        try:
            temp_path = os.path.join(temp_dir, f"heart_healer_{file.name}")
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            processed_images.append(AgnoImage(filepath=Path(temp_path)))
        except Exception as e:
            logger.error(f"Error processing image {file.name}: {str(e)}")
            continue

    return processed_images


# Process button and API key check
if st.button("✨ Build My Healing Plan", type="primary"):
    if not st.session_state.api_key_input:
        st.warning("Please enter your OpenAI API key in the sidebar first.")
    else:
        therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(
            st.session_state.api_key_input
        )

        if all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
            if user_input or uploaded_files:
                try:
                    all_images = process_images(uploaded_files)

                    st.header("🌈 Your Personalized Recovery Journey")

                    # 1) Emotional support
                    with st.spinner("🤗 Offering emotional first aid..."):
                        therapist_prompt = f"""
You are helping someone who is going through a breakup.

User's message:
{user_input}

Using their words (and any images, if provided), please:
1. Validate what they are feeling without minimizing it.
2. Normalize their reaction (why it makes sense to feel this way).
3. Offer a few comforting, human-sounding reflections.
4. Give 2–3 gentle suggestions for what they can do *today* to feel 1% better.
Keep the tone warm, conversational, and kind but in limited words.
"""
                        response = therapist_agent.run(
                            therapist_prompt,
                            images=all_images,
                        )
                        st.subheader("🤗 Emotional First Aid")
                        st.markdown(response.content)

                    # 2) Closure messages
                    with st.spinner("✍️ Drafting letters you’ll never send..."):
                        closure_prompt = f"""
Help the user process their breakup by writing for closure.

User's feelings / story:
{user_input}

Please create:
1. One unsent message to their ex (for their eyes only, not to actually send).
2. A short journal prompt to help them explore what they learned.
3. A brief self-compassion note they can read to themselves.

Tone: honest, respectful, and focused on self-worth but in limited words.
"""
                        response = closure_agent.run(
                            closure_prompt,
                            images=all_images,
                        )
                        st.subheader("✍️ Letters for Closure (Not to Send)")
                        st.markdown(response.content)

                    # 3) Recovery routine
                    with st.spinner("📅 Designing your 7-day healing gameplan..."):
                        routine_prompt = f"""
Design a simple 7-day breakup healing plan.

Context about the user:
{user_input}

Include for each day:
1. One small self-care or body movement activity.
2. One reflection or journaling idea.
3. One optional social connection / reaching out action.
4. Any digital / social media guideline if relevant.

Keep it realistic for someone who has low energy and is emotionally tired but in limited words.
"""
                        response = routine_planner_agent.run(
                            routine_prompt,
                            images=all_images,
                        )
                        st.subheader("📅 7-Day Healing Gameplan")
                        st.markdown(response.content)

                    # 4) Honest perspective
                    with st.spinner("💡 Offering a grounded reality check..."):
                        honesty_prompt = f"""
Provide an honest but compassionate perspective on this breakup.

User's situation:
{user_input}

Please cover:
1. A clear, neutral analysis of what might have gone wrong.
2. Patterns or red flags they might want to notice for the future.
3. Reasons this breakup might secretly be protecting or freeing them.
4. 3–5 concrete growth steps they can focus on over the next month.

Be direct but never cruel. The goal is clarity + growth, not blame but in limited words.
"""
                        response = brutal_honesty_agent.run(
                            honesty_prompt,
                            images=all_images,
                        )
                        st.subheader("💡 Reality Check & Growth Notes")
                        st.markdown(response.content)

                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error("An error occurred while generating your recovery plan. Please try again.")
            else:
                st.warning("Please share your feelings or upload at least one screenshot to get a recovery plan.")
        else:
            st.error("Failed to initialize agents. Please double-check your OpenAI API key.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; font-size: 0.9rem; opacity: 0.85;'>
    <p>💟 This tool is not a replacement for professional therapy.</p>
    <p>If you feel overwhelmed or unsafe, please reach out to a trusted person or local helpline.</p>
</div>
""",
    unsafe_allow_html=True,
)
