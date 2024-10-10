import streamlit as st
from datetime import datetime
import random
import emoji
from nlp_trans import create_response_generator, main

# Initialize the model and response generator
generate_response = main()

def setup_page():
    st.set_page_config(page_title="Chatbot for Depression Support", page_icon="💬")

def add_message(role, content):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})

def save_and_clear_chat():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    session_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_sessions.append({
        "name": session_name, 
        "messages": st.session_state.messages.copy()
    })
    st.session_state.messages = []

def handle_emoji(emoji_label):
    emoji_responses = {
        "very_happy": "I'm so happy to hear that you're feeling great! Keep shining!",
        "happy": "I'm glad to hear you're feeling happy! Keep up the positive vibes!",
        "neutral": "It's perfectly okay to feel neutral. How can I assist you today?",
        "slightly_sad": "I'm sorry you're feeling a bit down. I'm here to support you.",
        "sad": "I'm sorry you're feeling sad. Remember, it's okay to feel this way. I'm here for you.",
        "very_sad": "I'm really sorry you're feeling very sad. I'm here for you. Let's talk about it."
    }
    add_message("user", emoji_label)
    response = emoji_responses.get(emoji_label, "How can I assist you today?")
    add_message("bot", response)
    st.session_state.emoji_selected = True

def handle_depression_test():
    st.session_state.depression_test_started = True
    st.session_state.depression_test_completed = False

def calculate_depression_score(responses):
    return sum(responses)

def display_chat_history():
    for message in st.session_state.messages:
        role_emoji = "👤" if message["role"] == "user" else "🤖"
        role_color = "yellow" if message["role"] == "user" else "orange"
        st.markdown(
            f'<p style="color: {role_color};">{role_emoji} {message["role"].capitalize()}:</p> {message["content"]}',
            unsafe_allow_html=True
        )

def main_app():
    setup_page()
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "emoji_selected" not in st.session_state:
        st.session_state.emoji_selected = False
    if "depression_test_started" not in st.session_state:
        st.session_state.depression_test_started = False
    if "depression_score" not in st.session_state:
        st.session_state.depression_score = 0

    st.title("Chatbot for Depression Support")
    
    # Sidebar
    with st.sidebar:
        st.header("Options")
        if st.button("Clear Chat"):
            save_and_clear_chat()

    # Main chat interface
    chat_container = st.container()
    with chat_container:
        display_chat_history()
        
        # Emoji selection
        if not st.session_state.emoji_selected:
            st.markdown("### How are you feeling today?")
            emojis = ["😁", "😊", "😐", "😟", "😢", "😭"]
            emoji_labels = ["very_happy", "happy", "neutral", "slightly_sad", "sad", "very_sad"]
            cols = st.columns(len(emojis))
            for i, (emoji_char, col) in enumerate(zip(emojis, cols)):
                if col.button(emoji_char):
                    handle_emoji(emoji_labels[i])

        # Chat input
        if st.session_state.emoji_selected:
            user_input = st.text_input("Type your message here...")
            if user_input:
                add_message("user", user_input)
                bot_response = generate_response(user_input)
                add_message("bot", bot_response)

    # Depression test button
    if st.button("Take Depression Test"):
        st.session_state.depression_test_started = True

    # Display depression test if started
    if st.session_state.get("depression_test_started", False):
        st.markdown("### Depression Test")
        questions = [
            "How often have you felt little interest or pleasure in doing things?",
            "How often have you felt down, depressed, or hopeless?",
            "How often have you had trouble falling asleep, staying asleep, or sleeping too much?",
            "How often have you felt tired or had little energy?",
            "How often have you had a poor appetite or overeating?",
            "How often have you felt bad about yourself?",
            "How often have you had trouble concentrating on things?",
            "How often have you moved or spoken very slowly or been fidgety?",
            "How often have you had thoughts of self-harm?"
        ]
        
        options = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        responses = []
        
        for question in questions:
            response = st.select_slider(
                question,
                options=options,
                value=options[0]
            )
            responses.append(options.index(response))
        
        if st.button("Submit Test"):
            score = calculate_depression_score(responses)
            st.session_state.depression_score = score
            st.session_state.depression_test_completed = True
            st.session_state.depression_test_started = False

    # Display test results if completed
    if st.session_state.get("depression_test_completed", False):
        score = st.session_state.depression_score
        st.markdown(f"### Your Depression Score: {score}/27")
        
        if score <= 4:
            st.write("Your symptoms suggest minimal depression.")
        elif score <= 9:
            st.write("Your symptoms suggest mild depression.")
        elif score <= 14:
            st.write("Your symptoms suggest moderate depression.")
        elif score <= 19:
            st.write("Your symptoms suggest moderately severe depression.")
        else:
            st.write("Your symptoms suggest severe depression.")
            
        st.write("Remember, this is not a diagnosis. Please consult a mental health professional for proper evaluation.")

if __name__ == "__main__":
    main_app()