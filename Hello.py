import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit.components.v1 import html
import google.generativeai as genai
import os
from PIL import Image


load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))


LOGGER = get_logger(__name__)
st.set_page_config(
      page_title="Multi Modal AI",
      page_icon="🤖",
      initial_sidebar_state="collapsed"
  )

def get_gemini_repsonse(input,image,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0],prompt])
    return response.text

def gimini_pro(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
    
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type, 
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("Multi-Modal AI Chatbot")


temperature = st.slider(
    "Select the creativity temperature for the AI",
    min_value=0.0,  # Minimum value of the slider
    max_value=1.0,  # Maximum value of the slider
    value=0.5,  # Default value
    step=0.01,  # Set the step size for finer control
)
if temperature > 0.75:
    emoji = "🔥"
elif temperature > 0.5:
    emoji = "🌡️"
elif temperature > 0.25:
    emoji = "🌬️"
else:
    emoji = "❄️"
st.write(f"Creativity Temperature: {temperature} {emoji}")


def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    # Define the temperature range for your slider, as well as the emoji labels
    
    # Show the selected temperature with the corresponding emoji

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
    js= """
    alert("hello world");
    """
    my_html = f"<script>{js}</script>"
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))

input= ""
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Read the Image Sequence by sequence")

input_prompt="""
You are an expert in debuuging and describe images, read image very carefully and read user prompt and resopnse why this error happen
and provide a details solution

----
----
"""

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_repsonse(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)


