import streamlit as st
import pickle
import time

# --- PAGE CONFIGURATION ---
# "layout='wide'" uses the full width of the screen
st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS TO HIDE DEPLOY BUTTON ---
st.markdown(
    """
    <style>
    /* Hides the Deploy button */
    [data-testid="stAppDeployButton"] {
        display: none;
    }
    /* Optional: Hides the Streamlit main menu (hamburger) for a cleaner look */
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- LOAD AI MODELS ---
@st.cache_resource # This speeds up the app by keeping the model in memory
def load_models():
    try:
        with open('models/spam_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('models/vectorizer.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3296/3296464.png", width=100) # Cool shield icon
    st.title("About SpamShield")
    st.write("This AI uses Natural Language Processing (NLP) and a Naive Bayes classifier to detect malicious emails.")
    
    st.divider()
    st.markdown("### 🛠️ Features included:")
    st.markdown("- 🧠 Machine Learning Engine\n- 📊 Probability Scoring\n- 🧹 Automated Text Cleaning\n- ⚡ Real-Time Analysis")
    
    st.divider()
    st.caption("Developed by AKHIL!")

# --- MAIN DASHBOARD ---
st.title("🛡️ SpamShield AI Dashboard")
st.markdown("Identify spam, phishing attempts, and promotional clutter instantly using Artificial Intelligence.")
st.write("---")

# Stop the app here if models are missing
if model is None or vectorizer is None:
    st.error("🚨 Error: Trained model files not found! Please run 'python train.py' in your terminal first.")
    st.stop()

# --- CREATE TWO COLUMNS ---
# col1 is twice as wide as col2
col1, col2 = st.columns([2, 1])

# --- LEFT COLUMN: USER INPUT ---
with col1:
    st.subheader("✉️ Enter Email Details")
    
    # We use a form so the page doesn't refresh until the user clicks "Analyze"
    with st.form(key="email_form"):
        email_subject = st.text_input("Email Subject Line", placeholder="e.g., URGENT: Your account has been suspended!")
        email_body = st.text_area("Email Body Content", placeholder="Paste the full email message here...", height=250)
        
        # A big, wide button
        submit_button = st.form_submit_button(label="🔍 Analyze Email with AI", use_container_width=True)

# --- RIGHT COLUMN: RESULTS & ANALYSIS ---
with col2:
    st.subheader("📊 Analysis Results")
    
    if submit_button:
        if email_subject.strip() == "" and email_body.strip() == "":
            st.warning("⚠️ Please provide some text to analyze!")
        else:
            # 1. Show a professional loading animation
            with st.spinner("AI is analyzing text patterns..."):
                time.sleep(1) # Adds a 1-second delay for a realistic scanning effect
                
                # 2. Combine and prepare text
                full_email_text = email_subject + " " + email_body
                transformed_input = vectorizer.transform([full_email_text])
                
                # 3. Get Prediction AND Confidence Score
                prediction = model.predict(transformed_input)[0]
                prediction_probabilities = model.predict_proba(transformed_input)[0]
                
                # Grab the max probability and convert it to a percentage
                confidence_score = max(prediction_probabilities) * 100
                
                # 4. Display visually appealing results
                if prediction.lower() == 'spam':
                    st.error("🚨 **THREAT DETECTED: SPAM**")
                    st.metric(label="Spam Probability", value=f"{confidence_score:.2f}%")
                    st.write("⚠️ **Warning:** Do not click links or share personal info.")
                else:
                    st.success("✅ **SAFE: LEGITIMATE EMAIL (HAM)**")
                    st.metric(label="Safety Confidence", value=f"{confidence_score:.2f}%")
                    st.write("👍 This looks like a normal email.")
                
                st.divider()
                
                # 5. Display Text Analytics
                st.markdown("### 📝 Text Analytics")
                word_count = len(full_email_text.split())
                char_count = len(full_email_text)
                
                st.write(f"**Word Count:** {word_count} words")
                st.write(f"**Character Count:** {char_count} characters")
    else:
        st.info("Waiting for email input... Paste an email on the left and click Analyze.")

