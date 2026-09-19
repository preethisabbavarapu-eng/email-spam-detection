import joblib
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL & VECTORIZER
# ============================================================
@st.cache_resource
def load_assets():
    tfidf = joblib.load("sms_spam_tfidf_bigrams.pkl")
    svm_model = joblib.load("sms_spam_svm_bigrams.pkl")
    return tfidf, svm_model


try:
    tfidf, svm_model = load_assets()
except Exception as e:
    st.error(
        f"Unable to load model assets. Please check if file paths are correct. Error: {e}"
    )
    st.stop()


# Initialize Session State for text area input
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Reference Messages Dataset (Your Provided Examples)
REFERENCE_MESSAGES = {
    "Select a reference message...": "",
    "[SPAM] Prize Winner Alert": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "[SPAM] Free Mobile Upgrade": "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
    "[SPAM] Paid Message Offer": "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv",
    "[HAM] Conversational Message": "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
    "[HAM] Personal Reaction": "Even my brother is not like to speak with me. They treat me like aids patent.",
    "[HAM] Service Notification": "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune",
}


def set_text(text):
    st.session_state["input_text"] = text


# ============================================================
# SIDEBAR: MODEL PERFORMANCE METRICS
# ============================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/shield-against-piracy.png", width=64
    )
    st.title("Model Insights")
    st.caption("Linear SVM + TF-IDF (Bigrams)")

    st.markdown("---")
    st.subheader("Performance Metrics")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", "97.86%")
        st.metric("Precision", "92.62%")
    with col_b:
        st.metric("ROC-AUC", "98.17%")
        st.metric("Recall", "89.68%")

    st.metric("F1 Score", "91.13%")

    st.markdown("---")
    with st.expander("Confusion Matrix Details"):
        st.write("**True Negatives (TN):** 891")
        st.write("**True Positives (TP):** 113")
        st.write("**False Positives (FP):** 9")
        st.write("**False Negatives (FN):** 13")


# ============================================================
# MAIN INTERFACE
# ============================================================
st.title("🛡️ SMS Spam Classification Engine")
st.markdown(
    "Analyze text messages in real-time using modern NLP vectorization and Linear Support Vector Machines."
)

st.markdown("---")

# Dropdown Reference Selector
selected_option = st.selectbox(
    "📥 **Load a sample reference message:**",
    options=list(REFERENCE_MESSAGES.keys()),
)

if (
    selected_option != "Select a reference message..."
    and REFERENCE_MESSAGES[selected_option]
):
    st.session_state["input_text"] = REFERENCE_MESSAGES[selected_option]

# Quick Action Buttons
st.markdown("**Or pick a category shortcut:**")
b_col1, b_col2, b_col3, b_col4 = st.columns(4)

if b_col1.button("🚨 Prize Reward (Spam)"):
    set_text(
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
    )
    st.rerun()

if b_col2.button("🚨 Free Upgrade (Spam)"):
    set_text(
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
    )
    st.rerun()

if b_col3.button("💬 Personal Chat (Ham)"):
    set_text(
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
    )
    st.rerun()

if b_col4.button("🎵 Caller Tune (Ham)"):
    set_text(
        "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"
    )
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Main Form
with st.form(key="classifier_form"):
    user_input = st.text_area(
        label="SMS Input Text",
        value=st.session_state["input_text"],
        placeholder="Type, paste, or select a sample SMS message above...",
        height=140,
    )

    submit_button = st.form_submit_button(
        label="⚡ Analyze Message", use_container_width=True
    )

# ============================================================
# PREDICTION & RESULTS
# ============================================================
if submit_button or st.session_state["input_text"]:
    cleaned_input = user_input.strip()

    if not cleaned_input:
        st.warning("Please enter or select an SMS message to analyze.")
    else:
        # Transform input
        X_vec = tfidf.transform([cleaned_input])

        # Predict & Score
        prediction = svm_model.predict(X_vec)[0]
        score = svm_model.decision_function(X_vec)[0]

        st.markdown("### Analysis Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.error("### 🚨 SPAM DETECTED")
                st.caption(
                    "This message matches known patterns for unsolicited or promotional text."
                )
            else:
                st.success("### ✅ LEGITIMATE (HAM)")
                st.caption(
                    "This message appears to be safe standard communication."
                )

        with res_col2:
            st.markdown("**SVM Hyperplane Decision Score**")
            st.write(f"Raw Decision Score: `{score:.4f}`")

            if score > 0:
                norm_score = min(100, int(50 + (score * 15)))
                st.progress(
                    norm_score, text=f"Spam Intensity Signal: {norm_score}%"
                )
            else:
                norm_score = max(0, int(50 + (score * 15)))
                st.progress(
                    norm_score, text=f"Ham Confidence Level: {100 - norm_score}%"
                )

        st.divider()

        with st.expander("Technical Model Breakdown"):
            st.json(
                {
                    "Input Character Count": len(cleaned_input),
                    "Input Word Count": len(cleaned_input.split()),
                    "Prediction Class": "SPAM (1)"
                    if prediction == 1
                    else "HAM (0)",
                    "SVM Decision Function Score": float(score),
                    "Vectorization": "TF-IDF (Unigrams + Bigrams)",
                }
            )