import streamlit as st
import pandas as pd
import joblib
import re
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import isodate

# ========================
# 🔐 Load API Key
# ========================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

if not API_KEY:
    st.error("❌ API_KEY not found. Check your .env file.")
else:
    st.info("✅ API key loaded successfully.")


# ========================
# 🤖 Load Trained Model
# ========================
model = joblib.load("youtube_revenue_model.pkl")

# ========================
# 🧠 Helper Functions
# ========================
def extract_video_id(url: str):
    """Extract YouTube video ID from URL"""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def parse_duration(duration):
    """Convert ISO 8601 duration to minutes"""
    try:
        dur = isodate.parse_duration(duration)
        return dur.total_seconds() / 60
    except:
        return 0

def calc_engagement_rate(stats):
    """Calculate engagement rate safely"""
    views = int(stats.get("viewCount", 0))
    likes = int(stats.get("likeCount", 0))
    comments = int(stats.get("commentCount", 0))
    return (likes + comments) / views if views > 0 else 0

def get_video_stats(video_id: str):
    """Fetch video + channel stats via YouTube Data API"""
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)

        # ---- Fetch video info ----
        video_request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        video_response = video_request.execute()

        if not video_response.get("items"):
            return None

        video = video_response["items"][0]
        stats = video["statistics"]
        snippet = video["snippet"]
        duration = video["contentDetails"]["duration"]

        # ---- Fetch channel info ----
        channel_id = snippet.get("channelId", None)
        subscribers = 0
        country = "US"

        if channel_id:
            channel_request = youtube.channels().list(
                part="statistics,snippet",
                id=channel_id
            )
            channel_response = channel_request.execute()
            if channel_response.get("items"):
                channel_info = channel_response["items"][0]
                subscribers = int(channel_info["statistics"].get("subscriberCount", 0))
                country = channel_info["snippet"].get("country", "US")

        # ---- Build Feature Dict ----
        data = {
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
            "watch_time_minutes": int(stats.get("viewCount", 0)) * 5,  # Approximation
            "video_length_minutes": parse_duration(duration),
            "engagement_rate": calc_engagement_rate(stats),
            "subscribers": subscribers,
            "category": snippet.get("categoryId", "22"),  # Default: People & Blogs
            "country": country,
            "device": "Mobile"  # Default assumption
        }
        return data
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# ========================
# 🧩 Streamlit UI
# ========================
st.set_page_config(page_title="YouTube Ad Revenue Predictor", page_icon="🎥", layout="centered")
st.title("🎥 YouTube Ad Revenue Predictor")

st.write("Choose your preferred input method:")

# 🔘 Select Input Mode
mode = st.radio("Select Input Mode", ["Manual Input", "YouTube Link Input"])

# ---------- MANUAL INPUT ----------
if mode == "Manual Input":
    st.subheader("🧠 Manual Input Mode")

    with st.form("manual_form"):
        views = st.number_input("Views", min_value=0)
        likes = st.number_input("Likes", min_value=0)
        comments = st.number_input("Comments", min_value=0)
        watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0)
        video_length_minutes = st.number_input("Video Length (minutes)", min_value=0)
        engagement_rate = st.number_input("Engagement Rate", min_value=0.0)
        subscribers = st.number_input("Subscribers", min_value=0)
        category = st.selectbox("Category", ["Education", "Entertainment", "Gaming", "Tech", "Music","Lifestyle"])
        country = st.selectbox("Country", ['IN', 'CA', 'UK', 'US', 'DE', 'AU'])
        device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet","TV"])

        submit_manual = st.form_submit_button("🔮 Predict Revenue")

    if submit_manual:
        data = pd.DataFrame([{
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time_minutes,
            "video_length_minutes": video_length_minutes,
            "engagement_rate": engagement_rate,
            "subscribers": subscribers,
            "category": category,
            "country": country,
            "device": device
        }])
        pred = model.predict(data)[0]
        st.success(f"💰 Estimated Ad Revenue: ${pred:,.2f}")

# ---------- LINK INPUT ----------
elif mode == "YouTube Link Input":
    st.subheader("🔗 YouTube Link Mode")

    with st.form("youtube_form"):
        url = st.text_input(
            "Paste YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        submit_link = st.form_submit_button("📥 Fetch Video Stats")

    if submit_link:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
        else:
            st.info(f"🎬 Extracted Video ID: `{video_id}`")
            stats = get_video_stats(video_id)

            if stats:
                # ✅ Category Mapping (convert YouTube categoryId to trained label)
                category_map = {
                    "1": "Entertainment",       # Film & Animation → Entertainment
                    "2": "Tech",                # Autos & Vehicles → Tech
                    "10": "Music",              # Music → Music
                    "20": "Gaming",             # Gaming → Gaming
                    "22": "Lifestyle",          # People & Blogs → Lifestyle
                    "23": "Entertainment",      # Comedy → Entertainment
                    "24": "Entertainment",      # Entertainment → Entertainment
                    "27": "Education",          # Education → Education
                    "28": "Tech"                # Science & Technology → Tech
                }

                cat_id = str(stats.get("category", "22"))
                stats["category"] = category_map.get(cat_id, "Entertainment")

                st.success("✅ Successfully fetched video statistics!")
                st.session_state["video_stats"] = stats
            else:
                st.error("❌ Could not fetch video stats. Check API key or quota.")

    # Step 3: If data is stored, show customization UI
    if "video_stats" in st.session_state:
        stats = st.session_state["video_stats"]
        st.markdown("### 📊 Video Statistics Fetched:")
        st.json(stats)

        st.markdown("### 🛠 Customize Inputs Before Prediction")

        # 🎛️ Editable fields prefilled with fetched values
        country = st.selectbox(
            "🌍 Country",
            ['US', 'IN', 'UK', 'CA', 'AU', 'DE'],
            index=0 if stats["country"] not in ['US', 'IN', 'UK', 'CA', 'AU', 'DE'] else
            ['US', 'IN', 'UK', 'CA', 'AU', 'DE'].index(stats["country"])
        )

        category = st.selectbox(
            "🎭 Category",
            ["Education", "Entertainment", "Gaming", "Tech", "Music", "Lifestyle"],
            index=["Education", "Entertainment", "Gaming", "Tech", "Music", "Lifestyle"].index(stats["category"])
            if stats["category"] in ["Education", "Entertainment", "Gaming", "Tech", "Music", "Lifestyle"]
            else 1
        )

        device = st.selectbox(
            "💻 Device",
            ["Mobile", "Desktop", "Tablet", "TV"],
            index=0
        )

        # ✅ Update stats with user selections
        stats["country"] = country
        stats["category"] = category
        stats["device"] = device

        if st.button("🔮 Predict Revenue"):
            data = pd.DataFrame([stats])
            pred = model.predict(data)[0]
            st.success(f"💰 Estimated Ad Revenue: **${pred:,.2f}**")


