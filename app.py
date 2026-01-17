"""
Streamlit Demo cho PDF Language Classifier
Author: Nguyễn Việt Anh - 20215307
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference import LanguageClassifier

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="PDF Language Classifier",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Result box */
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .flag-display {
        font-size: 6rem;
        text-align: center;
        margin: 1rem 0;
        animation: bounce 1s ease;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    .language-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-display {
        font-size: 1.8rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Info box */
    .info-box {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Upload section */
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============ CONSTANTS ============
LANGUAGE_FLAGS = {
    'vn': '🇻🇳',
    'jp': '🇯🇵',
    'kr': '🇰🇷',
    'us': '🇺🇸'
}

LANGUAGE_FULL_NAMES = {
    'vn': 'Tiếng Việt (Vietnamese)',
    'jp': '日本語 (Japanese)',
    'kr': '한국어 (Korean)',
    'us': 'English'
}

LANGUAGE_COLORS = {
    'vn': '#FF6B6B',
    'jp': '#4ECDC4',
    'kr': '#45B7D1',
    'us': '#96CEB4'
}

# ============ HELPER FUNCTIONS ============
@st.cache_resource
def load_model():
    """Load model (cached để không load lại mỗi lần)"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        st.error("❌ Thư mục models không tồn tại!")
        st.stop()
    
    model_folders = [f for f in models_dir.iterdir() 
                     if f.is_dir() and f.name.startswith("xlm-roberta-lang")]
    
    if not model_folders:
        st.error("❌ Không tìm thấy model! Vui lòng train model trước bằng: `python src/train.py`")
        st.stop()
    
    # Get latest model
    latest_model = sorted(model_folders, key=lambda x: x.name)[-1]
    
    try:
        classifier = LanguageClassifier(str(latest_model))
        return classifier, latest_model.name
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        st.stop()

def create_probability_chart(probabilities):
    """Tạo biểu đồ xác suất"""
    # Prepare data
    languages = []
    probs = []
    colors = []
    
    for lang, data in probabilities.items():
        languages.append(LANGUAGE_FULL_NAMES[lang])
        probs.append(data['probability'] * 100)
        colors.append(LANGUAGE_COLORS[lang])
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=languages,
            x=probs,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=2)
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>Xác suất: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Xác suất dự đoán các ngôn ngữ",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial', 'color': '#333'}
        },
        xaxis_title="Xác suất (%)",
        yaxis_title="",
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        font=dict(size=13, family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, 100]
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig

def create_gauge_chart(confidence):
    """Tạo gauge chart cho confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Độ tin cậy", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 80], 'color': '#FFF4E5'},
                {'range': [80, 100], 'color': '#E5F5E5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    
    return fig

# ============ MAIN APP ============
def main():
    # Header
    st.markdown('<div class="main-header">🌐 PDF Language Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Phân loại ngôn ngữ tự động từ file PDF • Hỗ trợ 4 ngôn ngữ</div>',
        unsafe_allow_html=True
    )
    
    # Load model
    with st.spinner("⏳ Đang tải model..."):
        classifier, model_name = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Thông tin hệ thống")
        
        st.markdown(f"""
        **📊 Model:** XLM-RoBERTa Base  
        **📅 Version:** {model_name.split('-')[-1]}  
        **🎯 Accuracy:** ~96-98%
        
        ---
        
        **🌍 Ngôn ngữ hỗ trợ:**
        - 🇻🇳 **Tiếng Việt** (Vietnamese)
        - 🇯🇵 **日本語** (Japanese)  
        - 🇰🇷 **한국어** (Korean)
        - 🇺🇸 **English**
        
        ---
        
        **📝 Hướng dẫn sử dụng:**
        1. Upload file PDF (text-based)
        2. Nhấn nút "Phân tích"
        3. Xem kết quả dự đoán
        
        ---
        
        **⚙️ Technical Info:**
        - Text extraction: pdfminer.six
        - Max length: 512 tokens
        - Model size: ~560MB
        """)
        
        st.divider()
        
        # Sample files info
        with st.expander("📁 Thử với file mẫu"):
            st.info("Bạn có thể test với các file PDF bất kỳ. Đảm bảo PDF là text-based (không phải scan/ảnh).")
        
        st.divider()
        st.markdown("Made with ❤️ by Nguyễn Việt Anh - 20215307")
    
    # Main content
    st.markdown("---")
    
    # Upload section
    st.subheader("📤 Upload file PDF")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Chọn file PDF cần phân loại ngôn ngữ",
            type=['pdf'],
            help="Chỉ hỗ trợ file PDF text-based (có thể copy text được)",
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        # Display file info
        st.markdown("---")
        st.subheader("📋 Thông tin file")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Tên file", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        with col2:
            st.metric("💾 Kích thước", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("📑 Loại file", "PDF")
        with col4:
            st.metric("🔢 Pages", "N/A")
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button(
                "🚀 Phân tích ngôn ngữ",
                type="primary",
                use_container_width=True
            )
        
        if analyze_button:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Predict
            with st.spinner("🔍 Đang phân tích file PDF..."):
                try:
                    result = classifier.predict_from_pdf(tmp_path)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    Path(tmp_path).unlink(missing_ok=True)
                    return
            
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
            
            # Display results
            if not result['success']:
                st.error(f"❌ {result['error']}")
            else:
                lang = result['language']
                confidence = result['confidence']
                
                # Main result display
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="flag-display">{LANGUAGE_FLAGS[lang]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="language-name">{LANGUAGE_FULL_NAMES[lang]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-display">Độ tin cậy: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("✅ Phân tích hoàn tất!")
                
                # Charts
                st.markdown("---")
                st.subheader("📊 Chi tiết phân tích")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Probability chart
                    fig_prob = create_probability_chart(result['all_probabilities'])
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                with col2:
                    # Gauge chart
                    fig_gauge = create_gauge_chart(confidence)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Detailed probabilities
                st.markdown("---")
                st.subheader("🔢 Xác suất chi tiết")
                
                prob_cols = st.columns(4)
                
                for idx, (lang_code, data) in enumerate(sorted(
                    result['all_probabilities'].items(),
                    key=lambda x: x[1]['probability'],
                    reverse=True
                )):
                    with prob_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 2rem; text-align: center;">{LANGUAGE_FLAGS[lang_code]}</div>
                            <div style="text-align: center; font-weight: bold; margin: 0.5rem 0;">{LANGUAGE_FULL_NAMES[lang_code]}</div>
                            <div style="text-align: center; font-size: 1.5rem; color: {LANGUAGE_COLORS[lang_code]};">{data['percentage']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Text preview
                if 'text_preview' in result:
                    st.markdown("---")
                    with st.expander("📝 Xem trước nội dung PDF"):
                        st.info(f"**Độ dài text:** {result['text_length']:,} ký tự")
                        st.text_area(
                            "Text đã trích xuất:",
                            result['text_preview'],
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
    
    else:
        # Placeholder when no file
        st.info("👆 Vui lòng upload file PDF để bắt đầu phân tích")
        
        # Demo section
        st.markdown("---")
        st.subheader("🎯 Ví dụ text phân loại")
        
        demo_tab1, demo_tab2, demo_tab3, demo_tab4 = st.tabs([
            f"{LANGUAGE_FLAGS['vn']} Tiếng Việt",
            f"{LANGUAGE_FLAGS['jp']} 日本語",
            f"{LANGUAGE_FLAGS['kr']} 한국어",
            f"{LANGUAGE_FLAGS['us']} English"
        ])
        
        demo_texts = {
            'vn': "Trí tuệ nhân tạo đang thay đổi cách chúng ta sống và làm việc. Các mô hình ngôn ngữ lớn như GPT và Claude có khả năng hiểu và tạo ra văn bản một cách tự nhiên.",
            'jp': "人工知能は私たちの生活や仕事のやり方を変えています。GPTやClaudeのような大規模言語モデルは、自然な形でテキストを理解し生成する能力を持っています。",
            'kr': "인공지능은 우리가 살고 일하는 방식을 변화시키고 있습니다. GPT 및 Claude와 같은 대규모 언어 모델은 자연스럽게 텍스트를 이해하고 생성할 수 있습니다.",
            'us': "Artificial intelligence is transforming the way we live and work. Large language models like GPT and Claude have the ability to understand and generate text naturally."
        }
        
        with demo_tab1:
            st.write(demo_texts['vn'])
            if st.button("Test với text này", key="demo_vn"):
                result = classifier.predict_from_text(demo_texts['vn'])
                st.success(f"Dự đoán: **{result['language_name']}** ({result['confidence_percent']})")
        
        with demo_tab2:
            st.write(demo_texts['jp'])
            if st.button("Test với text này", key="demo_jp"):
                result = classifier.predict_from_text(demo_texts['jp'])
                st.success(f"Dự đoán: **{result['language_name']}** ({result['confidence_percent']})")
        
        with demo_tab3:
            st.write(demo_texts['kr'])
            if st.button("Test với text này", key="demo_kr"):
                result = classifier.predict_from_text(demo_texts['kr'])
                st.success(f"Dự đoán: **{result['language_name']}** ({result['confidence_percent']})")
        
        with demo_tab4:
            st.write(demo_texts['us'])
            if st.button("Test với text này", key="demo_us"):
                result = classifier.predict_from_text(demo_texts['us'])
                st.success(f"Dự đoán: **{result['language_name']}** ({result['confidence_percent']})")


if __name__ == "__main__":
    main()