"""
Enhanced Streamlit Demo với Long PDF Support
Hỗ trợ xử lý PDF lên đến 50,000 ký tự
Author: Nguyễn Việt Anh - 20215307
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference_enhanced import EnhancedLanguageClassifier

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Enhanced PDF Language Classifier",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
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
    
    .feature-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .chunk-info {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
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
def load_model(chunk_size):
    """Load model với chunk_size cấu hình"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        st.error("❌ Thư mục models không tồn tại!")
        st.stop()
    
    model_folders = [f for f in models_dir.iterdir() 
                     if f.is_dir() and f.name.startswith("xlm-roberta-lang")]
    
    if not model_folders:
        st.error("❌ Không tìm thấy model!")
        st.stop()
    
    latest_model = sorted(model_folders, key=lambda x: x.name)[-1]
    
    try:
        classifier = EnhancedLanguageClassifier(
            str(latest_model),
            chunk_size=chunk_size
        )
        return classifier, latest_model.name
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        st.stop()

def create_probability_chart(probabilities):
    """Tạo biểu đồ xác suất"""
    languages = []
    probs = []
    colors = []
    
    for lang, data in probabilities.items():
        languages.append(LANGUAGE_FULL_NAMES[lang])
        probs.append(data['probability'] * 100)
        colors.append(LANGUAGE_COLORS[lang])
    
    fig = go.Figure(data=[
        go.Bar(
            y=languages,
            x=probs,
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.3)', width=2)),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black'),
        )
    ])
    
    fig.update_layout(
        title="Xác suất dự đoán các ngôn ngữ",
        xaxis_title="Xác suất (%)",
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', range=[0, 100]),
    )
    
    return fig

def create_chunk_visualization(result):
    """Tạo visualization cho chunk predictions"""
    if 'chunk_predictions' not in result or not result.get('chunking_used', False):
        return None
    
    chunks = result['chunk_predictions']
    
    # Prepare data
    chunk_nums = list(range(1, len(chunks) + 1))
    languages = [c['language'] for c in chunks]
    confidences = [c['confidence'] * 100 for c in chunks]
    colors_list = [LANGUAGE_COLORS[lang] for lang in languages]
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Language per Chunk', 'Confidence per Chunk'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Plot 1: Language distribution
    fig.add_trace(
        go.Bar(
            x=chunk_nums,
            y=[1] * len(chunks),
            marker=dict(color=colors_list, line=dict(width=0)),
            showlegend=False,
            hovertemplate='Chunk %{x}<br>%{customdata}<extra></extra>',
            customdata=[f"{LANGUAGE_FULL_NAMES[lang]}" for lang in languages]
        ),
        row=1, col=1
    )
    
    # Plot 2: Confidence trend
    fig.add_trace(
        go.Scatter(
            x=chunk_nums,
            y=confidences,
            mode='lines+markers',
            marker=dict(size=8, color=colors_list),
            line=dict(color='gray', width=2),
            showlegend=False,
            hovertemplate='Chunk %{x}<br>Confidence: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Chunk Number", row=1, col=1)
    fig.update_xaxes(title_text="Chunk Number", row=1, col=2)
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Confidence (%)", row=1, col=2)
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=40))
    
    return fig

# ============ MAIN APP ============
def main():
    # Header
    st.markdown('<div class="main-header">🌐 Enhanced PDF Language Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Phân loại ngôn ngữ từ PDF dài • Hỗ trợ lên đến 50,000 ký tự</div>',
        unsafe_allow_html=True
    )
    
    # Feature badges
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="feature-badge">✨ Chunking Strategy</span>
        <span class="feature-badge">📄 50K Characters</span>
        <span class="feature-badge">🎯 Smart Aggregation</span>
        <span class="feature-badge">📊 Detailed Analytics</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Max characters
        max_chars = st.slider(
            "Max ký tự trích xuất",
            min_value=5000,
            max_value=50000,
            value=50000,
            step=5000,
            help="Số ký tự tối đa trích xuất từ PDF"
        )
        
        # Chunk size
        chunk_size = st.slider(
            "Kích thước chunk (chars)",
            min_value=1000,
            max_value=4000,
            value=2500,
            step=500,
            help="Kích thước mỗi chunk để xử lý. Nhỏ hơn = chính xác hơn nhưng chậm hơn."
        )
        
        st.divider()
        
        st.header("ℹ️ Thông tin")
        st.markdown(f"""
        **📊 Model:** XLM-RoBERTa Base  
        **🎯 Accuracy:** ~96-98%  
        **📏 Max input:** {max_chars:,} chars  
        **🧩 Chunk size:** {chunk_size} chars
        
        ---
        
        **🌍 Ngôn ngữ hỗ trợ:**
        - 🇻🇳 Tiếng Việt
        - 🇯🇵 日本語  
        - 🇰🇷 한국어
        - 🇺🇸 English
        
        ---
        
        **🚀 Chunking Strategy:**
        
        PDF dài → Chia nhỏ → Predict từng chunk → Aggregate kết quả
        
        **Lợi ích:**
        - ✅ Xử lý PDF rất dài
        - ✅ Kết quả chính xác hơn
        - ✅ Hiển thị chi tiết từng chunk
        """)
        
        st.divider()
        st.markdown("Made with ❤️ by Nguyễn Việt Anh - 20215307")
    
    # Load model với chunk_size từ sidebar
    with st.spinner("⏳ Đang tải model..."):
        classifier, model_name = load_model(chunk_size)
    
    st.success(f"✅ Model loaded: {model_name}")
    
    # Upload section
    st.markdown("---")
    st.subheader("📤 Upload file PDF")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Chọn file PDF",
            type=['pdf'],
            help=f"Hỗ trợ PDF text-based, tối đa {max_chars:,} ký tự",
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        # File info
        st.markdown("---")
        st.subheader("📋 Thông tin file")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Tên file", uploaded_file.name[:30] + "..." if len(uploaded_file.name) > 30 else uploaded_file.name)
        with col2:
            st.metric("💾 Kích thước", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("📏 Max chars", f"{max_chars:,}")
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button(
                "🚀 Phân tích PDF",
                type="primary",
                use_container_width=True
            )
        
        if analyze_button:
            # Save to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Predict
            with st.spinner("🔍 Đang phân tích PDF (có thể mất 10-30 giây cho PDF dài)..."):
                try:
                    result = classifier.predict_from_pdf(
                        tmp_path,
                        max_chars=max_chars
                    )
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
                
                # Main result
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 6rem; text-align: center;">{LANGUAGE_FLAGS[lang]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 2.5rem; font-weight: bold; color: white; text-align: center;">{LANGUAGE_FULL_NAMES[lang]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 1.8rem; color: rgba(255,255,255,0.9); text-align: center;">Mức độ ngôn ngữ: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Processing stats
                st.success("✅ Phân tích hoàn tất!")
                
                # Chunk info (nếu có chunking)
                if result.get('chunking_used', False):
                    st.markdown('<div class="chunk-info">', unsafe_allow_html=True)
                    st.markdown(f"""
                    **📊 Thông tin xử lý:**
                    - 📝 Text length: **{result.get('text_length', 0):,} ký tự**
                    - 🧩 Số chunks: **{result.get('num_chunks', 0)}**
                    - 🗳️ Majority vote: **{LANGUAGE_FULL_NAMES[result.get('majority_vote', lang)]}**
                    - 📈 Voting: {result.get('voting_details', {})}
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Charts
                st.markdown("---")
                st.subheader("📊 Chi tiết phân tích")
                
                # Probability chart
                fig_prob = create_probability_chart(result['all_probabilities'])
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Chunk visualization (nếu có)
                if result.get('chunking_used', False):
                    st.markdown("---")
                    st.subheader("🧩 Phân tích từng Chunk")
                    
                    fig_chunks = create_chunk_visualization(result)
                    if fig_chunks:
                        st.plotly_chart(fig_chunks, use_container_width=True)
                        
                        # Chunk details table
                        with st.expander("📋 Xem chi tiết từng chunk"):
                            import pandas as pd
                            
                            chunk_data = []
                            for i, chunk in enumerate(result.get('chunk_predictions', [])):
                                chunk_data.append({
                                    'Chunk': i + 1,
                                    'Language': LANGUAGE_FULL_NAMES[chunk['language']],
                                    'Confidence': f"{chunk['confidence']*100:.2f}%"
                                })
                            
                            df = pd.DataFrame(chunk_data)
                            st.dataframe(df, use_container_width=True)
                
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
                            <div style="text-align: center; font-weight: bold;">{LANGUAGE_FULL_NAMES[lang_code]}</div>
                            <div style="text-align: center; font-size: 1.5rem; color: {LANGUAGE_COLORS[lang_code]};">{data['percentage']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Preview text 1000 chars
                if 'text_preview' in result: 
                    st.markdown("---")
                    with st.expander("📝 Xem trước nội dung PDF"):
                        st.info(f"**Tổng độ dài:** {result.get('text_length', 0):,} ký tự")
                        st.text_area(
                            "Text preview:",
                            result['text_preview'], 
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
    
    else:
        # Placeholder
        st.info("👆 Vui lòng upload file PDF để bắt đầu phân tích")
        
        # Info boxes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Tính năng mới
            
            - ✨ **Chunking Strategy**: Chia PDF dài thành nhiều chunks nhỏ
            - 📊 **Smart Aggregation**: Kết hợp kết quả từ nhiều chunks
            - 📈 **Detailed Analytics**: Xem chi tiết từng chunk
            - 🎯 **Higher Accuracy**: Chính xác hơn với PDF dài
            """)
        
        with col2:
            st.markdown("""
            ### 📋 Hướng dẫn sử dụng
            
            1. Điều chỉnh cấu hình ở sidebar (nếu cần)
            2. Upload file PDF
            3. Nhấn "Phân tích PDF"
            4. Xem kết quả và phân tích chi tiết
            
            **Lưu ý:** PDF dài sẽ mất nhiều thời gian hơn để xử lý.
            """)


if __name__ == "__main__":
    main()