import gradio as gr
import re, string, requests, time
from bs4 import BeautifulSoup
from googlesearch import search
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------- Utilities ----------------------
class TextCleaner:
    def __init__(self):
        self.punctuation_table = str.maketrans('', '', string.punctuation + '«»…“”–')

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(self.punctuation_table)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

arabic_stopwords = [
    'في', 'من', 'على', 'و', 'عن', 'إلى', 'التي', 'الذي', 'أن', 'إن', 'كان', 'كما', 'لذلك', 'لكن',
    'أو', 'ما', 'لا', 'لم', 'لن', 'قد', 'هذا', 'هذه', 'هو', 'هي', 'هم', 'ثم', 'كل', 'هناك', 'بعد'
]

# ---------------------- Web Search Agent ----------------------
class GoogleSearch:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.cache = {}

    def get_page_text(self, url):
        if url in self.cache:
            return self.cache[url]
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                self.cache[url] = text[:1500]
                return self.cache[url]
        except:
            pass
        return ""

    def search(self, query):
        results = {}
        try:
            urls = search(query, lang='ar', num_results=3)
            for url in urls:
                content = self.get_page_text(url)
                if content:
                    results[url] = content
        except Exception as e:
            print(f"Search error: {e}")
        return results

class WebVerifier:
    def __init__(self):
        self.searcher = GoogleSearch()
        self.cleaner = TextCleaner()
        self.bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def extract_keywords(self, text):
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean_text(text)
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=arabic_stopwords).fit([cleaned_text])
        candidates = vectorizer.get_feature_names_out()
        doc_embedding = self.bert_model.encode([cleaned_text], convert_to_tensor=True)
        candidate_embeddings = self.bert_model.encode(candidates, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(doc_embedding, candidate_embeddings)[0]
        top_k_indices = similarities.topk(k=min(5, len(candidates))).indices
        top_keywords = [candidates[idx] for idx in top_k_indices]
        return " ".join(top_keywords)

    def get_similarity_score(self, input_text, sources):
        input_embed = self.bert_model.encode(input_text, convert_to_tensor=True)
        best_score = 0.0
        best_url = None
        for url, content in sources.items():
            cleaned = self.cleaner.clean_text(content)
            para_embed = self.bert_model.encode(cleaned, convert_to_tensor=True)
            score = util.pytorch_cos_sim(input_embed, para_embed).item()
            if score > best_score:
                best_score = score
                best_url = url
        return best_url, best_score

    def verify(self, text):
        keywords = self.extract_keywords(text)
        sources = self.searcher.search(keywords)
        return self.get_similarity_score(text, sources)

# ---------------------- Fake News Model ----------------------
tokenizer = AutoTokenizer.from_pretrained("Aseelalzaben03/sadaqai-bestmodel")
model = AutoModelForSequenceClassification.from_pretrained("Aseelalzaben03/sadaqai-bestmodel")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

verifier = WebVerifier()

# ---------------------- Final Verdict ----------------------
# ---------------------- Final Verdict ----------------------
def final_verdict(news_text):
    if not news_text.strip():
        return gr.update(value="❗ الرجاء إدخال نص للتحقق", visible=True), "white"

    model_result = classifier(news_text)[0]
    model_label = model_result['label']
    model_score = model_result['score']

    try:
        url, web_score = verifier.verify(news_text)
    except Exception as e:
        print(f"Web verification failed: {e}")
        url, web_score = None, 0.0

    final_score = round((model_score + web_score) / 2, 2)

    if final_score >= 0.65:
        verdict = "✅ الخبر موثوق"
        color = "#d4edda"
    elif final_score <= 0.35:
        verdict = "❌ الخبر غير موثوق"
        color = "#f8d7da"
    else:
        verdict = "⚠ الخبر مشكوك فيه"
        color = "#fff3cd"



    trusted_sources = ["https://www.aljazeera.net", "https://arabic.cnn.com", "https://www.bbc.com/arabic"]
    source_links = "\n".join([f"🔗 {link}" for link in trusted_sources]) if verdict != "✅ الخبر موثوق" else "-"

    result_text = f"""
    <div style='text-align:center;'>
        <strong>{verdict}</strong><br><br>
        <strong>📊 النسبة الكلية للثقة:</strong> {final_score}<br>
        <strong>🔗 المصدر الأقرب:</strong> {url if url else 'لا يوجد'}<br><br>
        <strong>📚 مصادر موثوقة:</strong><br>{source_links}
    </div>
    """

    return result_text, color

# ---------------------- Interface ----------------------
with gr.Blocks(css="""
    body { background-color: #fff0f0; direction: rtl; font-family: 'Arial'; }
    h1, h2, label, .markdown-text, textarea, input { text-align: center !important; }
    .gr-textbox textarea { text-align: center !important; }
    #description { text-align: center !important; }
""") as demo:
    gr.Markdown("# 🔎 صدق AI - منصة التحقق من الأخبار العربية")
    gr.Markdown("## 🧠 صدق AI هي منصة ذكية تساعدك على التحقق من صحة الأخبار باللغة العربية.\nتقوم بتحليل الخبر، ومقارنته بمصادر موثوقة، وتقديم تقييم دقيق حول موثوقيته.", elem_id="description")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="✍️ أدخل نص الخبر", placeholder="اكتب الخبر هنا...", lines=4)
        with gr.Column():
            url_input = gr.Textbox(label="🌐 أدخل رابط للمقال (اختياري)", placeholder="ضع الرابط هنا...", lines=2)

    check_btn = gr.Button("✅ تحقق الآن")
    status_bar = gr.Textbox(visible=False, interactive=False, show_label=False)
    result_box = gr.HTML(visible=False)
    bg_color = gr.State("white")

    def wrapped_verification(news_text, url_text):
        full_text = news_text.strip()
        if url_text.strip():
            full_text += f" {url_text.strip()}"

        # أولاً، أظهر رسالة "جاري التحقق"
        yield gr.update(visible=True, value="🔄 جاري التحقق من الخبر..."), gr.update(visible=False)
        
        time.sleep(1)

        # ثانياً، نفذ التحقق الفعلي
        result, color = final_verdict(full_text)

        # ثالثاً، أخفِ الـ status bar وفعّل النتيجة
        yield gr.update(visible=False), gr.update(value=result, visible=True)

    check_btn.click(
        fn=wrapped_verification,
        inputs=[text_input, url_input],
        outputs=[status_bar, result_box],
        show_progress="full"
    )

# ---------------------- Run ----------------------
demo.launch(share=True)
