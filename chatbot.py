from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carregar modelo e tokenizador pré-treinado
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Função para gerar resposta
def generate_response(input_text):
    # Converte a entrada para lowercase para facilitar a comparação
    input_lower = input_text.lower()
    
    # Regras manuais para perguntas sobre você
    if "your name" in input_lower or "who are you" in input_lower:
        return "Hello! My name is Pedro Henrique Campos Moreira. I’m an AI enthusiast and Computer Vision developer studying at the Federal University of Viçosa (UFV)."

    elif "study" in input_lower or "university" in input_lower or "ufv" in input_lower:
        return (
            "I’m pursuing a Bachelor’s in Information Systems at the Federal University of Viçosa (UFV), "
            "expected to graduate in early 2026. I also hold a Technical Diploma in Systems Analysis & Development "
            "from SENAI – CETEL and a Postgraduate Certificate in Robotics from CEFET."
        )

    elif "objective" in input_lower or "career goal" in input_lower:
        return (
            "My goal is to lead AI and Computer Vision initiatives that drive innovation in agriculture, "
            "healthcare, and smart industry by combining robust ML pipelines, cloud architectures, and edge inference."
        )

    elif "summary" in input_lower or "professional summary" in input_lower:
        return (
            "I’m a results-driven developer with 4+ years of experience building AI solutions—from drone‑based "
            "weed detection to real‑time object detection on edge devices. I blend strong programming skills "
            "with data analysis, MLOps, and team leadership to deliver scalable, high‑impact systems."
        )

    elif "skills" in input_lower or "technical skills" in input_lower:
        return (
            "Technical proficiency:\n"
            "- **Programming & Scripting**: Python, C/C++, C#, Java, PHP, JavaScript, SQL, HTML/CSS\n"
            "- **Machine Learning & CV**: scikit‑learn, TensorFlow, Keras, PyTorch, OpenCV, YOLO, OpenVINO\n"
            "- **NLP & Transformers**: Hugging Face, spaCy, NLTK\n"
            "- **Reinforcement Learning & Optimization**: Stable Baselines3, RLlib, OR‑Tools\n"
            "- **MLOps & Cloud**: Docker, Kubernetes, MLflow, Airflow, Kubeflow, AWS (S3, EC2, SageMaker)\n"
            "- **Databases & Big Data**: PostgreSQL, MongoDB, Apache Spark"
        )

    elif "soft skills" in input_lower or "personal skills" in input_lower:
        return (
            "Soft skills:\n"
            "- Effective communicator and presenter\n"
            "- Team leadership and mentorship\n"
            "- Agile/SCRUM methodologies\n"
            "- Problem-solving and analytical thinking\n"
            "- Adaptability and continuous learning"
        )

    elif "languages" in input_lower or "idiomas" in input_lower:
        return (
            "Language proficiency:\n"
            "- Portuguese: Native\n"
            "- English: Advanced (TOEFL iBT 100)\n"
            "- Spanish: Intermediate (DELE B2)\n"
            "- French: Beginner (DELF A2)"
        )

    elif "experience" in input_lower or "professional experience" in input_lower:
        return (
            "Professional experience:\n"
            "- **AI & Computer Vision Engineer**, Águila Hub (Remote – São Paulo), Nov/2024 – Present\n"
            "  • Leading R&D for weed and pest detection in aerial imagery; managing a team of 3 engineers.\n"
            "- **AI Intern**, Garza Inteligência Financeira (Remote – São Paulo), Mar/2024 – Nov/2024\n"
            "  • Built time‑series forecasting models for market analytics; reduced forecasting error by 12%.\n"
            "- **Data Analyst Intern**, Itaú‑Unibanco (Belo Horizonte), 2020 – 2021\n"
            "  • Developed dashboards for risk assessment; automated ETL pipelines handling 10M+ records.\n"
            "- **Business Advisor**, Itaú‑Unibanco (Uberlândia), 2021 – 2022\n"
            "  • Advised high‑net‑worth clients; increased portfolio growth by 15% year‑over‑year.\n"
            "- **Freelance Developer**, Jan/2019 – Aug/2023\n"
            "  • Delivered custom web and desktop applications for SMEs and startups; managed full SDLC."
        )

    elif "projects" in input_lower or "your projects" in input_lower:
        return (
            "Highlighted projects:\n"
            "- **Drone‑based Weed Classification**: YOLOv5 pipeline on 4K imagery, 94% accuracy, deployed on AWS Lambda.\n"
            "- **Automated Exam Grading Platform**: Flask + OpenCV OMR solution; national hackathon winner (2023).\n"
            "- **10‑Year Disease Forecasting**: Hybrid ARIMA‑LSTM models on public health datasets; MAPE < 8%.\n"
            "- **Edge‑Optimized Object Detection**: Real‑time inference on Raspberry Pi 4 using OpenVINO; <50ms latency.\n"
            "- **Chatbot for Student Support**: Built with Rasa and Python, integrated into UFV LMS, handling 200+ daily queries."
        )

    elif "education" in input_lower or "formation" in input_lower:
        return (
            "Education:\n"
            "- **B.Sc. in Information Systems**, Federal University of Viçosa (UFV), 2022–2026 (expected)\n"
            "- **Technical Diploma**, Systems Analysis & Development, SENAI – CETEL, 2021\n"
            "- **Postgraduate Certificate**, Robotics, CEFET, 2022"
        )

    elif "certifications" in input_lower or "certified" in input_lower:
        return (
            "Certifications:\n"
            "- TensorFlow Developer Certificate (2024)\n"
            "- AWS Certified Machine Learning – Specialty (2023)\n"
            "- Professional Scrum Master I (2022)\n"
            "- Cisco CCNA Routing & Switching (2021)\n"
            "- Microsoft Azure AI Fundamentals (2023)"
        )

    elif "awards" in input_lower or "honors" in input_lower:
        return (
            "Awards & Honors:\n"
            "- 1st Place, UFV National Hackathon (2023)\n"
            "- Scholarship for Academic Excellence, UFV (2022 & 2023)\n"
            "- Best Undergraduate Research Poster, Brazilian AI Symposium (2021)\n"
            "- Dean’s List, UFV (2022–2024)"
        )

    elif "publications" in input_lower or "papers" in input_lower:
        return (
            "Publications & Talks:\n"
            "- “Deep Learning for Real‑Time Weed Detection,” presented at Brazilian CV Conference (2024).\n"
            "- Co‑author of “LSTM‑ARIMA Hybrid Models for Disease Forecasting,” Journal of AI Research (2023).\n"
            "- Guest speaker on AI in agriculture, TEDxUFV (2023)."
        )

    elif "conferences" in input_lower or "congress" in input_lower:
        return (
            "Conferences & Workshops:\n"
            "- Attendee, NeurIPS (2024)\n"
            "- Workshop organizer, PyData São Paulo (2024)\n"
            "- Panelist on “Edge AI” at Latin America AI Summit (2023)"
        )

    elif "volunteer" in input_lower or "community" in input_lower:
        return (
            "Volunteer & Leadership:\n"
            "- Mentor, UFV AI & Robotics Club (2023–Present)\n"
            "- Volunteer Instructor, Code for Brazil Workshops (2022)\n"
            "- Organizer, São Paulo Tech Meetup on Computer Vision (2024)"
        )

    elif "interests" in input_lower or "hobbies" in input_lower:
        return (
            "Interests & Hobbies:\n"
            "- Drone photography and aerial mapping\n"
            "- Competitive chess (Candidate Master)\n"
            "- Cycling, hiking, and outdoor exploration\n"
            "- Reading sci‑fi, tech blogs, and academic papers\n"
            "- Blogging about AI on Medium: @pedro_ai_insights"
        )

    elif "contact" in input_lower or "how to reach you" in input_lower:
        return (
            "Contact Information:\n"
            "- 📍 Rio Paranaíba, MG, Brazil\n"
            "- ✉️ pedrocampos6388@gmail.com\n"
            "- 📞 +55 31 98272‑3063\n"
            "- 🔗 LinkedIn: linkedin.com/in/pedro-campos-5760a92ab\n"
            "- 📂 GitHub: github.com/JFcamp\n"
            "- 🌐 Portfolio: pedrocampos.dev"
        )

    # Se nenhuma regra acima se aplicar, use o modelo para gerar uma resposta
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response