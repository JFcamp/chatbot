import streamlit as st
from chatbot import generate_response

# Configurações da página
st.set_page_config(page_title="Chatbot de IA", layout="centered")

# Título e descrição
st.markdown("""
## 🤖 Como esse chatbot funciona?
1. **Regras Manuais**: Para perguntas específicas sobre Pedro, como formação e projetos, o chatbot usa regras pré-definidas.
2. **Modelo de IA**: Para outras perguntas, ele usa o modelo **DialoGPT** (treinado pela Microsoft) para gerar respostas naturais.
3. **Processo de NLP**: O texto é tokenizado, analisado e respondido com base em padrões aprendidos durante o treinamento do modelo.
""")

# Histórico de conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do usuário
if prompt := st.chat_input("Digite sua pergunta aqui..."):
    # Adicionar pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gerar resposta do chatbot
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Botão para limpar histórico
if st.button("Limpar Conversa"):
    st.session_state.messages = []