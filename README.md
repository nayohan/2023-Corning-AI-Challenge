
<div align="center">
    <a href="https://github.com/CrayLabs/SmartSim"><img src="./app/assets/logo-glass-bg.png" width="30%"><img></a>
</div>

# Corning AI : Multi-turn 대화가 가능한 대화 챗봇 📖

Welcome to **ClosedAI-chatbot**. 해당 repo는 LangChain과 ChromaDB를 활용하여 대화를 가능하게 합니다. 

## 📖 How it Works

아래 그림은 **"ClosedAI-chatbot"** 가 작동하는 과정을 보여줍니다. 사용자는 질문하고자하는 대화를 입력합니다. 사용자 질의와 관련한 논문들은 더 작은 크기로 청킹(chunking)되어 임베딩으로 생성되어 ChromaDB에 저장됩니다. ClosedAI-chatbot은 사용자의 질의에 대해 관련 문서를 기반으로 답변을 반환합니다.

![ref arch](app/assets/lanchain.webp)


## 🛠 Components

1. **LangChain's ArXiv Loader**: PaperDB에 대화에 필요한 문서를 기반으로 불러옵니다.
2. **Chunking + Embedding**: LangChain을 사용하여 논문을 임의의 사이즈로 분할한 다음 임베딩을 생성합니다.
3. **ChromaDB**: RAG를 위한 효율적인 벡터 저장, 인덱싱 및 검색을 시연합니다.
4. **RetrievalQA**: LangChain의 Retrieval QA 및 Local LLM을 기반으로 사용자는 제출한 주제별로 검색된 논문에 대한 쿼리를 작성할 수 있습니다.
    
    1) Retrieval 모델
       - 문서 임베딩을 위한 모델 : "BAAI/bge-base-en-v1.5"

            아래와 같은 흐름으로 Retrieval 모델을 활용하여 문서를 임베딩합니다.
        ![ref arch](app/assets/chunk.webp)
       ![ref arch](app/assets/fe2a8d84-2d2e-4e0f-b5a2-24e7b0bf33c7_image.webp)
        
    2) LLM 모델
       - 화학 도메인에 학습한 LLM 모델 : "nayohan/corningQA-llama2-13b-chat"
       - 학습의 전체 프레임워크는 다음과 같습니다.

       <p align="center"><img width="500" alt="image" src="app/assets/LLM_train.png">

        - ClosedAI-chatbot은 multi-turn dialogue를 지원합니다.

        <p align="center"><img width="500" alt="image" src="app/assets/multi_turn.png">

5. **Python Libraries**: Making use of tools such as [`ChromaDB`], [`Langchain`](https://www.langchain.com/), [`Streamlit`](https://streamlit.io/), etc



### Run Locally

1. First, clone this repo and cd into it.
    ```bash
    $ git clone https://github.com/nayohan/2023-Corning-AI-Challenge
    ```

2. Create your env file:
    ```bash
    $ conda create -n <env_name> python=3.10
    ```

3. Install dependencies:
    You should have Python 3.10+ installed and a virtual environment set up.
    ```bash
    $ pip install -r requirements.txt
    ```

4. 대화 주제에 관련하여 필요한 문서를 해당 경로에 업로드 해줍니다. **(PDF/Word 지원)**
    ```bash
    $ PaperDB/*.pdf
    ```

5. Run the app:
    ```bash
    $ streamlit run app.py
    ```

6. Navigate to:
    ```
    http://localhost:8501/
    ```
    
### 활용 예시

1. 시작 페이지
    - 토픽 주제에 대해 먼저 입력합니다.

   <p align="center"><img width="800" alt="image" src="app/assets/interface.png">

3. 대화 페이지
    - 질문을 입력하면 답변과 관련 context를 보여줍니다.
   <p align="center"><img width="600" alt="image" src="app/assets/interface1.png">

4. 멀티 턴 대화 예시

   <p align="center"><img width="600" alt="image" src="app/assets/interface2.png">

