import streamlit as st
from io import BytesIO
import pytesseract
from PIL import Image
import cv2
import librosa
import torch
from langchain import PromptTemplate
from langchain.chains import LLMChain
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
import google.generativeai as genai
import os
import PIL.Image

load_dotenv()


GOOGLE_API_KEY='YOUR_API_KEY'
genai.configure(api_key=GOOGLE_API_KEY)

#preview = st.text_area("File Preview", "", height=150, key="preview")
#upload_state = st.text_area("Upload State", "", key="upload_state")
#predict_state = st.text_area("Predict State", "", key="predict_state")

def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio)

def handleInput(data, extension,tokenizer, model, imageModel):
    extractedText = ''
    if extension == 'txt':
        extractedText = data.decode('utf-8')
    elif extension == 'audio':
        #print('Its an audio')
        temp_file_to_save = './temp_audio.wav'
        write_bytesio_to_file(temp_file_to_save, data)
        audio, rate = librosa.load(temp_file_to_save, sr = 16000)
        input_values = tokenizer(audio, return_tensors = "pt").input_values
        logits = model(input_values).logits
        prediction = torch.argmax(logits, dim = -1)
        #print(prediction)
        extractedText = tokenizer.batch_decode(prediction)[0]
    elif extension == 'img':
        extractedText = ''
        temp_file_to_save = './temp_image.jpg'
        write_bytesio_to_file(temp_file_to_save, data)
        img = PIL.Image.open('./temp_image.jpg')
        print(type(img))
        extractedText = imageModel.generate_content(img).text
    elif extension == 'video':
        extractedText = ''
        temp_file_to_save = './temp_video.mp4'
        write_bytesio_to_file(temp_file_to_save, data)
        #font_scale = 1.5    
        #font = cv2.FONT_HERSHEY_PLAIN

        cap =cv2.VideoCapture(temp_file_to_save)

        if not cap.isOpened():
            cap =cv2.VideoCapture(write_bytesio_to_file)
        if not cap.isOpened():
            raise IOError("Cannot open video")
            
        counter= 0
        while True:
            ret,frame=cap.read()
            counter +=1;
            if ((counter%20)==0):
                
                converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(converted)
                #imgchar = pytesseract.image_to_string(frame)
                #print(imgchar.strip())
                response = imageModel.generate_content(pil_im)
                print(response.text.strip())
                extractedText = extractedText.strip() + ' ' + response.text.strip()
                
                #imgboxes = pytesseract.image_to_boxes(frame)
                
                #for boxes in imgboxes.splitlines():
                #    boxes = boxes.split(' ')
                #    x,y,w,h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                #    cv2.rectangle(frame, (x, imgH-y), (w,imgH-h), (0,0,255),3)
                    
                #cv2.putText(frame, imgchar, (x1 + int(w1/50), y1 + int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                
                #font = cv2.FONT_HERSHEY_SIMPLEX
                
                #cv2.imshow('Text Detection Tutorial',frame)
                
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
            if not ret:
                break
                       
        cap.release()
        cv2.destroyAllWindows()

    return extractedText.strip()

def display():
    for (i, message) in enumerate(st.session_state.conversation):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
            
def upload(uploaded_file, extension, model, audioTokenizer, audioModel, imageModel):
    
    if uploaded_file is None or extension == 'unknown':
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        data = uploaded_file.getvalue()
        parsedData = handleInput(data, extension, audioTokenizer, audioModel, imageModel)

        template = """ Tell me in two lines about {}
        """
        response = model.generate_content([template.format(parsedData)])
        
        #output = chain.run(parsedData)
        st.session_state.conversation.extend([parsedData, response.text])
        display()
        


def main():
    
    model = genai.GenerativeModel(model_name = "gemini-pro")
    imageModel = genai.GenerativeModel('gemini-pro-vision')
    #pipeline = pipeline(
    #    "text2text-generation",
    #    model=flanModel, 
    #    tokenizer=flanTokenizer, 
    #    max_length=1024
    #)

    #local_llm = HuggingFacePipeline(pipeline=pipeline)

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    audioTokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    audioModel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    st.set_page_config(page_title="MY ANSWER TO GEMINI",
                        page_icon="🧊")
    st.write("""
    # MY ANSWER TO GEMINI
    ### Multi Modality BOT
    """)
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = []


    uploaded_file = st.file_uploader("Choose a file")
    extension  = 'unknown'
    parsedData = ''
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        
        if(uploaded_file.name.endswith('txt')):
            extension = 'txt'
            st.session_state["preview"] = "Its a text"
        elif (uploaded_file.name.endswith('wav')):
            extension = 'audio'
            st.session_state["preview"] = "Its an audio"
        elif (uploaded_file.name.endswith('mp4')):
            extension = 'video'
            st.session_state["preview"] = "Its a video"
        elif (uploaded_file.name.endswith('jpg')):
            extension = 'img'
            st.session_state["preview"] = "Its an image"
        else:
            print("Unsupported media type")
            st.session_state["preview"] += "Its an Unsupported media type"

    st.button("Ask me something in any modality by uploading file", on_click=upload(uploaded_file, extension, model, audioTokenizer, audioModel, imageModel))

if __name__ == '__main__':
    main()