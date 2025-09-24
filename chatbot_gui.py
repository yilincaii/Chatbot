import numpy as np
import pickle
import json
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model, load_model
from tkinter import *

# Load models
classification_model = load_model('classification_model.h5')
seq2seq_model = load_model('seq2seq_model.h5')

# Load saved parameters
save_params = pickle.load(open('save_params_seq2seq.pkl','rb'))
save_params_classification = pickle.load(open('save_params_classification.pkl','rb'))

# Load language processing params
tokenizer = save_params['tokenizer']
reverse_tokenizer_word_index = save_params['reverse_tokenizer_word_index']
maxlen_questions = save_params['maxlen_questions']
maxlen_answers = save_params['maxlen_answers']
EMBEDDING_DIM = save_params['EMBEDDING_DIM']
START_TAG = save_params['START_TAG']
END_TAG = save_params['END_TAG']
categories_list = save_params_classification['categories_list']  
class_tokenizer = save_params_classification['tokenizer']

# Load seq2seq model params
encoder_inputs = seq2seq_model.layers[0].input
encoder_states = seq2seq_model.layers[4].output[1:]
decoder_inputs = seq2seq_model.layers[1].input
decoder_lstm = seq2seq_model.layers[5]
decoder_embedding = seq2seq_model.layers[3].output
decoder_dense = seq2seq_model.layers[6]

encoder_model = Model(encoder_inputs, encoder_states) 
decoder_state_input_h = Input(shape=(EMBEDDING_DIM ,))
decoder_state_input_c = Input(shape=(EMBEDDING_DIM ,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def sentence_parsing(tokenizer, tokenized_sentence):
  bag = []
  for cur_word in tokenizer.word_index:
    bag.append(1) if tokenizer.word_index[cur_word] in tokenized_sentence else bag.append(0)
  return bag

def chatbot_response(question):
  # Parse user input question
  tokenized_question = tokenizer.texts_to_sequences([question])
  parsed_q_input = sequence.pad_sequences(tokenized_question , maxlen=maxlen_questions , padding='post')

  # Make inference with seq2seq model
  states_values = encoder_model.predict(parsed_q_input)
  empty_target_seq = np.zeros((1,1))
  empty_target_seq[0, 0] = tokenizer.word_index[START_TAG]
  decoded_translation = ''

  while True:
      dec_outputs , h , c = decoder_model.predict([empty_target_seq] + states_values)
      sampled_word_index = np.argmax( dec_outputs[0, 0, :] ) 
      if sampled_word_index in reverse_tokenizer_word_index:
          word = reverse_tokenizer_word_index[sampled_word_index]
          if word == END_TAG or len(decoded_translation.split()) >= maxlen_answers:
            break
          else:
            decoded_translation += ' {}'.format( word )
      empty_target_seq[ 0 , 0 ] = sampled_word_index
      states_values = [ h , c ]

  return decoded_translation

def get_question_category(question):
  # parse input question
  tokenized_question = class_tokenizer.texts_to_sequences([question])
  parsed_question = sentence_parsing(class_tokenizer, tokenized_question[0])

  # predict question category
  model_output = classification_model.predict(np.array([parsed_question]))
  category_result = categories_list[np.argmax(model_output)]

  return category_result

def process_question():
    global input_box
    msg = input_box.get("1.0",'end-1c').strip()
    input_box.delete("0.0",END)

    if msg != '':
        chat_context.tag_config('bot', foreground="blue")
        chat_context.tag_config('user', foreground="red")
        chat_context.config(state=NORMAL)
        chat_context.insert(END, "You: " + msg + '\n\n', 'user')
        chat_context.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        category = get_question_category(msg)
        chat_context.insert(END, "Ace: " + "[" + category + "]" + res + '\n\n', 'bot')
        
        chat_context.config(state=DISABLED)
        chat_context.yview(END)

# Build GUI components
root = Tk()
root.title("Chatbot @AceCodeInterview")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
chat_context = Text(root, bd=0, bg="white", height="8", width="50")
chat_context.config(state=DISABLED)

# Bind scrollbar to Chat window
scroll_bar = Scrollbar(root, command=chat_context.yview, cursor="heart")
chat_context['yscrollcommand'] = scroll_bar.set

#Create the box to enter message
input_box = Text(root, bd=0, bg="white",width="29", height="5")

#Create Button to send message
send_button = Button(root, text="Send", width="12", height=5,
                    bd=0, highlightbackground='skyblue',
                    command= process_question)

# Place all components on the screen
scroll_bar.place(x=376,y=6, height=386)
chat_context.place(x=6,y=6, height=386, width=370)
input_box.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)

root.mainloop()
