import numpy as np
import pandas as pd
import config
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import text,sequence
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, GlobalMaxPooling1D, GRU
from tensorflow.keras.layers import Embedding,Dense,SpatialDropout1D,concatenate,Input,GlobalAveragePooling1D


def get_model():
    embedding_matrix = np.array(pd.read_csv('crawl(100k,300).csv'))
    inp = Input(shape=(config.MAX_LEN, ))
    x = Embedding(config.VOCAB_SIZE, config.EMBEDDING_SIZE, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    drop1 = Dropout(0.3)(conc)
    # dense1 = Dense(64,activation='relu')(drop1)
    # drop2 = Dropout(0.1)(dense1)
    outp = Dense(4, activation="sigmoid")(drop1)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def convert_tokenizer_to_json(tokenizer):
    import json
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def getVectors(x):
    tok = text.Tokenizer(config.VOCAB_SIZE)
    tok.fit_on_texts(x)
    convert_tokenizer_to_json(tok)
    seq = tok.texts_to_sequences(x)
    padded_text = sequence.pad_sequences(seq,maxlen=config.MAX_LEN)
    return padded_text




if __name__=='__main__':
    train = pd.read_excel('Data_Train.xlsx')
    test = pd.read_excel('Data_Test.xlsx')
    X = train['STORY']
    y = pd.get_dummies(train['SECTION'])
    vec = getVectors(X)
    print(vec.shape)
    X_train,X_test,y_train,y_test = train_test_split(vec,y,test_size=0.3,random_state=42)
    #print(X_train.shape)
    model = get_model()
    model.summary()
    model.fit(X_train,y_train,validation_split=0.3,batch_size=config.BATCH_SIZE,epochs=config.EPOCHS)
    joblib.dump(model,'classifier.sav')
    
