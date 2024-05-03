from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import merge, TimeDistributed, Conv1D, Bidirectional


class Generate_model():
    def __init__(self):
        self.model = Sequential()
    def generate_lstm_model(self,n_input, n_out, n_features):
        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu',  input_shape=(n_input, n_features)))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(n_out))
        self.model.summary()
        # 模型编译
        self.model.compile(loss="mse", optimizer='adam')
        return self.model

    def generate_seq2seq_model(self,n_input, n_out, n_features):
        self.model = Sequential()
        self.model.add(LSTM(128,input_shape=(n_input, n_features)))
        self.model.add(Dense(10, activation="relu"))
        # 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
        self.model.add(RepeatVector(1))#此为步长
        # Decoder(第二个 LSTM)
        self.model.add(LSTM(128,return_sequences=True))
        # TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
        self.model.add(TimeDistributed(Dense(units=n_out, activation="linear")))
        self.model.add(Flatten())#扁平层将（None,1,8)变为（None,1*8)
        self.model.summary()
        self.model.compile(loss="mse", optimizer='adam')
        return self.model

    # ------------------------------------------------------------------------------------------------------#
    #   注意力模块，主要是实现对step维度的注意力机制
    #   在这里大家可能会疑惑，为什么需要先Permute再进行注意力机制的施加。
    #   这是因为，如果我们直接进行全连接的话，我们的最后一维是特征维度，这个时候，我们每个step的特征是分开的，
    #   此时进行全连接的话，得出来注意力权值每一个step之间是不存在特征交换的，自然也就不准确了。
    #   所以在这里我们需要首先将step维度转到最后一维，然后再进行全连接，根据每一个step的特征获得注意力机制的权值。
    def attention_block(self,inputs,time_step):
        # batch_size, time_steps, lstm_units -> batch_size, lstm_units, time_steps
        a = Permute((2, 1))(inputs)
        # batch_size, lstm_units, time_steps -> batch_size, lstm_units, time_steps
        a = Dense(time_step, activation='softmax')(a)#和步长有关
        # batch_size, lstm_units, time_steps -> batch_size, time_steps, lstm_units
        a_probs = Permute((2, 1), name='attention_vec')(a)
        # 相当于获得每一个step中，每个特征的权重
        output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        return output_attention_mul

    def generate_attention_model(self, n_input, n_out, n_features):
        inputs = Input(shape=(n_input, n_features,))
        # (batch_size, time_steps, input_dim) -> (batch_size, input_dim, lstm_units)
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        attention_mul = self.attention_block(lstm_out,n_input)
        # (batch_size, input_dim, lstm_units) -> (batch_size, input_dim*lstm_units)
        dropout=Dropout(0.8)(attention_mul)
        flatten = Flatten()(dropout)
        output = Dense(n_out, activation='sigmoid')(flatten)
        model = Model(inputs=[inputs], outputs=output)
        model.summary()
        model.compile(loss="mse", optimizer='adam')
        return model

    def generate_seq2seq_attention_model(self, n_input, n_out, n_features):
        inputs = Input(shape=(n_input, n_features,))
        lstm_out1 = LSTM(128, return_sequences=True)(inputs)
        attention_mul = self.attention_block(lstm_out1, n_input)
        # (batch_size, input_dim, lstm_units) -> (batch_size, input_dim*lstm_units)
        attention_mul = Flatten()(attention_mul)
        output1 = Dense(n_out, activation='sigmoid')(attention_mul)
        repeatVector=RepeatVector(1)(output1)
        lstm_out2 = LSTM(128, return_sequences=True)(repeatVector)
        output2=TimeDistributed(Dense(n_out))(lstm_out2)
        flatten=Flatten()(output2)
        model = Model(inputs=[inputs], outputs=flatten)
        model.summary()
        model.compile(loss="mse", optimizer='adam')
        return model

    def cnn_lstm_attention_model(self, n_input, n_out, n_features):
        inputs = Input(shape=(n_input, n_features))
        x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'
        x = Dropout(0.3)(x)
        # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
        # 对于GPU可以使用CuDNNLSTM
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
        lstm_out = Dropout(0.3)(lstm_out)
        attention_mul = self.attention_block(lstm_out, n_input)
        attention_mul = Flatten()(attention_mul)
        output = Dense(n_out, activation='sigmoid')(attention_mul)
        model = Model(inputs=[inputs], outputs=output)
        model.summary()
        model.compile(loss="mse", optimizer='adam')
        return model

