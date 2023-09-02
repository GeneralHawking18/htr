import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import ResNet, BasicBlock
import torch
from torchvision.models.efficientnet import (
    EfficientNet, 
    efficientnet_b4, 
    EfficientNet_B4_Weights
)
import einops 
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models.convnext import * # ConvNeXt_Base_Weights, convnext_base


class CRNNv6Att(nn.Module): 
    def __init__(self, num_classes, img_width=128, img_height=32): 
        super().__init__()
        # Lớp conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Lớp conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Lớp conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Lớp conv4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Lớp conv5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512, momentum = 0.2)

        # Lớp conv6
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512, momentum = 0.2)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))

        # Lớp conv7
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2)
        
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p = 0.25),
        )
        
        self.att = SelfAttention(31)
        
        
        # Lớp RNNs
        self.att_gru1 = AttnDecoderRNN2(512, 512)
        # self.att_gru2 = AttnDecoderRNN2(512, 256)
        self.gru2 = nn.GRU(
            input_size = 512,
            hidden_size = 128,
            bidirectional = True,
            dropout = 0.25
            
        )
        self.lstm1 = nn.LSTM(
            input_size = 512,
            hidden_size = 128,
            bidirectional=True,
            dropout = 0.25,
            # num_layers = 2,
        )
            
        self.lstm2 = nn.LSTM(
            input_size = 256,
            dropout = 0.25,
            hidden_size = 128,
            bidirectional=True,
        )
    
        
        
        # Lớp đầu ra
        self.fc = nn.Linear(256, num_classes)
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        self._init_weights()

    def _init_weights(self) -> None:
        # for module in [self.cnn[0][0], self.map_to_seq, self.rnn1, self.rnn2, self.dense]:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool6(x)
        
        x = self.conv7(x)
        x = self.relu(x)

        _, c, h, w = x.shape

        
        x = x.reshape(-1, c*h, w)
        # x = x.permute(0, 2, 1)
        # x = self.proj(x)
        x = x.permute(0, 2, 1)
        x = self.att(x)
        x = x.permute(1, 0, 2)
        
        # x = self.proj(x)
        # print("before permute: ", x.shape)
        # print("after permute: ", x.shape)
        # x, _, _= self.att_gru1(x)
        # x = x.permute(1, 0, 2)
        # print("shape 1st gru: ", x.shape)
        # x, _, _= self.att_gru2(x)
        # x = x.permute(1, 0, 2)
        # print("shape 2nd gru: ", x.shape)
        # x, _ = self.gru2(x)
        # 
        # print(x.shape)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        x = self.fc(x)
        # x = self.log_softmax(x)
        return x
    

class SelfAttention(nn.Module):
    def __init__(self, timestep):
        super().__init__()
        self.proj = nn.Linear(timestep, timestep)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, inputs):
        input_dim = inputs.shape[2]
        a = inputs.permute(0, 2, 1)
        a = self.proj(a)
        a = self.softmax(a)
        a = a.mean(dim = 1)
        a = a.repeat(input_dim, 1, 1).permute(1, 2, 0)
        out_att_mul = inputs * a
        return out_att_mul

def attention_rnn(inputs):
 
    input_dim = inputs.shape[2]
    timestep = inputs.shape[1]

    a = inputs.permute(0, 2, 1) # Permutes the dimensions of the input according to a given pattern.
    print("after permute ", a.shape)
    a = F.softmax(nn.Linear(timestep, timestep)(a), dim=-1) # Alignment Model + Softmax

    a = torch.mean(a, dim=1) # Dim reduction
 
    a = a.repeat(input_dim, 1, 1).permute(1, 2, 0) # Repeat vector


    output_attention_mul = inputs * a # Weighted Average
    return output_attention_mul

class BahdanauAttention2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # print("query, keys: ", query.shape, keys.shape)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN2(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention2(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_size = hidden_size

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)
        # print("decoder_hidden: ", decoder_hidden.shape)
        decoder_outputs = []
        attentions = []

        for i in range(seq_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs.permute(1, 0, 2), decoder_hidden, attentions
    def forward_step(self, hidden, encoder_outputs):
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
         # torch.cat((embedded, context), dim=2)
        # print("input_gru, embedded: ", input_gru.shape, embedded.shape)
        # print("context, hidden: ", context.shape, hidden.shape)
        output, hidden = self.gru(context, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size).to(device)
        self.Ua = nn.Linear(hidden_size, hidden_size).to(device)
        self.Va = nn.Linear(hidden_size, 1).to(device)

    def forward(self, query, keys):
        # print(keys.shape) # 31 x 1024 x 512
        
        # keys = keys.unsqueeze(dim = 1).repeat_interleave(query.size(0), dim = 1) # 31 x 4 x 32 x 512
        # print(keys.shape)
        # query: 1 x 1024 x 512
        # print("keys shape: ", keys.shape)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        #print("scores: ", scores.shape)
        
        # scores = scores.squeeze(-1).unsqueeze(0)
        
        # print("scores: ", scores.shape) # 31 x 4 x 32 x 1
        
        weights = F.softmax(scores, dim=0)
        context = einops.einsum(weights, keys, 'l b one, l b f  -> one b f')
        # context = torch.bmm(weights, keys) # 1 x 4 x 32 x 512
        # print("context, weights: ", context.shape, weights.shape)
        return context, weights.squeeze(dim = -1)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout = 0.1):
        super(AttnDecoderRNN, self).__init__()
        # self.embedding = nn.Embedding(output_size, hidden_size)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.max_len = max_len
        
        self.attention = BahdanauAttention(hidden_size)
        # self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        # Lớp RNNs
        self.lstm = nn.LSTM(
            input_size = 512,
            hidden_size = 512,
            # bidirectional=True,
            dropout = dropout,
        )

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(1)
        seq_len = encoder_outputs.size(0)
        # decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = (
            torch.zeros(1, batch_size, self.hidden_size).to(device),
            torch.zeros(1, batch_size, self.hidden_size).to(device),
        )
        
        decoder_outputs = []
        attentions = []
        
        

        for i in range(seq_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.stack(attentions, dim=0)

        return decoder_outputs# , decoder_hidden, attentions


    def forward_step(self, hidden, encoder_outputs):
        context, attn_weights = self.attention(hidden[0], encoder_outputs)
      
        # context = context.mean(dim = 0, keepdim = True)
        # print("context: ", context.shape)
        # hidden = (hidden[:1], hidden[1:])
        # print(hidden)
        # print("hidden: ", hidden[0].shape, hidden[1].shape)
        output, hidden = self.lstm(context, hidden)
        # print("output, hidden: ", output.shape)
        output = self.out(output)
        # print(output.shape)
        return output, hidden, attn_weights

class CustomizedResNet(ResNet):
    def __init__(self):        
        super().__init__(BasicBlock, [3, 4, 6, 3])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = self.fc(x)

        return x
    

class CustomizedEffNet(EfficientNet):
    def __init__(self):        
        super().__init__(BasicBlock, [2, 2, 2, 2])
        
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return x

class CRNNv5(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.blstm = BidirectionalLSTM(input_size, hidden_size, 
                                       output_size, num_layers)
        self.fe = FeatureExtractor()

    def forward(self, x):
        # print("fe shape: ", x.shape)
        out = self.fe(x)
        out = self.blstm(out)
        return out

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), 
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), 
                         self.hidden_size).to(device)

        outputs, hidden = self.lstm(x, (h0,c0))
        outputs = torch.stack([self.fc(outputs[i])
                                for i in range(outputs.shape[0])])
        outputs = self.softmax(outputs)
        return outputs
                
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(768,64)
#         self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x)))) 
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))
        out = self.maxpool(self.relu(self.bn3(self.conv3(out)))) 

        # print(out.shape)
        out = out.permute(0, 2, 3, 1)
        # print(out.shape)
        out = out.reshape(out.shape[0], -1, out.shape[2]*out.shape[3])
        out = torch.stack([self.relu(self.fc(out[i]))
                                for i in range(out.shape[0])])
        #out = self.dropout(out)
        return out

class _BidirectionalLSTM(nn.Module):

    def __init__(self, inputs_size: int, hidden_size: int, output_size: int):
        super(_BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(x)
        sequence_length, batch_size, inputs_size = recurrent.size()
        sequence_length2 = recurrent.view(sequence_length * batch_size, inputs_size)

        out = self.linear(sequence_length2)  # [sequence_length * batch_size, output_size]
        out = out.view(sequence_length, batch_size, -1)  # [sequence_length, batch_size, output_size]

        return out

class CRNNv4(nn.Module): 
    def __init__(self, num_classes, img_width=128, img_height=32): 
        super().__init__()
        # Lớp conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Lớp conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Lớp conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Lớp conv4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Lớp conv5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512, momentum = 0.2)

        # Lớp conv6
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512, momentum = 0.2)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))

        # Lớp conv7
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2)
        
        self.lstm1 = nn.LSTM(
            input_size = 512,
            hidden_size = 128,
            bidirectional=True,
            dropout = 0.25,
            # num_layers = 2,
        )
            
        self.lstm2 = nn.LSTM(
            input_size = 256,
            dropout = 0.25,
            hidden_size = 128,
            bidirectional=True,
        )
    
        
        # Lớp đầu ra
        self.fc = nn.Linear(256, num_classes)
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        self._init_weights()

    def _init_weights(self) -> None:
        # for module in [self.cnn[0][0], self.map_to_seq, self.rnn1, self.rnn2, self.dense]:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool6(x)
        
        x = self.conv7(x)
        x = self.relu(x)

        _, c, h, w = x.shape
        x = x.reshape(-1, c*h, w)
        x = x.permute(2, 0, 1)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        x = self.fc(x)
        return x
    
class CRNN2(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 16 * 64

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 8 * 32

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 4 x 16

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 2 x 16

            nn.Conv2d(512, 512, (2, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # image size: 1 x 16
        )
        
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p = 0.3),
        )
        self.recurrent_layers = nn.Sequential(
            _BidirectionalLSTM(512, 256, 256),
            _BidirectionalLSTM(256, 256, num_classes),
        )

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Feature sequence
        features = self.convolutional_layers(x)  # [b, c, h, w]
        features = features.squeeze(2)  # [b, c, w]
        features = features.permute(2, 0, 1)  # [w, b, c]

        # Deep bidirectional LSTM
        out = self.recurrent_layers(features)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)



class CRNNv1(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class, path_pretrain,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super().__init__()

        
        # self.cnn, (output_channel, output_height, output_width) = \
            # self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
        # print("out_c, out_h, out_w", output_channel, output_height, output_width)
        output_channel, output_height, output_width = 128, 4, 13
        self.cnn = CustomizedResNet()
        for param in self.cnn.parameters():
            param.requires_grad = False
        # PATH = "/content/resnet18-5c106cde.pth" 
        self.cnn.load_state_dict(torch.load(path_pretrain))
    
        # self.cnn.layer2 = nn.Identity()
        # self.cnn.layer3 = nn.Identity()
        # self.cnn.layer4 = nn.Identity()
        self.cnn.avgpool = nn.Identity()
        self.cnn.fc = nn.Identity()

        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

        self._initialize_weights()

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)
        
        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        # print("shape images: ", images.shape)
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        print("after conv: ", conv.shape)


        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        
        print("after permute: ", conv.shape)
        seq = self.map_to_seq(conv)
        print("seq shape: ", seq.shape)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

    def _initialize_weights(self) -> None:
        for module in [self.cnn.conv1, self.map_to_seq, self.rnn1, self.rnn2, self.dense]:
        # for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class CRNNv2(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super().__init__()

        
        # self.cnn, (output_channel, output_height, output_width) = \
            # self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
        # print("out_c, out_h, out_w", output_channel, output_height, output_width)
        output_channel, output_height, output_width = 128, 4, 13
        """self.cnn = CustomizedResNet()
        for param in self.cnn.parameters():
            param.requires_grad = False
        PATH = "/content/resnet18-5c106cde.pth"
        self.cnn.load_state_dict(torch.load(PATH))"""
        eff_net_b4 = efficientnet_b4(EfficientNet_B4_Weights)
        self.cnn = eff_net_b4.features


        # self.cnn.layer3 = nn.Identity()
        # self.cnn.layer4 = nn.Identity()
        # self.cnn.avgpool = nn.Identity()
        # self.cnn.fc = nn.Identity()

        self.cnn[0][0] = nn.Conv2d(1, 48, kernel_size=7, stride=2, padding=3, bias=False)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

        self._initialize_weights()

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)
        
        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        # print("shape images: ", images.shape)
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        print("after conv: ", conv.shape)


        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        
        print("after permute: ", conv.shape)
        seq = self.map_to_seq(conv)
        print("seq shape: ", seq.shape)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

    def _initialize_weights(self) -> None:
        for module in [self.cnn[0][0], self.map_to_seq, self.rnn1, self.rnn2, self.dense]:
        # for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
class CRNNv3(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super().__init__()

        
        # self.cnn, (output_channel, output_height, output_width) = \
            # self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
        # print("out_c, out_h, out_w", output_channel, output_height, output_width)
        
        output_channel, output_height = 512, 6
        # convnext =  convnext_base(ConvNeXt_Base_Weights).features # convnext_tiny(ConvNeXt_Tiny_Weights).features
        convnext = convnext_base(ConvNeXt_Base_Weights).features
        convnext[6][1] = nn.Identity()
        for i in range(7, 8):
            convnext[i] = nn.Identity()
        #
        for param in convnext.parameters():
            param.requires_grad = False
        convnext[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        for i in range(6, 8):
            convnext[i] = nn.Identity()
            
        # convnext[4][1] = nn.Identity() 
        self.cnn = convnext

        # self.cnn.layer3 = nn.Identity()
        # self.cnn.layer4 = nn.Identity()
        # self.cnn.avgpool = nn.Identity()
        # self.cnn.fc = nn.Identity()

        

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

        self._initialize_weights()

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1), stride = (2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)
        
        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        # print("shape images: ", images.shape)
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        # print("after conv: ", conv.shape)


        conv = conv.reshape(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        
        # print("after permute: ", conv.shape)
        seq = self.map_to_seq(conv)
        # print("seq shape: ", seq.shape)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

    def _initialize_weights(self) -> None:
        for module in [self.cnn[0][0], self.map_to_seq, self.rnn1, self.rnn2, self.dense]:
        # for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

