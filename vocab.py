from fastai.vision.all import *


_vocab = list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~°ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯự̀́̃̉ẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ')

class Vocab(Transform):
    def __init__(self, chars=_vocab, max_padding=64):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3
        self.na = 4
        self.max_padding = max_padding
        self.chars = chars

        self.c2i = {c:i+5 for i, c in enumerate(chars)}

        self.i2c = {i+5:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'
        self.i2c[4] = '<na>'
        
    def encodes(self, chars):
        res = Munch()
        input_ids = [self.go] + [self.c2i.get(c, self.na) for c in chars] + [self.eos]
        attention_mask = [1] * len(input_ids)
        input_ids += [self.pad] * (self.max_padding - len(input_ids))
        attention_mask = [0] * (self.max_padding - len(input_ids))
        
        res.input_ids = input_ids
        res.attention_mask = attention_mask
        return res
    
    def decodes(self, ids):
        if not isinstance(ids, list):
            ids = ids.tolist()
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def n_classes(self):
        return len(self)

    def __len__(self):
        return len(self.c2i) + 5
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars