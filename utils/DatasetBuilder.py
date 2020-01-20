import json
from typing import List
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer
import numpy as np


class DatasetBuilder:
    __headlines: List[str] = []
    __sarcastic: List[int] = []
    __max_length: int = 50

    def parse_json(self, path: str) -> None:
        """
        This function will parse and load the json file to internal
        variable
        :param path: str
        :return: None
        """
        with open(path) as jsonfile:
            json_data = json.loads(jsonfile.read())
            for json_line in json_data:
                self.__headlines.append(json_line['headline'])
                self.__sarcastic.append(json_line['is_sarcastic'])

    @staticmethod
    def __get_tokenizer() -> BertTokenizer:
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def create_dataset(self):
        if len(self.__headlines) == 0 or len(self.__sarcastic) == 0:
            raise Exception('Please create headlines and sarcastic list by running parse_json')

        x_inputs_ids, x_token_type_ids, x_attention_mask, y_train = [], [], [], []
        tokenizer = self.__get_tokenizer()

        for index, (headline, is_sarcastic) in enumerate(zip(self.__headlines, self.__sarcastic)):
            categorical = to_categorical(is_sarcastic, 2)

            output = tokenizer.encode_plus(headline, max_length=self.__max_length)

            input_length: int = int(len(output["input_ids"]))
            _attention_mask = [1] * len(output["input_ids"]) + ([0] * (self.__max_length - input_length))

            _input_ids = pad_sequences([output["input_ids"]], maxlen=self.__max_length, padding='post',
                                       truncating='post')
            _token_type_ids = pad_sequences([output["token_type_ids"]], maxlen=self.__max_length,
                                            padding='post',
                                            truncating='post')

            x_inputs_ids.append(_input_ids[0])
            x_token_type_ids.append(_token_type_ids[0])
            x_attention_mask.append(_attention_mask)
            y_train.append(categorical)

        return np.asarray(x_inputs_ids), np.asarray(x_token_type_ids), np.asarray(x_attention_mask), np.asarray(y_train)
